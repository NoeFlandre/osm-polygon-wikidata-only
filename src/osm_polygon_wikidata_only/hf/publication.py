"""Pure dataset publication assembly.

This module owns the construction of publication-op lists for the
three documented publication contracts. Assemblers return
``list[PublicationOp]`` -- one ``add`` op per local artifact, plus
explicit ``delete`` ops for the legacy paths (augmentation manifest
and coverage map) to migrate the remote layout.

* Legacy core publication
  (called by :func:`cli.commands._enqueue_core_upload`):
    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. README
    8. canonical coverage map (add)
    9. legacy coverage map (delete)

* Unified sync with changed core
  (called by ``cli.run_sync._build_region_publication``):
    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. canonical coverage map (add)
    8. legacy coverage map (delete)
    9. wikipedia documents
    10. wikipedia sections
    11. wikivoyage documents
    12. wikivoyage sections
    13. wikidata facts
    14. canonical augmentation manifest (add)
    15. legacy augmentation manifest (delete)
    16. README

* Augmentation-only publication (legacy
  ``cli.commands._augmentation_upload_files`` behavior):
    1. wikipedia documents
    2. wikipedia sections
    3. wikivoyage documents
    4. wikivoyage sections
    5. wikidata facts
    6. canonical augmentation manifest (add)
    7. legacy augmentation manifest (delete)
    8. README

Canonical remote layout
------------------------
::

  manifests/
    processed_pbfs.json
    augmentation_manifest.json

The legacy ``augmentation/manifests/augmentation_manifest.json``
path is referenced only by the explicitly-named
:data:`osm_polygon_wikidata_only.hf.repo_layout.LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE`
constant and disappears from the remote after the first atomic
migration commit succeeds.

The assembly functions are PURE: each returns the ordered op list
but performs no upload and accepts no ``submit`` callable. CLI code
performs exactly one queue/direct submission after successful
assembly. Failures inside an assembler raise BEFORE any file is
published: required local artifacts are validated at the top of
each entry point, and snapshot generation failures propagate
without being swallowed.

The module owns no HF upload state, no
:class:`BackgroundUploadQueue`, and no CLI concerns. Snapshot
directories and filenames are stable: the legacy-core snapshots
live under ``data_root.cache / "upload_manifest_snapshots"``, the
augmentation-only snapshots live under
``data_root.cache / "augmentation_upload_snapshots"``, and the
unified-sync snapshots live under
``data_root.cache / "sync_upload_snapshots" / <stem>``.

World-land fallback policy is decided by the caller: the legacy
core command logs a warning, the unified sync command swallows the
exception silently. Callers pass a ``warning_callback`` (or
``None`` for the silent policy) to each entry point so the policy
stays with the caller and the publication module never invents a
new logger identity.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf._dataset_stats.augmentation import (
    compute_augmentation_stats,
)
from osm_polygon_wikidata_only.hf._uploader.plan import (
    PublicationOp,
    add_op,
    delete_op,
)
from osm_polygon_wikidata_only.hf.coverage_map import (
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.dataset_stats import (
    compute_dataset_stats,
    render_stats_section,
)
from osm_polygon_wikidata_only.hf.geographic_text_coverage import (
    generate_geographic_polygon_count as _generate_geographic_polygon_count,
)
from osm_polygon_wikidata_only.hf.geographic_text_coverage import (
    generate_geographic_text_coverage as _generate_geographic_text_coverage,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_ARTICLES_DIR,
    LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
    LEGACY_REMOTE_COVERAGE_MAP_FILE,
    REMOTE_AUGMENTATION_MANIFEST_FILE,
    REMOTE_CONTAINMENT_RETIREMENT_FILE,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
    REMOTE_WIKIPEDIA_DOCUMENTS_DIR,
    canonical_region_paths,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.publication")


def _augmentation_migration_ops(
    augmentation_manifest_path: Path,
) -> list[PublicationOp]:
    """Return the augmentation-manifest ops that unify the remote layout.

    Always two ops:

    * ``add`` of the canonical
      ``REMOTE_AUGMENTATION_MANIFEST_FILE`` (whose local source is
      the per-region augmentation-manifest snapshot or the original
      ``augmentation_result.manifest_path``).
    * ``delete`` of the legacy
      ``LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE`` -- safely
      idempotent on every subsequent publication (the remote file
      is already gone).
    """
    return [
        add_op(
            augmentation_manifest_path,
            path_in_repo=REMOTE_AUGMENTATION_MANIFEST_FILE,
        ),
        delete_op(LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE),
    ]


def _legacy_article_retirement_ops(
    *, stem: str, canonical_document_path: Path
) -> list[PublicationOp]:
    """Atomically replace one legacy article object with its canonical document."""
    return [
        add_op(
            canonical_document_path,
            path_in_repo=f"{REMOTE_WIKIPEDIA_DOCUMENTS_DIR}/{stem}.parquet",
        ),
        delete_op(f"{LEGACY_REMOTE_ARTICLES_DIR}/{stem}.parquet"),
    ]


def _snapshot_canonical_document(core: ProcessResult, destination: Path) -> Path:
    """Convert a core article table to the canonical document schema for publication."""
    article_table = pq.read_table(core.articles_path)  # type: ignore[no-untyped-call]
    canonical = build_wikipedia_document_table(article_table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(canonical, destination, compression="snappy")  # type: ignore[no-untyped-call]
    return destination


# ---------------------------------------------------------------------------
# Snapshots: manifest, README, geographic coverage PNGs
# ---------------------------------------------------------------------------


def snapshot_upload_manifests(
    *,
    data_root: DataRoot,
    core: ProcessResult,
) -> tuple[Path, Path]:
    """Build the legacy-core processed-manifest snapshot and return the
    README snapshot destination (but do not yet write the README).

    Returns the ``(manifest_snapshot_path, readme_snapshot_path)`` tuple.
    The README is rendered last by :func:`write_readme_snapshot` after
    every other snapshot has been written, so a partial core upload
    never reaches the Hub.
    """
    snapshots = data_root.cache / "upload_manifest_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / f"{core.polygons_path.stem}.json"
    atomic_write_text(snapshot, core.manifest_path.read_text(encoding="utf-8"))
    card_snapshot = snapshots / f"{core.polygons_path.stem}-README.md"
    return snapshot, card_snapshot


def write_readme_snapshot(
    data_root: DataRoot,
    repo_id: str,
    destination: Path,
) -> None:
    """Render the canonical dataset README from current local artifacts.

    The README is recomputed by:

    1. Aggregating the processed-PBFs manifest counts for the headline
       row.
    2. Computing the core :class:`DatasetStats` snapshot via
       :func:`compute_dataset_stats`.
    3. Computing the augmentation :class:`AugmentationStats` snapshot
       via :func:`compute_augmentation_stats`. The per-file summary
       cache lives under ``data_root.cache``, so a warm refresh
       performs zero Parquet table reads.
    4. Passing both snapshots to :func:`render_stats_section` so the
       rendered card always includes the documented sections -- the
       legacy three sections plus the augmentation coverage,
       Wikipedia and Wikivoyage corpora, Wikidata facts, and storage
       accounting.

    The README must be written AFTER every other snapshot so a
    partial core upload never reaches the Hub. The destination is
    written atomically via
    :func:`osm_polygon_wikidata_only.io.atomic.atomic_write_text`.
    """
    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    aggregate = {
        key: sum(int(entry.get(key, 0)) for entry in entries.values())
        for key in ("polygon_count", "article_count", "unique_wikidata_count")
    }
    core_stats = compute_dataset_stats(data_root.processed)
    augmentation_stats = compute_augmentation_stats(
        data_root.processed,
        cache_index_dir=data_root.cache,
    )
    stats_section = render_stats_section(
        core_stats,
        augmentation_stats=augmentation_stats,
    )
    atomic_write_text(
        destination,
        render_dataset_card(
            repo_id=repo_id,
            stats=aggregate,
            polygon_columns=list(POLYGON_COLUMNS),
            polygon_descriptions=POLYGON_DESCRIPTIONS,
            article_columns=list(ARTICLE_COLUMNS),
            article_descriptions=ARTICLE_DESCRIPTIONS,
            link_columns=list(POLYGON_ARTICLE_COLUMNS),
            link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
            maintainer="Noé Flandre",
            stats_section=stats_section,
        ),
    )


def refresh_coverage_assets(
    *,
    data_root: DataRoot,
    snapshot_stem: str,
    snapshots_dir: Path,
    world_land_warning: Callable[[str], None] | None,
) -> tuple[Path, Path, Path]:
    """Render the three legacy core coverage PNGs into ``snapshots_dir``.

    ``world_land_warning`` controls the world-land fallback policy.
    Pass a logging-like callable (e.g. ``LOGGER.warning``) to record
    a warning when land data is unavailable, or ``None`` to swallow
    the exception silently. The publication module never invents a
    logger identity of its own.

    Returns ``(map_snapshot, geo_text_snapshot, polygon_count_snapshot)``.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    map_snapshot = snapshots_dir / f"{snapshot_stem}-coverage_map.png"
    lons, lats = load_centroids_from_parquet(data_root.processed_polygons)
    try:
        land_path = ensure_world_land(data_root.cache)
    # ``except Exception`` retained: ``ensure_world_land`` performs
    # network I/O via ``urllib.request.urlretrieve`` and filesystem
    # mkdir/stat, raising a broad, unstable set of exception types
    # (``URLError``, ``HTTPError``, ``ContentTooShortError``,
    # ``socket.timeout``, ``OSError``). Documented fallback: render
    # the map without continents + invoke ``world_land_warning`` when
    # not ``None``.
    except Exception:
        if world_land_warning is not None:
            world_land_warning("Could not fetch world land data; map will omit continents")
        land_path = None
    generate_coverage_map(lons, lats, map_snapshot, land_geojson_path=land_path)
    geo_snapshot = snapshots_dir / f"{snapshot_stem}-geographic_text_coverage.png"
    _generate_geographic_text_coverage_snapshot(data_root, geo_snapshot)
    polygon_count_snapshot = snapshots_dir / f"{snapshot_stem}-geographic_polygon_count.png"
    _generate_geographic_polygon_count_snapshot(data_root, polygon_count_snapshot)
    return map_snapshot, geo_snapshot, polygon_count_snapshot


def _generate_geographic_text_coverage_snapshot(
    data_root: DataRoot,
    destination: Path,
) -> Path:
    """Build the geographic Wikipedia text coverage PNG into ``destination``."""
    land_cache = data_root.cache
    result = _generate_geographic_text_coverage(
        data_root.processed,
        destination,
        land_cache_dir=land_cache,
    )
    return result.output_path


def _generate_geographic_polygon_count_snapshot(
    data_root: DataRoot,
    destination: Path,
) -> Path:
    """Build the geographic polygon density PNG into ``destination``."""
    land_cache = data_root.cache
    result = _generate_geographic_polygon_count(
        data_root.processed,
        destination,
        land_cache_dir=land_cache,
    )
    return result.output_path


# ---------------------------------------------------------------------------
# Required-artifact validation (always called by entry points)
# ---------------------------------------------------------------------------


def _validate_core_artifacts(core: ProcessResult | CorePublicationArtifacts) -> None:
    if isinstance(core, CorePublicationArtifacts):
        paths: Sequence[Path] = (
            core.polygons_path,
            core.polygon_articles_path,
            core.manifest_path,
        )
    else:
        paths = (
            core.polygons_path,
            core.articles_path,
            core.polygon_articles_path,
            core.manifest_path,
        )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Core artifact missing before upload: {path}")


def _validate_augmentation_artifacts(augmentation: AugmentationResult) -> None:
    paths: Sequence[Path] = (
        augmentation.wikipedia_documents_path,
        augmentation.wikipedia_sections_path,
        augmentation.wikivoyage_documents_path,
        augmentation.wikivoyage_sections_path,
        augmentation.wikidata_facts_path,
        augmentation.manifest_path,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Augmentation artifact missing before upload: {path}")


# ---------------------------------------------------------------------------
# Coverage refresh decision
# ---------------------------------------------------------------------------


def coverage_refresh_required(core: object | None) -> bool:
    """Coverage assets change only when a core polygon artifact changes."""
    return core is not None


# ---------------------------------------------------------------------------
# Assembly contracts (pure: each returns the ordered op list)
# ---------------------------------------------------------------------------


def assemble_core_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    core: ProcessResult,
    world_land_warning: Callable[[str], None],
) -> list[PublicationOp]:
    """Assemble the legacy core publication op list.

    Returns the ordered list of :class:`PublicationOp` records:

    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. README
    8. canonical coverage map (add)
    9. legacy coverage map (delete)

    The function is pure: no HF upload state is owned here. The
    caller submits the returned list. Required artifacts are
    validated before any snapshot is written, and any snapshot
    failure propagates without producing a partial op list. The
    legacy core publication does NOT touch the augmentation
    manifests directory at all.
    """
    _validate_core_artifacts(core)
    snapshot, card_snapshot = snapshot_upload_manifests(data_root=data_root, core=core)
    map_snapshot, geo_snapshot, polygon_count_snapshot = refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem=core.polygons_path.stem,
        snapshots_dir=data_root.cache / "upload_manifest_snapshots",
        world_land_warning=world_land_warning,
    )
    write_readme_snapshot(data_root, repo_id, card_snapshot)
    canonical_document = _snapshot_canonical_document(
        core,
        data_root.cache
        / "upload_manifest_snapshots"
        / f"{core.articles_path.stem}-wikipedia-documents.parquet",
    )
    return [
        add_op(core.polygons_path, path_in_repo=f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
        *_legacy_article_retirement_ops(
            stem=core.articles_path.stem,
            canonical_document_path=canonical_document,
        ),
        add_op(
            core.polygon_articles_path,
            path_in_repo=f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
        ),
        add_op(snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
        add_op(geo_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        add_op(polygon_count_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        add_op(card_snapshot, path_in_repo="README.md"),
        add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
        delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
    ]


def assemble_region_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | CorePublicationArtifacts | None,
    world_land_warning: Callable[[str], None] | None,
) -> list[PublicationOp]:
    """Assemble one atomic region upload (sync-dir publication).

    File ordering follows the documented contract. When ``core`` is
    provided, the eight core operations are prepended to the eight
    augmentation operations. When ``core`` is ``None``, only the
    augmentation block is produced. Coverage assets are refreshed
    only when ``core`` is not ``None``.

    The augmentation block ALWAYS emits the canonical
    ``add`` op + the legacy ``delete`` op. The first publication
    after the migration picks up the new canonical path and removes
    the legacy object. Subsequent publications are idempotent: the
    delete op affects a path that no longer exists.

    The function is pure: no HF upload state is owned here. The
    caller submits the returned list. Required artifacts are
    validated before any snapshot is written, and any snapshot
    failure propagates without producing a partial op list.
    """
    if core is not None:
        _validate_core_artifacts(core)
    _validate_augmentation_artifacts(augmentation)
    snapshots = data_root.cache / "sync_upload_snapshots" / stem
    snapshots.mkdir(parents=True, exist_ok=True)
    augmentation_manifest_snapshot = snapshots / "augmentation_manifest.json"
    atomic_write_text(augmentation_manifest_snapshot, augmentation.manifest_path.read_text())
    readme_snapshot = snapshots / "README.md"

    ops: list[PublicationOp] = []

    if coverage_refresh_required(core):
        assert core is not None
        processed_manifest_snapshot = snapshots / "processed_pbfs.json"
        atomic_write_text(
            processed_manifest_snapshot,
            (data_root.processed_manifests / "processed_pbfs.json").read_text(),
        )
        map_snapshot = snapshots / "coverage_map.png"
        lons, lats = load_centroids_from_parquet(data_root.processed_polygons)
        try:
            land_path = ensure_world_land(data_root.cache)
        # ``except Exception`` retained: same rationale as the legacy
        # core path -- ``ensure_world_land`` does network I/O via
        # ``urllib.request.urlretrieve`` which raises a broad,
        # unstable set of exception types. The sync path passes
        # ``world_land_warning=None`` (silent fallback).
        except Exception:
            if world_land_warning is not None:
                world_land_warning("Could not fetch world land data; map will omit continents")
            land_path = None
        generate_coverage_map(lons, lats, map_snapshot, land_geojson_path=land_path)
        geographic_text_snapshot = snapshots / "geographic_text_coverage.png"
        _generate_geographic_text_coverage_snapshot(data_root, geographic_text_snapshot)
        geographic_polygon_count_snapshot = snapshots / "geographic_polygon_count.png"
        _generate_geographic_polygon_count_snapshot(data_root, geographic_polygon_count_snapshot)
        ops.extend(
            [
                add_op(
                    core.polygons_path,
                    path_in_repo=f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}",
                ),
                add_op(
                    core.polygon_articles_path,
                    path_in_repo=f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
                ),
                add_op(processed_manifest_snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
                add_op(geographic_text_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
                add_op(
                    geographic_polygon_count_snapshot,
                    path_in_repo=REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
                ),
                add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
                delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
            ]
        )

    ops.extend(
        [
            *_legacy_article_retirement_ops(
                stem=stem,
                canonical_document_path=augmentation.wikipedia_documents_path,
            ),
            add_op(
                augmentation.wikipedia_sections_path,
                path_in_repo=f"wikipedia/sections/{stem}.parquet",
            ),
            add_op(
                augmentation.wikivoyage_documents_path,
                path_in_repo=f"wikivoyage/documents/{stem}.parquet",
            ),
            add_op(
                augmentation.wikivoyage_sections_path,
                path_in_repo=f"wikivoyage/sections/{stem}.parquet",
            ),
            add_op(
                augmentation.wikidata_facts_path,
                path_in_repo=f"wikidata/facts/{stem}.parquet",
            ),
            *_augmentation_migration_ops(augmentation_manifest_snapshot),
            add_op(readme_snapshot, path_in_repo="README.md"),
        ]
    )
    write_readme_snapshot(data_root, repo_id, readme_snapshot)
    return ops


def assemble_augmentation_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    augmentation: AugmentationResult,
) -> list[PublicationOp]:
    """Assemble one augmentation-only publication op list.

    File ordering follows the documented contract:

    1. wikipedia documents
    2. wikipedia sections
    3. wikivoyage documents
    4. wikivoyage sections
    5. wikidata facts
    6. canonical augmentation manifest (add)
    7. legacy augmentation manifest (delete)
    8. README

    No coverage assets are regenerated. No new stem-augmentation
    manifest snapshot is created for this contract: the legacy
    augmentation command uploads the original
    ``augmentation_result.manifest_path`` directly. The README
    snapshot is rendered by this function immediately before
    returning. The function is pure: no HF upload state is owned
    here.
    """
    _validate_augmentation_artifacts(augmentation)
    snapshots = data_root.cache / "augmentation_upload_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    readme_snapshot = snapshots / f"{augmentation.wikipedia_documents_path.stem}-README.md"
    write_readme_snapshot(data_root, repo_id, readme_snapshot)
    return [
        *_legacy_article_retirement_ops(
            stem=augmentation.wikipedia_documents_path.stem,
            canonical_document_path=augmentation.wikipedia_documents_path,
        ),
        add_op(
            augmentation.wikipedia_sections_path,
            path_in_repo=str(augmentation.wikipedia_sections_path.relative_to(data_root.processed)),
        ),
        add_op(
            augmentation.wikivoyage_documents_path,
            path_in_repo=str(
                augmentation.wikivoyage_documents_path.relative_to(data_root.processed)
            ),
        ),
        add_op(
            augmentation.wikivoyage_sections_path,
            path_in_repo=str(
                augmentation.wikivoyage_sections_path.relative_to(data_root.processed)
            ),
        ),
        add_op(
            augmentation.wikidata_facts_path,
            path_in_repo=str(augmentation.wikidata_facts_path.relative_to(data_root.processed)),
        ),
        *_augmentation_migration_ops(augmentation.manifest_path),
        add_op(readme_snapshot, path_in_repo="README.md"),
    ]


class PublicationValidationError(ValueError):
    """Raised when validation of publication artifacts or manifest fails."""


@dataclass(frozen=True, slots=True)
class CorePublicationArtifacts:
    polygons_path: Path
    polygon_articles_path: Path
    wikipedia_documents_path: Path | None
    manifest_path: Path
    stem: str
    manifest_entry: dict[str, Any]


def load_existing_core_artifacts(data_root: DataRoot, stem: str) -> CorePublicationArtifacts:
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    polygon_articles_path = data_root.processed_links / f"{stem}.parquet"
    manifest_path = data_root.processed_manifests / "processed_pbfs.json"

    # Check existence
    if not polygons_path.is_file():
        raise FileNotFoundError(f"Polygons file missing: {polygons_path}")
    if not polygon_articles_path.is_file():
        raise FileNotFoundError(f"Polygon articles links file missing: {polygon_articles_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest missing: {manifest_path}")

    # Check manifest entry
    manifest_data = load_manifest(manifest_path)
    manifest_key = f"{stem}.osm.pbf"
    if manifest_key not in manifest_data:
        raise KeyError(f"Stem {stem} not found in processed manifests")
    entry = manifest_data[manifest_key]

    # Validate entry details using actionable project-specific errors
    if entry.get("source_pbf") != manifest_key:
        raise PublicationValidationError(
            f"Manifest entry source_pbf mismatch for key '{manifest_key}': "
            f"expected '{manifest_key}', got '{entry.get('source_pbf')}'"
        )
    if entry.get("polygons_path") != f"polygons/{stem}.parquet":
        raise PublicationValidationError(
            f"Manifest entry polygons_path mismatch for key '{manifest_key}': "
            f"expected 'polygons/{stem}.parquet', got '{entry.get('polygons_path')}'"
        )
    if entry.get("polygon_articles_path") != f"polygon_articles/{stem}.parquet":
        raise PublicationValidationError(
            f"Manifest entry polygon_articles_path mismatch for key '{manifest_key}': "
            f"expected 'polygon_articles/{stem}.parquet', got '{entry.get('polygon_articles_path')}'"
        )

    # Validate schemas
    from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema

    try:
        poly_arrow_schema = pq.read_schema(polygons_path)  # type: ignore[no-untyped-call]
    except Exception as e:
        raise PublicationValidationError(f"Could not read schema for {polygons_path}: {e}") from e
    if not poly_arrow_schema.equals(polygon_schema(), check_metadata=True):
        raise PublicationValidationError(f"Schema mismatch for polygons parquet: {polygons_path}")

    try:
        links_arrow_schema = pq.read_schema(polygon_articles_path)  # type: ignore[no-untyped-call]
    except Exception as e:
        raise PublicationValidationError(
            f"Could not read schema for {polygon_articles_path}: {e}"
        ) from e
    if not links_arrow_schema.equals(polygon_article_schema(), check_metadata=True):
        raise PublicationValidationError(
            f"Schema mismatch for polygon links parquet: {polygon_articles_path}"
        )

    wikipedia_documents_path: Path | None = None
    wiki_doc_file = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    if wiki_doc_file.is_file():
        if (
            entry.get("wikipedia_documents_path")
            and entry.get("wikipedia_documents_path") != f"wikipedia/documents/{stem}.parquet"
        ):
            raise PublicationValidationError(
                f"Manifest entry wikipedia_documents_path mismatch for key '{manifest_key}': "
                f"expected 'wikipedia/documents/{stem}.parquet', got '{entry.get('wikipedia_documents_path')}'"
            )

        from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
            wikipedia_document_schema,
        )

        try:
            wiki_doc_arrow_schema = pq.read_schema(wiki_doc_file)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise PublicationValidationError(
                f"Could not read schema for {wiki_doc_file}: {e}"
            ) from e
        if not wiki_doc_arrow_schema.equals(wikipedia_document_schema(), check_metadata=True):
            raise PublicationValidationError(
                f"Schema mismatch for wikipedia documents parquet: {wiki_doc_file}"
            )
        wikipedia_documents_path = wiki_doc_file
    else:
        aug_manifest_path = (
            data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        if aug_manifest_path.is_file():
            try:
                aug_manifest = json.loads(aug_manifest_path.read_text(encoding="utf-8"))
                if stem in aug_manifest:
                    raise PublicationValidationError(
                        f"Wikipedia documents file missing for augmented region {stem}: {wiki_doc_file}"
                    )
            except json.JSONDecodeError as exc:
                raise PublicationValidationError(
                    f"Malformed augmentation manifest JSON: {exc}"
                ) from exc

    return CorePublicationArtifacts(
        polygons_path=polygons_path,
        polygon_articles_path=polygon_articles_path,
        wikipedia_documents_path=wikipedia_documents_path,
        manifest_path=manifest_path,
        stem=stem,
        manifest_entry=entry,
    )


def assemble_metadata_only_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    world_land_warning: Callable[[str], None] | None = None,
) -> list[PublicationOp]:
    """Assemble repository-level metadata assets when no region is repaired."""
    snapshots = data_root.cache / "metadata_upload_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    processed_manifest = data_root.processed_manifests / "processed_pbfs.json"
    if not processed_manifest.is_file():
        raise FileNotFoundError("Local processed manifest is missing")

    processed_manifest_snapshot = snapshots / "processed_pbfs.json"
    atomic_write_text(processed_manifest_snapshot, processed_manifest.read_text(encoding="utf-8"))

    augmentation_manifest = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    augmentation_manifest_snapshot = snapshots / "augmentation_manifest.json"
    has_aug_manifest = False
    if augmentation_manifest.is_file():
        atomic_write_text(
            augmentation_manifest_snapshot, augmentation_manifest.read_text(encoding="utf-8")
        )
        has_aug_manifest = True

    readme_snapshot = snapshots / "README.md"
    map_snapshot, geo_snapshot, polygon_count_snapshot = refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem="metadata",
        snapshots_dir=snapshots,
        world_land_warning=world_land_warning,
    )
    write_readme_snapshot(data_root, repo_id, readme_snapshot)

    ops = [
        add_op(processed_manifest_snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
        add_op(geo_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        add_op(polygon_count_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
        delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
    ]
    if has_aug_manifest:
        ops.extend(_augmentation_migration_ops(augmentation_manifest_snapshot))

    ops.append(add_op(readme_snapshot, path_in_repo="README.md"))
    return ops


def assemble_containment_retirement_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    parent_children: dict[str, tuple[str, ...]],
    world_land_warning: Callable[[str], None] | None = None,
) -> list[PublicationOp]:
    """Assemble one atomic parent replacement + contained-child retirement."""
    if not parent_children:
        raise PublicationValidationError("Containment retirement requires at least one parent")
    operations: list[PublicationOp] = []
    for parent in sorted(parent_children):
        for local_relative, remote in canonical_region_paths(parent).items():
            local = data_root.processed / local_relative
            if not local.is_file():
                raise FileNotFoundError(f"Canonical containment parent artifact missing: {local}")
            operations.append(add_op(local, path_in_repo=remote))
        for child in sorted(parent_children[parent]):
            for remote in canonical_region_paths(child).values():
                operations.append(delete_op(remote))

    retirement_manifest = data_root.processed / "manifests" / "containment_retirements.json"
    if not retirement_manifest.is_file():
        raise FileNotFoundError("Local containment retirement manifest is missing")

    metadata_ops = assemble_metadata_only_upload(
        data_root=data_root,
        repo_id=repo_id,
        world_land_warning=world_land_warning,
    )
    readme = metadata_ops[-1]
    if readme.path_in_repo != "README.md":
        raise PublicationValidationError("Metadata publication must end with README.md")
    operations.extend(metadata_ops[:-1])
    operations.append(add_op(retirement_manifest, path_in_repo=REMOTE_CONTAINMENT_RETIREMENT_FILE))
    operations.append(readme)
    return operations


__all__ = [
    "CorePublicationArtifacts",
    "PublicationValidationError",
    "assemble_augmentation_upload",
    "assemble_containment_retirement_upload",
    "assemble_core_upload",
    "assemble_metadata_only_upload",
    "assemble_region_upload",
    "coverage_refresh_required",
    "load_existing_core_artifacts",
    "refresh_coverage_assets",
    "snapshot_upload_manifests",
    "write_readme_snapshot",
]
