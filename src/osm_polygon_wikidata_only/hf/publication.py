"""Pure dataset publication assembly.

This module owns the construction of ``(local_path, remote_path)``
upload file lists for the three documented publication contracts:

* Legacy core publication
  (called by :func:`cli.commands._enqueue_core_upload`):
    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. README
    8. legacy coverage map
* Unified sync with changed core
  (called by ``cli.run_sync._build_region_publication``):
    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. legacy coverage map
    8. wikipedia documents
    9. wikipedia sections
    10. wikivoyage documents
    11. wikivoyage sections
    12. wikidata facts
    13. augmentation manifest snapshot
    14. README
* Augmentation-only publication (legacy
  ``cli.commands._augmentation_upload_files`` behavior):
    1. wikipedia documents
    2. wikipedia sections
    3. wikivoyage documents
    4. wikivoyage sections
    5. wikidata facts
    6. augmentation manifest (the original
       ``augmentation_result.manifest_path``, NOT a stem snapshot)
    7. README

The assembly functions are PURE: each returns the ordered file
list but performs no upload and accepts no ``submit`` callable.
CLI code performs exactly one queue/direct submission after
successful assembly. Failures inside an assembler raise BEFORE
any file is published: required local artifacts are validated at
the top of each entry point, and snapshot generation failures
propagate without being swallowed.

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

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
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
    REMOTE_ARTICLES_DIR,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.publication")


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
    stats_section = render_stats_section(compute_dataset_stats(data_root.processed))
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


def _validate_core_artifacts(core: ProcessResult) -> None:
    paths: Sequence[Path] = (
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
# Assembly contracts (pure: each returns the ordered file list)
# ---------------------------------------------------------------------------


def assemble_core_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    core: ProcessResult,
    world_land_warning: Callable[[str], None],
) -> list[tuple[Path, str]]:
    """Assemble the legacy core publication file list.

    Returns the ordered list of ``(local_path, remote_path)`` tuples:

    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. geographic text coverage
    6. geographic polygon count
    7. README
    8. legacy coverage map

    The function is pure: no HF upload state is owned here. The
    caller submits the returned list. Required artifacts are
    validated before any snapshot is written, and any snapshot
    failure propagates without producing a partial file list.
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
    return [
        (core.polygons_path, f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
        (core.articles_path, f"{REMOTE_ARTICLES_DIR}/{core.articles_path.name}"),
        (
            core.polygon_articles_path,
            f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
        ),
        (snapshot, REMOTE_MANIFEST_FILE),
        (geo_snapshot, REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        (polygon_count_snapshot, REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        (card_snapshot, "README.md"),
        (map_snapshot, REMOTE_COVERAGE_MAP_FILE),
    ]


def assemble_region_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | None,
    world_land_warning: Callable[[str], None] | None,
) -> list[tuple[Path, str]]:
    """Assemble one atomic region upload (sync-dir publication).

    File ordering follows the documented contract. When ``core`` is
    provided, the seven core artifacts are prepended to the seven
    augmentation artifacts. When ``core`` is ``None``, only the
    augmentation block is produced. Coverage assets are refreshed
    only when ``core`` is not ``None``.

    The function is pure: no HF upload state is owned here. The
    caller submits the returned list. Required artifacts are
    validated before any snapshot is written, and any snapshot
    failure propagates without producing a partial file list.

    ``world_land_warning`` controls the world-land fallback policy.
    Pass a logging-like callable to record a warning when land
    data is unavailable, or ``None`` to swallow the exception
    silently (the unified-sync policy).
    """
    if core is not None:
        _validate_core_artifacts(core)
    _validate_augmentation_artifacts(augmentation)
    snapshots = data_root.cache / "sync_upload_snapshots" / stem
    snapshots.mkdir(parents=True, exist_ok=True)
    augmentation_manifest_snapshot = snapshots / "augmentation_manifest.json"
    atomic_write_text(augmentation_manifest_snapshot, augmentation.manifest_path.read_text())
    readme_snapshot = snapshots / "README.md"

    files: list[tuple[Path, str]] = []

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
        except Exception:
            if world_land_warning is not None:
                world_land_warning("Could not fetch world land data; map will omit continents")
            land_path = None
        generate_coverage_map(lons, lats, map_snapshot, land_geojson_path=land_path)
        geographic_text_snapshot = snapshots / "geographic_text_coverage.png"
        _generate_geographic_text_coverage_snapshot(data_root, geographic_text_snapshot)
        geographic_polygon_count_snapshot = snapshots / "geographic_polygon_count.png"
        _generate_geographic_polygon_count_snapshot(data_root, geographic_polygon_count_snapshot)
        files.extend(
            [
                (core.polygons_path, f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
                (core.articles_path, f"{REMOTE_ARTICLES_DIR}/{core.articles_path.name}"),
                (
                    core.polygon_articles_path,
                    f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
                ),
                (processed_manifest_snapshot, REMOTE_MANIFEST_FILE),
                (geographic_text_snapshot, REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
                (
                    geographic_polygon_count_snapshot,
                    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
                ),
                (map_snapshot, REMOTE_COVERAGE_MAP_FILE),
            ]
        )

    files.extend(
        [
            (
                augmentation.wikipedia_documents_path,
                f"wikipedia/documents/{stem}.parquet",
            ),
            (
                augmentation.wikipedia_sections_path,
                f"wikipedia/sections/{stem}.parquet",
            ),
            (
                augmentation.wikivoyage_documents_path,
                f"wikivoyage/documents/{stem}.parquet",
            ),
            (
                augmentation.wikivoyage_sections_path,
                f"wikivoyage/sections/{stem}.parquet",
            ),
            (augmentation.wikidata_facts_path, f"wikidata/facts/{stem}.parquet"),
            (
                augmentation_manifest_snapshot,
                "augmentation/manifests/augmentation_manifest.json",
            ),
            (readme_snapshot, "README.md"),
        ]
    )
    write_readme_snapshot(data_root, repo_id, readme_snapshot)
    return files


def assemble_augmentation_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    augmentation: AugmentationResult,
) -> list[tuple[Path, str]]:
    """Assemble one augmentation-only publication file list.

    File ordering follows the documented contract:

    1. wikipedia documents
    2. wikipedia sections
    3. wikivoyage documents
    4. wikivoyage sections
    5. wikidata facts
    6. augmentation manifest (the original
       ``augmentation_result.manifest_path``, NOT a stem snapshot)
    7. README

    No coverage assets are regenerated. No new stem-augmentation
    manifest snapshot is created for this contract: the legacy
    augmentation command uploads ``augmentation_result.manifest_path``
    directly. The README snapshot is rendered by this function
    immediately before returning, so the caller can submit the
    resulting list directly. The function is pure: no HF upload
    state is owned here.
    """
    _validate_augmentation_artifacts(augmentation)
    snapshots = data_root.cache / "augmentation_upload_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    readme_snapshot = snapshots / f"{augmentation.wikipedia_documents_path.stem}-README.md"
    write_readme_snapshot(data_root, repo_id, readme_snapshot)
    return [
        (
            augmentation.wikipedia_documents_path,
            str(augmentation.wikipedia_documents_path.relative_to(data_root.processed)),
        ),
        (
            augmentation.wikipedia_sections_path,
            str(augmentation.wikipedia_sections_path.relative_to(data_root.processed)),
        ),
        (
            augmentation.wikivoyage_documents_path,
            str(augmentation.wikivoyage_documents_path.relative_to(data_root.processed)),
        ),
        (
            augmentation.wikivoyage_sections_path,
            str(augmentation.wikivoyage_sections_path.relative_to(data_root.processed)),
        ),
        (
            augmentation.wikidata_facts_path,
            str(augmentation.wikidata_facts_path.relative_to(data_root.processed)),
        ),
        (augmentation.manifest_path, "augmentation/manifests/augmentation_manifest.json"),
        (readme_snapshot, "README.md"),
    ]


__all__ = [
    "assemble_augmentation_upload",
    "assemble_core_upload",
    "assemble_region_upload",
    "coverage_refresh_required",
    "refresh_coverage_assets",
    "snapshot_upload_manifests",
    "write_readme_snapshot",
]
