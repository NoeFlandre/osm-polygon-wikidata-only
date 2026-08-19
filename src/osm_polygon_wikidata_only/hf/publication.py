"""Public dataset publication facade.

This module owns the construction of publication-op lists for the
three documented publication contracts. Assemblers return
``list[PublicationOp]`` -- one ``add`` op per local artifact, plus
explicit ``delete`` ops for legacy paths to migrate the remote layout.

* Legacy core publication
  (called by :func:`cli.commands._enqueue_core_upload`):
    1. polygons
    2. canonical Wikipedia documents
    3. legacy articles (delete)
    4. polygon_articles
    5. processed manifest
    6. combined Wikipedia/Wikivoyage text presence
    7. combined Wikipedia/Wikivoyage H3 text density
    8. legacy Wikipedia H3 coverage (delete)
    9. legacy all-polygon H3 density (delete)
    10. static dataset hero image (add)
    11. README
    12. canonical all-polygons coverage map (add)
    13. legacy coverage map (delete)

* Unified sync with changed core
  (called by ``cli.run_sync._build_region_publication``):
    1. polygons
    2. polygon_articles
    3. processed manifest
    4. combined Wikipedia/Wikivoyage text presence
    5. combined Wikipedia/Wikivoyage H3 text density
    6. legacy Wikipedia H3 coverage (delete)
    7. legacy all-polygon H3 density (delete)
    8. canonical all-polygons coverage map (add)
    9. legacy coverage map (delete)
    10. wikipedia documents
    11. legacy articles (delete)
    12. wikipedia sections
    13. wikivoyage documents
    14. wikivoyage sections
    15. wikidata facts
    16. canonical augmentation manifest (add)
    17. legacy augmentation manifest (delete)
    18. static dataset hero image (add)
    19. README (when metadata is refreshed)

* Augmentation-only publication (legacy
  ``cli.commands._augmentation_upload_files`` behavior):
    1. wikipedia documents
    2. legacy articles (delete)
    3. wikipedia sections
    4. wikivoyage documents
    5. wikivoyage sections
    6. wikidata facts
    7. canonical augmentation manifest (add)
    8. legacy augmentation manifest (delete)
    9. combined Wikipedia/Wikivoyage point map
    10. combined Wikipedia/Wikivoyage H3 text density
    11. legacy Wikipedia H3 coverage (delete)
    12. legacy all-polygon H3 density (delete)
    13. static dataset hero image (add)
    14. README

The static hero image is stored locally at ``assets/dataset_hero.png`` and
published remotely as ``assets/dataset_hero.png`` whenever a README snapshot
is published. It is a presentation asset, separate from the generated maps.

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

The legacy geographic H3 assets are likewise retained only as explicit
delete targets. Each retirement is paired with an add for
``assets/geographic_text_density.png`` in the same atomic commit.

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

The ordered assemblers live in the focused modules under
``hf._publication``. This module keeps the stable import surface and binds
their helper dependencies at call time so existing injection and test seams
remain valid.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.integrity import INTEGRITY_CONTRACT_VERSION
from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS,
    CANONICAL_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf._dataset_stats.augmentation import (
    compute_augmentation_stats,
)
from osm_polygon_wikidata_only.hf._publication import artifacts as _publication_artifacts
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    load_existing_core_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.hooks import PublicationHooks
from osm_polygon_wikidata_only.hf._publication.models import (
    CorePublicationArtifacts,
    PublicationValidationError,
)
from osm_polygon_wikidata_only.hf._uploader.plan import (
    PublicationOp,
    add_op,
    delete_op,
)
from osm_polygon_wikidata_only.hf.continent_stats import (
    compute_continent_stats,
    render_continent_stats,
)
from osm_polygon_wikidata_only.hf.coverage_map import (
    ensure_world_countries,
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card, render_rejections_section
from osm_polygon_wikidata_only.hf.dataset_stats import (
    compute_dataset_stats,
    render_stats_section,
)
from osm_polygon_wikidata_only.hf.geographic_text_density import (
    generate_geographic_text_density as _generate_geographic_text_density,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    TextPresenceSnapshot,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    generate_geographic_text_presence as _generate_geographic_text_presence,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    load_text_presence as _load_text_presence,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_ARTICLES_DIR,
    LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
    LOCAL_DATASET_HERO_FILE,
    REMOTE_AUGMENTATION_MANIFEST_FILE,
    REMOTE_DATASET_HERO_FILE,
    REMOTE_WIKIPEDIA_DOCUMENTS_DIR,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.publication")

# Preserve the focused validation aliases used by downstream integrations and
# the existing publication ownership contract.
_validate_core_artifacts = _publication_artifacts.validate_core_artifacts
_validate_augmentation_artifacts = _publication_artifacts.validate_augmentation_artifacts


def _dataset_hero_op() -> PublicationOp:
    """Return the immutable repository hero asset publication operation."""
    if not LOCAL_DATASET_HERO_FILE.is_file():
        raise FileNotFoundError(f"Dataset hero asset is missing: {LOCAL_DATASET_HERO_FILE}")
    return add_op(LOCAL_DATASET_HERO_FILE, path_in_repo=REMOTE_DATASET_HERO_FILE)


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

    1. Computing the core :class:`DatasetStats` snapshot from the
       finalized Parquet tables via
       :func:`compute_dataset_stats`.
    2. Computing the augmentation :class:`AugmentationStats` snapshot
       via :func:`compute_augmentation_stats`. The per-file summary
       cache lives under ``data_root.cache``, so a warm refresh
       performs zero Parquet table reads.
    3. Computing public continent statistics from polygon centroids and
       the bundled Natural Earth Admin-0 reference.
    4. Rendering the public snapshot, Wikipedia and Wikivoyage corpora,
       Wikidata facts, storage accounting, and continent distribution.

    The README must be written AFTER every other snapshot so a
    partial core upload never reaches the Hub. The destination is
    written atomically via
    :func:`osm_polygon_wikidata_only.io.atomic.atomic_write_text`.
    """
    core_stats = compute_dataset_stats(data_root.processed)
    aggregate = {
        "polygon_count": core_stats.polygon_count,
        "article_count": core_stats.article_count,
        "unique_wikidata_count": core_stats.unique_wikidata_count,
    }
    augmentation_stats = compute_augmentation_stats(
        data_root.processed,
        cache_index_dir=data_root.cache,
    )
    stats_section = render_stats_section(
        core_stats,
        augmentation_stats=augmentation_stats,
    )
    if any(data_root.processed_polygons.glob("*.parquet")):
        countries_path = ensure_world_countries(data_root.cache)
        stats_section += "\n" + render_continent_stats(
            compute_continent_stats(data_root.processed, countries_path)
        )
    rejections_section: str | None = None
    audit_path = data_root.processed / "integrity" / "integrity_audit.json"
    if audit_path.is_file():
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            audit_payload = None
        if isinstance(audit_payload, dict):
            audit_contract = str(audit_payload.get("contract_version", INTEGRITY_CONTRACT_VERSION))
            rejections_section = render_rejections_section(
                {**audit_payload, "contract_version": audit_contract}
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
            link_columns=list(CANONICAL_COLUMNS),
            link_descriptions=CANONICAL_DESCRIPTIONS,
            maintainer="Noé Flandre",
            stats_section=stats_section,
            rejections_section=rejections_section,
        ),
    )


def refresh_coverage_assets(
    *,
    data_root: DataRoot,
    snapshot_stem: str,
    snapshots_dir: Path,
    world_land_warning: Callable[[str], None] | None,
) -> tuple[Path, Path, Path]:
    """Render the three public coverage PNGs into ``snapshots_dir``.

    ``world_land_warning`` controls the world-land fallback policy.
    Pass a logging-like callable (e.g. ``LOGGER.warning``) to record
    a warning when land data is unavailable, or ``None`` to swallow
    the exception silently. The publication module never invents a
    logger identity of its own.

    Returns the all-polygons, combined-point, and combined H3-density snapshots.
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
    text_presence_snapshot = snapshots_dir / f"{snapshot_stem}-geographic_text_presence.png"
    _generate_geographic_text_presence(
        data_root.processed,
        text_presence_snapshot,
        land_geojson_path=land_path,
        snapshot=(text_snapshot := _load_text_presence(data_root.processed)),
    )
    density_snapshot = snapshots_dir / f"{snapshot_stem}-geographic_text_density.png"
    _generate_geographic_text_density_snapshot(
        data_root,
        density_snapshot,
        snapshot=text_snapshot,
    )
    return map_snapshot, text_presence_snapshot, density_snapshot


def _generate_geographic_text_density_snapshot(
    data_root: DataRoot,
    destination: Path,
    *,
    snapshot: TextPresenceSnapshot | None = None,
) -> Path:
    """Build the combined text-density PNG into ``destination``."""
    result = _generate_geographic_text_density(
        data_root.processed,
        destination,
        land_cache_dir=data_root.cache,
        snapshot=snapshot,
    )
    return result.output_path


# ---------------------------------------------------------------------------
# Coverage refresh decision
# ---------------------------------------------------------------------------


def coverage_refresh_required(core: object | None) -> bool:
    """Coverage assets change only when a core polygon artifact changes."""
    return core is not None


def _publication_hooks() -> PublicationHooks:
    """Build current helper bindings for compatibility and test injection.

    The bindings are resolved for every assembly call so existing callers that
    patch the historical publication helpers retain the same behavior.
    """

    return PublicationHooks(
        dataset_hero_op=_dataset_hero_op,
        augmentation_migration_ops=_augmentation_migration_ops,
        legacy_article_retirement_ops=_legacy_article_retirement_ops,
        snapshot_upload_manifests=snapshot_upload_manifests,
        snapshot_canonical_document=_snapshot_canonical_document,
        metadata_only_upload=assemble_metadata_only_upload,
        write_readme_snapshot=write_readme_snapshot,
        refresh_coverage_assets=refresh_coverage_assets,
        ensure_world_land=ensure_world_land,
        generate_coverage_map=generate_coverage_map,
        load_centroids_from_parquet=load_centroids_from_parquet,
        generate_geographic_text_presence=_generate_geographic_text_presence,
        load_text_presence=_load_text_presence,
        generate_geographic_text_density_snapshot=_generate_geographic_text_density_snapshot,
    )


# ---------------------------------------------------------------------------
# Public compatibility facade
# ---------------------------------------------------------------------------


def assemble_core_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    core: ProcessResult,
    world_land_warning: Callable[[str], None],
) -> list[PublicationOp]:
    """Assemble the legacy core publication plan."""
    from osm_polygon_wikidata_only.hf._publication.core import (
        assemble_core_upload as _assemble_core_upload,
    )

    return _assemble_core_upload(
        data_root=data_root,
        repo_id=repo_id,
        core=core,
        world_land_warning=world_land_warning,
        hooks=_publication_hooks(),
    )


def assemble_region_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | CorePublicationArtifacts | None,
    world_land_warning: Callable[[str], None] | None,
    refresh_maps: bool = True,
) -> list[PublicationOp]:
    """Assemble one unified-sync region publication plan."""
    from osm_polygon_wikidata_only.hf._publication.region import (
        assemble_region_upload as _assemble_region_upload,
    )

    return _assemble_region_upload(
        data_root=data_root,
        repo_id=repo_id,
        stem=stem,
        augmentation=augmentation,
        core=core,
        world_land_warning=world_land_warning,
        refresh_maps=refresh_maps,
        hooks=_publication_hooks(),
    )


def assemble_augmentation_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    augmentation: AugmentationResult,
) -> list[PublicationOp]:
    """Assemble one legacy augmentation publication plan."""
    from osm_polygon_wikidata_only.hf._publication.augmentation import (
        assemble_augmentation_upload as _assemble_augmentation_upload,
    )

    return _assemble_augmentation_upload(
        data_root=data_root,
        repo_id=repo_id,
        augmentation=augmentation,
        hooks=_publication_hooks(),
    )


def assemble_metadata_only_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    world_land_warning: Callable[[str], None] | None = None,
) -> list[PublicationOp]:
    """Assemble repository metadata publication operations."""
    from osm_polygon_wikidata_only.hf._publication.metadata import (
        assemble_metadata_only_upload as _assemble_metadata_only_upload,
    )

    return _assemble_metadata_only_upload(
        data_root=data_root,
        repo_id=repo_id,
        world_land_warning=world_land_warning,
        hooks=_publication_hooks(),
    )


def assemble_containment_retirement_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    parent_children: dict[str, tuple[str, ...]],
    world_land_warning: Callable[[str], None] | None = None,
) -> list[PublicationOp]:
    """Assemble a contained-region retirement publication plan."""
    from osm_polygon_wikidata_only.hf._publication.metadata import (
        assemble_containment_retirement_upload as _assemble_containment_retirement_upload,
    )

    return _assemble_containment_retirement_upload(
        data_root=data_root,
        repo_id=repo_id,
        parent_children=parent_children,
        world_land_warning=world_land_warning,
        hooks=_publication_hooks(),
    )


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
