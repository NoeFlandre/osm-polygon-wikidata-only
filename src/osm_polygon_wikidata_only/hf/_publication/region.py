"""Unified sync region publication assembly."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    validate_augmentation_artifacts as _validate_augmentation_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    validate_core_artifacts as _validate_core_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.hooks import PublicationHooks
from osm_polygon_wikidata_only.hf._publication.models import CorePublicationArtifacts
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_COVERAGE_MAP_FILE,
    LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.publication")


def assemble_region_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | CorePublicationArtifacts | None,
    world_land_warning: Callable[[str], None] | None,
    refresh_maps: bool = True,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    """Assemble one atomic region upload (sync-dir publication).

    File ordering follows the documented contract. When ``core`` is
    provided, the core operations are prepended to the augmentation
    operations. When ``core`` is ``None``, the augmentation block also
    refreshes the Wikivoyage-sensitive combined text-presence map.
    ``refresh_maps=False`` is reserved for migration/recovery transactions
    followed by one repository-level metadata publication. Those regional
    commits contain data and manifests only; maps and README are generated
    once after every regional upload has drained.

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
    hero_op = hooks.dataset_hero_op() if refresh_maps else None
    snapshots = data_root.cache / "sync_upload_snapshots" / stem
    snapshots.mkdir(parents=True, exist_ok=True)
    augmentation_manifest_snapshot = _snapshot_augmentation_manifest(snapshots, augmentation)
    ops = _core_operations(
        data_root,
        core,
        snapshots,
        refresh_maps=refresh_maps,
        world_land_warning=world_land_warning,
        hooks=hooks,
    )
    ops.extend(
        _augmentation_operations(
            data_root,
            stem,
            augmentation,
            core,
            snapshots,
            augmentation_manifest_snapshot,
            refresh_maps=refresh_maps,
            hooks=hooks,
        )
    )
    if refresh_maps:
        assert hero_op is not None
        readme_snapshot = snapshots / "README.md"
        hooks.write_readme_snapshot(data_root, repo_id, readme_snapshot)
        ops.append(hero_op)
        ops.append(add_op(readme_snapshot, path_in_repo="README.md"))
    return ops


def _snapshot_augmentation_manifest(snapshots: Path, augmentation: AugmentationResult) -> Path:
    snapshot = snapshots / "augmentation_manifest.json"
    atomic_write_text(snapshot, augmentation.manifest_path.read_text())
    return snapshot


def _core_operations(
    data_root: DataRoot,
    core: ProcessResult | CorePublicationArtifacts | None,
    snapshots: Path,
    *,
    refresh_maps: bool,
    world_land_warning: Callable[[str], None] | None,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    if core is None:
        return []
    processed_manifest_snapshot = snapshots / "processed_pbfs.json"
    atomic_write_text(
        processed_manifest_snapshot,
        (data_root.processed_manifests / "processed_pbfs.json").read_text(),
    )
    operations = [
        add_op(core.polygons_path, path_in_repo=f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
        add_op(
            core.polygon_articles_path,
            path_in_repo=f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
        ),
        add_op(processed_manifest_snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
    ]
    if refresh_maps:
        operations.extend(
            _core_map_operations(
                data_root,
                snapshots,
                world_land_warning=world_land_warning,
                hooks=hooks,
            )
        )
    return operations


def _core_map_operations(
    data_root: DataRoot,
    snapshots: Path,
    *,
    world_land_warning: Callable[[str], None] | None,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    map_snapshot = snapshots / "coverage_map.png"
    lons, lats = hooks.load_centroids_from_parquet(data_root.processed_polygons)
    land_path = _core_land_path(data_root, world_land_warning, hooks)
    hooks.generate_coverage_map(lons, lats, map_snapshot, land_geojson_path=land_path)
    text_presence_snapshot, text_density_snapshot = _text_map_snapshots(
        data_root, snapshots, land_path, hooks
    )
    return [
        add_op(text_presence_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE),
        add_op(text_density_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
        delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
    ]


def _core_land_path(
    data_root: DataRoot,
    world_land_warning: Callable[[str], None] | None,
    hooks: PublicationHooks,
) -> Path | None:
    try:
        return hooks.ensure_world_land(data_root.cache)
    # ``ensure_world_land`` performs network I/O and can raise several
    # exception types; this unified path intentionally falls back to a map
    # without continents.
    except Exception:
        if world_land_warning is not None:
            world_land_warning("Could not fetch world land data; map will omit continents")
        return None


def _text_map_snapshots(
    data_root: DataRoot,
    snapshots: Path,
    land_path: Path | None,
    hooks: PublicationHooks,
) -> tuple[Path, Path]:
    text_presence_snapshot = snapshots / "geographic_text_presence.png"
    text_snapshot = hooks.load_text_presence(data_root.processed)
    hooks.generate_geographic_text_presence(
        data_root.processed,
        text_presence_snapshot,
        land_geojson_path=land_path,
        snapshot=text_snapshot,
    )
    text_density_snapshot = snapshots / "geographic_text_density.png"
    hooks.generate_geographic_text_density_snapshot(
        data_root,
        text_density_snapshot,
        snapshot=text_snapshot,
    )
    return text_presence_snapshot, text_density_snapshot


def _augmentation_operations(
    data_root: DataRoot,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | CorePublicationArtifacts | None,
    snapshots: Path,
    augmentation_manifest_snapshot: Path,
    *,
    refresh_maps: bool,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    operations: list[PublicationOp] = []
    if core is None and augmentation.polygon_document_links_path is not None:
        operations.append(
            add_op(
                augmentation.polygon_document_links_path,
                path_in_repo=f"{REMOTE_LINKS_DIR}/{stem}.parquet",
            )
        )
    operations.extend(
        [
            *hooks.legacy_article_retirement_ops(
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
            *hooks.augmentation_migration_ops(augmentation_manifest_snapshot),
        ]
    )
    if core is None and refresh_maps:
        operations.extend(_augmentation_only_map_operations(data_root, snapshots, hooks))
    return operations


def _augmentation_only_map_operations(
    data_root: DataRoot,
    snapshots: Path,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    text_presence_snapshot = snapshots / "geographic_text_presence.png"
    try:
        land_path = hooks.ensure_world_land(data_root.cache)
    except Exception:
        LOGGER.warning("Could not fetch world land data; combined text map will omit continents")
        land_path = None
    text_presence_snapshot, text_density_snapshot = _text_map_snapshots(
        data_root, snapshots, land_path, hooks
    )
    return [
        add_op(text_presence_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE),
        add_op(text_density_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
    ]


__all__ = ["assemble_region_upload"]
