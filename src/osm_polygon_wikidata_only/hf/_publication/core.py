"""Legacy core publication assembly."""

from __future__ import annotations

from collections.abc import Callable

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    validate_core_artifacts as _validate_core_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.hooks import PublicationHooks
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
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult


def assemble_core_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    core: ProcessResult,
    world_land_warning: Callable[[str], None],
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    """Assemble the legacy core publication op list.

    Returns the ordered list of :class:`PublicationOp` records:

    1. polygons
    2. articles
    3. polygon_articles
    4. processed manifest
    5. combined text point map
    6. combined text H3 density
    7. legacy Wikipedia H3 coverage (delete)
    8. legacy all-polygon H3 density (delete)
    9. README
    10. canonical coverage map (add)
    11. legacy coverage map (delete)

    The function is pure: no HF upload state is owned here. The
    caller submits the returned list. Required artifacts are
    validated before any snapshot is written, and any snapshot
    failure propagates without producing a partial op list. The
    legacy core publication does NOT touch the augmentation
    manifests directory at all.
    """
    _validate_core_artifacts(core)
    hero_op = hooks.dataset_hero_op()
    snapshot, card_snapshot = hooks.snapshot_upload_manifests(data_root=data_root, core=core)
    map_snapshot, text_presence_snapshot, density_snapshot = hooks.refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem=core.polygons_path.stem,
        snapshots_dir=data_root.cache / "upload_manifest_snapshots",
        world_land_warning=world_land_warning,
    )
    hooks.write_readme_snapshot(data_root, repo_id, card_snapshot)
    canonical_document = hooks.snapshot_canonical_document(
        core,
        data_root.cache
        / "upload_manifest_snapshots"
        / f"{core.articles_path.stem}-wikipedia-documents.parquet",
    )
    return [
        add_op(core.polygons_path, path_in_repo=f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
        *hooks.legacy_article_retirement_ops(
            stem=core.articles_path.stem,
            canonical_document_path=canonical_document,
        ),
        add_op(
            core.polygon_articles_path,
            path_in_repo=f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}",
        ),
        add_op(snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
        add_op(text_presence_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE),
        add_op(density_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        hero_op,
        add_op(card_snapshot, path_in_repo="README.md"),
        add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
        delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
    ]


__all__ = ["assemble_core_upload"]
