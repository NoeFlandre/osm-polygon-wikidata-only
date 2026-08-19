"""Legacy augmentation publication assembly."""

from __future__ import annotations

import logging

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.artifacts import (
    validate_augmentation_artifacts as _validate_augmentation_artifacts,
)
from osm_polygon_wikidata_only.hf._publication.hooks import PublicationHooks
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_LINKS_DIR,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.publication")


def assemble_augmentation_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    augmentation: AugmentationResult,
    hooks: PublicationHooks,
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
    8. combined Wikipedia/Wikivoyage coverage map
    9. README

    The combined text-presence map is regenerated because Wikivoyage
    documents change its numerator. The other coverage assets depend
    only on core tables and are reused. No new stem-augmentation
    manifest snapshot is created for this contract: the legacy
    augmentation command uploads the original
    ``augmentation_result.manifest_path`` directly. The README
    snapshot is rendered by this function immediately before
    returning. The function is pure: no HF upload state is owned
    here.
    """
    _validate_augmentation_artifacts(augmentation)
    hero_op = hooks.dataset_hero_op()
    snapshots = data_root.cache / "augmentation_upload_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    readme_snapshot = snapshots / f"{augmentation.wikipedia_documents_path.stem}-README.md"
    text_presence_snapshot = (
        snapshots / f"{augmentation.wikipedia_documents_path.stem}-geographic_text_presence.png"
    )
    try:
        land_path = hooks.ensure_world_land(data_root.cache)
    except Exception:
        LOGGER.warning("Could not fetch world land data; combined text map will omit continents")
        land_path = None
    text_snapshot = hooks.load_text_presence(data_root.processed)
    hooks.generate_geographic_text_presence(
        data_root.processed,
        text_presence_snapshot,
        land_geojson_path=land_path,
        snapshot=text_snapshot,
    )
    density_snapshot = (
        snapshots / f"{augmentation.wikipedia_documents_path.stem}-geographic_text_density.png"
    )
    hooks.generate_geographic_text_density_snapshot(
        data_root,
        density_snapshot,
        snapshot=text_snapshot,
    )
    hooks.write_readme_snapshot(data_root, repo_id, readme_snapshot)
    return [
        *(
            [
                add_op(
                    augmentation.polygon_document_links_path,
                    path_in_repo=(
                        f"{REMOTE_LINKS_DIR}/{augmentation.polygon_document_links_path.name}"
                    ),
                )
            ]
            if augmentation.polygon_document_links_path is not None
            else []
        ),
        *hooks.legacy_article_retirement_ops(
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
        *hooks.augmentation_migration_ops(augmentation.manifest_path),
        add_op(text_presence_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE),
        add_op(density_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        hero_op,
        add_op(readme_snapshot, path_in_repo="README.md"),
    ]


__all__ = ["assemble_augmentation_upload"]
