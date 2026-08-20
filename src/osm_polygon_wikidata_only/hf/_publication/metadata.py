"""Repository metadata and containment retirement publication assembly."""

from __future__ import annotations

from collections.abc import Callable

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._publication.hooks import PublicationHooks
from osm_polygon_wikidata_only.hf._publication.models import PublicationValidationError
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op, delete_op
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_COVERAGE_MAP_FILE,
    LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_CONTAINMENT_RETIREMENT_FILE,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_MANIFEST_FILE,
    canonical_region_paths,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text


def assemble_metadata_only_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    world_land_warning: Callable[[str], None] | None = None,
    hooks: PublicationHooks,
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
    map_snapshot, text_presence_snapshot, density_snapshot = hooks.refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem="metadata",
        snapshots_dir=snapshots,
        world_land_warning=world_land_warning,
    )
    hooks.write_readme_snapshot(data_root, repo_id, readme_snapshot)
    hero_op = hooks.dataset_hero_op()

    ops = [
        add_op(processed_manifest_snapshot, path_in_repo=REMOTE_MANIFEST_FILE),
        add_op(text_presence_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE),
        add_op(density_snapshot, path_in_repo=REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE),
        delete_op(LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE),
        add_op(map_snapshot, path_in_repo=REMOTE_COVERAGE_MAP_FILE),
        delete_op(LEGACY_REMOTE_COVERAGE_MAP_FILE),
        hero_op,
    ]
    if has_aug_manifest:
        ops.extend(hooks.augmentation_migration_ops(augmentation_manifest_snapshot))

    ops.append(add_op(readme_snapshot, path_in_repo="README.md"))
    return ops


def assemble_containment_retirement_upload(
    *,
    data_root: DataRoot,
    repo_id: str,
    parent_children: dict[str, tuple[str, ...]],
    world_land_warning: Callable[[str], None] | None = None,
    hooks: PublicationHooks,
) -> list[PublicationOp]:
    """Assemble one atomic parent replacement + contained-child retirement."""
    if not parent_children:
        raise PublicationValidationError("Containment retirement requires at least one parent")
    operations = _containment_operations(data_root, parent_children)
    retirement_manifest = _require_retirement_manifest(data_root)
    metadata_ops = hooks.metadata_only_upload(
        data_root=data_root, repo_id=repo_id, world_land_warning=world_land_warning
    )
    readme = _require_readme_operation(metadata_ops)
    operations.extend(metadata_ops[:-1])
    operations.append(add_op(retirement_manifest, path_in_repo=REMOTE_CONTAINMENT_RETIREMENT_FILE))
    operations.append(readme)
    return operations


def _containment_operations(
    data_root: DataRoot,
    parent_children: dict[str, tuple[str, ...]],
) -> list[PublicationOp]:
    operations: list[PublicationOp] = []
    for parent in sorted(parent_children):
        operations.extend(_parent_operations(data_root, parent))
        operations.extend(_child_delete_operations(parent_children[parent]))
    return operations


def _parent_operations(data_root: DataRoot, parent: str) -> list[PublicationOp]:
    operations: list[PublicationOp] = []
    for local_relative, remote in canonical_region_paths(parent).items():
        local = data_root.processed / local_relative
        if not local.is_file():
            raise FileNotFoundError(f"Canonical containment parent artifact missing: {local}")
        operations.append(add_op(local, path_in_repo=remote))
    return operations


def _child_delete_operations(children: tuple[str, ...]) -> list[PublicationOp]:
    return [
        delete_op(remote)
        for child in sorted(children)
        for remote in canonical_region_paths(child).values()
    ]


def _require_retirement_manifest(data_root: DataRoot):
    retirement_manifest = data_root.processed / "manifests" / "containment_retirements.json"
    if not retirement_manifest.is_file():
        raise FileNotFoundError("Local containment retirement manifest is missing")
    return retirement_manifest


def _require_readme_operation(metadata_ops: list[PublicationOp]) -> PublicationOp:
    readme = metadata_ops[-1]
    if readme.path_in_repo != "README.md":
        raise PublicationValidationError("Metadata publication must end with README.md")
    return readme


__all__ = ["assemble_containment_retirement_upload", "assemble_metadata_only_upload"]
