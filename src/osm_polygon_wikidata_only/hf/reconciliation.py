"""Reconcile local dataset state with the canonical remote inventory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augmentation_is_current,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_AUGMENTATION_MANIFEST_FILE,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_DATASET_HERO_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_MANIFEST_FILE,
    canonical_region_paths,
)
from osm_polygon_wikidata_only.io.manifest import load_manifest


class ReconciliationValidationError(ValueError):
    """Raised when remote or local reconciliation validation fails."""


@dataclass(frozen=True, slots=True)
class ReconciliationPlan:
    present: tuple[tuple[str, str], ...]
    missing: tuple[tuple[str, str], ...]
    unexpected: tuple[str, ...]
    repository_refresh: tuple[str, ...]
    stems_to_publish: frozenset[str]
    stems_to_augment: frozenset[str]


@dataclass(frozen=True, slots=True)
class _StemReconciliation:
    present: tuple[tuple[str, str], ...] = ()
    missing: tuple[tuple[str, str], ...] = ()
    publish: bool = False
    augment: bool = False


class ReconciliationPlanner:
    def __init__(
        self,
        data_root: DataRoot,
        inventory: RemoteInventory,
        stems: set[str],
        augmentation_current: dict[str, bool] | None = None,
    ) -> None:
        self.data_root = data_root
        self.inventory = inventory
        self.stems = stems
        self.augmentation_current = augmentation_current or {}

    def plan(self) -> ReconciliationPlan:
        manifest_path = self.data_root.processed_manifests / "processed_pbfs.json"
        manifest_entries = load_manifest(manifest_path)
        aug_manifest_path = (
            self.data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        aug_manifest_entries = _load_augmentation_manifest(aug_manifest_path)
        present: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        stems_to_publish: list[str] = []
        stems_to_augment: list[str] = []
        for stem in sorted(self.stems):
            result = _plan_stem(
                self.data_root,
                self.inventory,
                stem,
                manifest_entries=manifest_entries,
                augmentation_manifest=aug_manifest_entries,
                augmentation_current=self.augmentation_current,
            )
            present.extend(result.present)
            missing.extend(result.missing)
            if result.publish:
                stems_to_publish.append(stem)
            if result.augment:
                stems_to_augment.append(stem)

        unexpected = _unexpected_remote_files(self.data_root, self.inventory)
        repository_refresh = _repository_refresh(self.inventory)

        return ReconciliationPlan(
            present=tuple(sorted(present)),
            missing=tuple(sorted(missing)),
            unexpected=tuple(sorted(unexpected)),
            repository_refresh=tuple(sorted(repository_refresh)),
            stems_to_publish=frozenset(stems_to_publish),
            stems_to_augment=frozenset(stems_to_augment),
        )


def _load_augmentation_manifest(path: Path) -> object:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReconciliationValidationError(f"Malformed augmentation manifest JSON: {exc}") from exc


def _validate_core_state(
    stem: str,
    *,
    manifest_exists: bool,
    files_exist: bool,
    any_exists: bool,
    polygons_exists: bool,
    links_exists: bool,
) -> None:
    if any_exists and not (manifest_exists and files_exist):
        raise ReconciliationValidationError(
            f"Inconsistent core state for {stem}: manifest_exists={manifest_exists}, "
            f"polygons={polygons_exists}, links={links_exists}"
        )


def _resolve_augmentation_status(
    data_root: DataRoot,
    stem: str,
    *,
    augmentation_manifest: object,
    augmentation_current: dict[str, bool],
) -> bool:
    documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    in_aug_manifest = isinstance(augmentation_manifest, Mapping) and stem in augmentation_manifest
    if in_aug_manifest and not documents_path.is_file():
        raise ReconciliationValidationError(
            f"Inconsistent augmentation for {stem}: manifest claims completed but missing required canonical documents file"
        )
    if stem in augmentation_current:
        return augmentation_current[stem]
    return augmentation_is_current(data_root, stem)


def _expected_region_paths(stem: str, *, augmented: bool) -> dict[str, str]:
    return {
        local_rel: remote_rel
        for local_rel, remote_rel in canonical_region_paths(stem).items()
        if "/".join(local_rel.split("/")[:-1]) in ("polygons", "polygon_articles") or augmented
    }


def _collect_region_presence(
    stem: str,
    inventory: RemoteInventory,
    expected_paths: dict[str, str],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    present: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for rel_path, remote_path in expected_paths.items():
        corpus_id = "/".join(rel_path.split("/")[:-1])
        target = present if inventory.contains(remote_path) else missing
        target.append((stem, corpus_id))
    return tuple(present), tuple(missing)


def _plan_stem(
    data_root: DataRoot,
    inventory: RemoteInventory,
    stem: str,
    *,
    manifest_entries: Mapping[str, object],
    augmentation_manifest: object,
    augmentation_current: dict[str, bool],
) -> _StemReconciliation:
    core_state = _core_state(data_root, stem, manifest_entries)
    _validate_core_state(stem, **core_state)
    files_exist = core_state["files_exist"]
    augmented = _resolve_augmentation_status(
        data_root,
        stem,
        augmentation_manifest=augmentation_manifest,
        augmentation_current=augmentation_current,
    )
    if not files_exist:
        return _StemReconciliation()
    present, missing = _collect_region_presence(
        stem,
        inventory,
        _expected_region_paths(stem, augmented=augmented),
    )
    has_gap = bool(missing)
    return _StemReconciliation(
        present=present,
        missing=missing,
        publish=has_gap and augmented,
        augment=not augmented,
    )


def _core_state(
    data_root: DataRoot,
    stem: str,
    manifest_entries: Mapping[str, object],
) -> dict[str, bool]:
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"
    manifest_exists = f"{stem}.osm.pbf" in manifest_entries
    polygons_exists = polygons_path.is_file()
    links_exists = links_path.is_file()
    return {
        "manifest_exists": manifest_exists,
        "files_exist": polygons_exists and links_exists,
        "any_exists": manifest_exists or polygons_exists or links_exists,
        "polygons_exists": polygons_exists,
        "links_exists": links_exists,
    }


def _unexpected_remote_files(data_root: DataRoot, inventory: RemoteInventory) -> list[str]:
    dummy_paths = canonical_region_paths("dummy")
    prefixes = sorted(
        {remote_path.removesuffix("dummy.parquet") for remote_path in dummy_paths.values()}
    )
    local_stems = {path.stem for path in data_root.processed_polygons.glob("*.parquet")}
    return [
        remote_file
        for remote_file in sorted(inventory.files)
        if _is_unexpected_region_file(remote_file, prefixes, local_stems)
    ]


def _is_unexpected_region_file(
    remote_file: str,
    prefixes: list[str],
    local_stems: set[str],
) -> bool:
    if not remote_file.endswith(".parquet"):
        return False
    if not any(remote_file.startswith(prefix) for prefix in prefixes):
        return False
    return Path(remote_file).stem not in local_stems


def _repository_refresh(inventory: RemoteInventory) -> list[str]:
    repo_files = (
        REMOTE_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
        "README.md",
        REMOTE_COVERAGE_MAP_FILE,
        REMOTE_DATASET_HERO_FILE,
        REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    )
    return [path for path in repo_files if not inventory.contains(path)]
