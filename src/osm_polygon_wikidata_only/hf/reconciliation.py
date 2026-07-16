from __future__ import annotations

import json
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
    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
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
        present: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        unexpected: list[str] = []
        stems_to_publish: list[str] = []
        stems_to_augment: list[str] = []
        repository_refresh: list[str] = []

        manifest_path = self.data_root.processed_manifests / "processed_pbfs.json"
        manifest_entries = load_manifest(manifest_path)

        # Load and parse augmentation_manifest.json once per plan, not once per stem
        aug_manifest_path = (
            self.data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        aug_manifest_entries = {}
        if aug_manifest_path.is_file():
            try:
                aug_manifest_entries = json.loads(aug_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ReconciliationValidationError(
                    f"Malformed augmentation manifest JSON: {exc}"
                ) from exc

        for stem in sorted(self.stems):
            # Check local core completeness
            polygons_path = self.data_root.processed_polygons / f"{stem}.parquet"
            polygon_articles_path = self.data_root.processed_links / f"{stem}.parquet"
            manifest_key = f"{stem}.osm.pbf"

            core_manifest_exists = manifest_key in manifest_entries
            core_files_exist = polygons_path.is_file() and polygon_articles_path.is_file()
            core_any_exist = (
                core_manifest_exists or polygons_path.is_file() or polygon_articles_path.is_file()
            )

            if core_any_exist and not (core_manifest_exists and core_files_exist):
                raise ReconciliationValidationError(
                    f"Inconsistent core state for {stem}: manifest_exists={core_manifest_exists}, "
                    f"polygons={polygons_path.is_file()}, links={polygon_articles_path.is_file()}"
                )

            # Check if augmented
            wikipedia_documents_path = (
                self.data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
            )

            in_aug_manifest = stem in aug_manifest_entries

            # Fail closed ONLY when manifest claims completion but required canonical file is missing
            if in_aug_manifest and not wikipedia_documents_path.is_file():
                raise ReconciliationValidationError(
                    f"Inconsistent augmentation for {stem}: manifest claims completed but missing required canonical documents file"
                )

            # Retrieve precomputed/cached augmentation status, or calculate once
            if stem in self.augmentation_current:
                is_augmented_complete = self.augmentation_current[stem]
            else:
                is_augmented_complete = augmentation_is_current(self.data_root, stem)

            if core_files_exist:
                # Use canonical_region_paths(stem) as single canonical path definition
                all_paths = canonical_region_paths(stem)
                expected_paths = {}
                for local_rel, remote_rel in all_paths.items():
                    corpus_id = "/".join(local_rel.split("/")[:-1])
                    if corpus_id in ("polygons", "polygon_articles") or is_augmented_complete:
                        expected_paths[local_rel] = remote_rel

                region_has_gap = False
                for rel_path, remote_path in expected_paths.items():
                    corpus_id = "/".join(rel_path.split("/")[:-1])
                    if self.inventory.contains(remote_path):
                        present.append((stem, corpus_id))
                    else:
                        missing.append((stem, corpus_id))
                        region_has_gap = True

                if region_has_gap:
                    if is_augmented_complete:
                        stems_to_publish.append(stem)
                    else:
                        stems_to_augment.append(stem)
                else:
                    if not is_augmented_complete:
                        stems_to_augment.append(stem)

        # Derive unexpected prefixes dynamically from canonical_region_paths
        dummy_paths = canonical_region_paths("dummy")
        remote_prefixes = sorted(
            {remote_path.removesuffix("dummy.parquet") for remote_path in dummy_paths.values()}
        )

        all_local_stems = {p.stem for p in self.data_root.processed_polygons.glob("*.parquet")}
        for remote_file in sorted(self.inventory.files):
            if any(
                remote_file.startswith(prefix) for prefix in remote_prefixes
            ) and remote_file.endswith(".parquet"):
                file_stem = Path(remote_file).stem
                if file_stem not in all_local_stems:
                    unexpected.append(remote_file)

        # Repository-level metadata/assets requiring refresh (missing only)
        repo_files = [
            (REMOTE_MANIFEST_FILE, "manifests/processed_pbfs.json"),
            (REMOTE_AUGMENTATION_MANIFEST_FILE, "manifests/augmentation_manifest.json"),
            ("README.md", "README.md"),
            (REMOTE_COVERAGE_MAP_FILE, "assets/coverage_map.png"),
            (REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE, "assets/geographic_wikipedia_text_coverage.png"),
            (REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE, "assets/geographic_polygon_count.png"),
        ]
        for remote_path, _ in repo_files:
            if not self.inventory.contains(remote_path):
                repository_refresh.append(remote_path)

        return ReconciliationPlan(
            present=tuple(sorted(present)),
            missing=tuple(sorted(missing)),
            unexpected=tuple(sorted(unexpected)),
            repository_refresh=tuple(sorted(repository_refresh)),
            stems_to_publish=frozenset(stems_to_publish),
            stems_to_augment=frozenset(stems_to_augment),
        )
