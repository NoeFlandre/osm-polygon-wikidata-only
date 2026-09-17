"""Shared CLI integration for deterministic V1 and V2 language releases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageInventory,
    LanguageInventoryError,
    build_language_inventory,
    language_split_name,
    normalize_language,
)
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps

DEFAULT_BATCH_SIZE = 65_536
V1_LANGUAGE_SPLITS_DIRNAME = "language_splits"
V2_LANGUAGE_SPLITS_DIRNAME = "language_splits"


class LanguageSplitVersion(StrEnum):
    """One published dataset contract covered by the release command."""

    V1 = "v1"
    V2 = "v2"


class LanguageSplitReleaseError(ValueError):
    """Raised when a selected language release cannot be planned safely."""


@dataclass(frozen=True, slots=True)
class LanguageSplitVersionPlan:
    """Validated source inventory and deterministic output plan for one version."""

    version: LanguageSplitVersion
    processed_root: Path
    output_root: Path
    manifest_path: Path
    inventory: LanguageInventory

    def to_dict(
        self,
        data_root: Path,
        *,
        inventory: LanguageInventory | None = None,
    ) -> dict[str, object]:
        """Serialize the plan with paths relative to the operator data root."""
        active_inventory = inventory or self.inventory
        return {
            "dataset_version": self.version.value,
            "dataset_id": active_inventory.dataset_id,
            "processed_root": _relative_path(self.processed_root, data_root),
            "output_root": _relative_path(self.output_root, data_root),
            "manifest_path": _relative_path(self.manifest_path, data_root),
            "source_manifest": _relative_path(
                self.processed_root / active_inventory.source_manifest, data_root
            ),
            "source_manifest_sha256": active_inventory.source_manifest_sha256,
            "artifact_fingerprint": active_inventory.artifact_fingerprint,
            "languages": list(active_inventory.languages),
            "tables": [table.to_dict() for table in active_inventory.tables],
            "expected_files": _expected_files(self, active_inventory),
        }


@dataclass(frozen=True, slots=True)
class LanguageSplitReleasePlan:
    """Complete deterministic plan for one selected version set."""

    data_root: Path
    dataset_version: str
    batch_size: int
    releases: tuple[LanguageSplitVersionPlan, ...]

    @property
    def versions(self) -> tuple[str, ...]:
        """Return selected versions in their stable execution order."""
        return tuple(release.version.value for release in self.releases)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible release plan."""
        return {
            "command": "language-splits",
            "dataset_version": self.dataset_version,
            "batch_size": self.batch_size,
            "releases": [release.to_dict(self.data_root) for release in self.releases],
        }


@dataclass(frozen=True, slots=True)
class LanguageSplitReleaseResult:
    """Result of a dry-run plan or a generated language release."""

    plan: LanguageSplitReleasePlan
    dry_run: bool
    generated: tuple[Any, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return deterministic JSON for scripts and publication handoffs."""
        payload = self.plan.to_payload()
        payload["dry_run"] = self.dry_run
        payload["status"] = "planned" if self.dry_run else "generated"
        if not self.dry_run:
            payload["releases"] = [
                _generated_release_payload(version_plan, generated, self.plan.data_root)
                for version_plan, generated in zip(self.plan.releases, self.generated, strict=True)
            ]
        return payload

    def to_json(self) -> str:
        """Serialize the result as one stable compact JSON document."""
        return json_dumps(self.to_payload())


def plan_language_split_release(
    data_root: DataRoot | Path,
    *,
    dataset_version: str = "both",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> LanguageSplitReleasePlan:
    """Validate selected inventories and return a no-write release plan."""
    root = _resolve_data_root(data_root)
    _validate_batch_size(batch_size)
    versions = _selected_versions(dataset_version)
    release_plans = tuple(_plan_version(root, version) for version in versions)
    return LanguageSplitReleasePlan(
        data_root=root,
        dataset_version=dataset_version,
        batch_size=batch_size,
        releases=release_plans,
    )


def run_language_split_release(
    data_root: DataRoot | Path,
    *,
    dataset_version: str = "both",
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> LanguageSplitReleaseResult:
    """Preflight all selected versions, then delegate generation if requested."""
    plan = plan_language_split_release(
        data_root,
        dataset_version=dataset_version,
        batch_size=batch_size,
    )
    if dry_run:
        return LanguageSplitReleaseResult(plan=plan, dry_run=True)

    generated = tuple(
        _generate_version(release, batch_size=plan.batch_size) for release in plan.releases
    )
    return LanguageSplitReleaseResult(plan=plan, dry_run=False, generated=generated)


def _resolve_data_root(data_root: DataRoot | Path) -> Path:
    root = data_root.path if isinstance(data_root, DataRoot) else Path(data_root)
    root = root.resolve()
    if not root.is_dir():
        raise LanguageSplitReleaseError(f"Data root is not a directory: {root}")
    return root


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise LanguageSplitReleaseError(f"batch_size must be positive, got {batch_size}")


def _selected_versions(dataset_version: str) -> tuple[LanguageSplitVersion, ...]:
    if dataset_version == "both":
        return (LanguageSplitVersion.V1, LanguageSplitVersion.V2)
    try:
        return (LanguageSplitVersion(dataset_version),)
    except ValueError as error:
        raise LanguageSplitReleaseError(
            f"dataset_version must be one of v1, v2, both; got {dataset_version!r}"
        ) from error


def _plan_version(root: Path, version: LanguageSplitVersion) -> LanguageSplitVersionPlan:
    processed_root = root / ("processed" if version is LanguageSplitVersion.V1 else "processed_v2")
    contract = DatasetContract.V1 if version is LanguageSplitVersion.V1 else DatasetContract.V2
    try:
        inventory = build_language_inventory(processed_root, contract)
    except LanguageInventoryError as error:
        raise LanguageSplitReleaseError(
            f"{version.value} language inventory validation failed: {error}"
        ) from error
    if version is LanguageSplitVersion.V1:
        output_root = processed_root / V1_LANGUAGE_SPLITS_DIRNAME
        manifest_path = output_root / "manifests/language_splits_v1.json"
    else:
        output_root = processed_root / V2_LANGUAGE_SPLITS_DIRNAME
        manifest_path = processed_root / "manifests/language_splits.json"
    return LanguageSplitVersionPlan(
        version=version,
        processed_root=processed_root,
        output_root=output_root,
        manifest_path=manifest_path,
        inventory=inventory,
    )


def _expected_files(
    plan: LanguageSplitVersionPlan,
    inventory: LanguageInventory | None = None,
) -> list[dict[str, object]]:
    active_inventory = plan.inventory if inventory is None else inventory
    if plan.version is LanguageSplitVersion.V1:
        records = _expected_v1_files(plan, active_inventory)
    else:
        records = _expected_v2_files(plan, active_inventory)
    return sorted(records, key=_expected_file_sort_key)


def _expected_v1_files(
    plan: LanguageSplitVersionPlan,
    inventory: LanguageInventory,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for table in inventory.tables:
        for bucket in table.buckets:
            if bucket.row_count == 0:
                continue
            path = (
                plan.output_root
                / "data"
                / table.configuration
                / f"{bucket.split}-00000-of-00001.parquet"
            )
            records.append(
                {
                    "table": table.table.value,
                    "configuration": table.configuration,
                    "language": bucket.language,
                    "split": bucket.split,
                    "path": _relative_path(path, plan.processed_root.parent),
                    "row_count": bucket.row_count,
                }
            )
    return records


def _expected_v2_files(
    plan: LanguageSplitVersionPlan,
    inventory: LanguageInventory,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for table in inventory.tables:
        for source_file in table.source_files:
            source_path = plan.processed_root / source_file
            counts = _source_language_counts(source_path, table.language_column)
            for language, row_count in counts.items():
                path = (
                    plan.output_root
                    / table.configuration
                    / language_split_name(language)
                    / f"{source_path.stem}.parquet"
                )
                records.append(
                    {
                        "table": table.table.value,
                        "configuration": table.configuration,
                        "language": language,
                        "split": language_split_name(language),
                        "path": _relative_path(path, plan.processed_root.parent),
                        "source_file": source_file,
                        "row_count": row_count,
                        "status": "candidate",
                    }
                )
    return records


def _source_language_counts(source_path: Path, language_column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        with pq.ParquetFile(source_path) as parquet_file:
            for batch in parquet_file.iter_batches(
                columns=[language_column], batch_size=DEFAULT_BATCH_SIZE
            ):
                for value in batch.column(0).to_pylist():
                    language = normalize_language(value).partition
                    counts[language] = counts.get(language, 0) + 1
    except (OSError, RuntimeError, ValueError) as error:
        raise LanguageSplitReleaseError(
            f"Could not inspect V2 source file for dry-run planning: {source_path}: {error}"
        ) from error
    return dict(sorted(counts.items(), key=lambda item: (item[0] == "unknown", item[0])))


def _expected_file_sort_key(record: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(record["table"]),
        str(record["language"] == "unknown"),
        str(record["language"]),
        str(record.get("path", record.get("path_template", ""))),
    )


def _generate_version(plan: LanguageSplitVersionPlan, *, batch_size: int) -> Any:
    try:
        if plan.version is LanguageSplitVersion.V1:
            from osm_polygon_wikidata_only.hf.v1_language_splits import (
                generate_v1_language_splits,
            )

            return generate_v1_language_splits(
                plan.processed_root,
                plan.output_root,
                batch_size=batch_size,
            )

        from osm_polygon_wikidata_only.v2.language_splits import build_v2_language_splits

        return build_v2_language_splits(
            plan.processed_root,
            output_root=plan.output_root,
            batch_size=batch_size,
        )
    except LanguageSplitReleaseError:
        raise
    except Exception as error:
        raise LanguageSplitReleaseError(
            f"{plan.version.value} language split generation failed: {error}"
        ) from error


def _generated_release_payload(
    plan: LanguageSplitVersionPlan,
    generated: Any,
    data_root: Path,
) -> dict[str, object]:
    actual_inventory = _recompute_inventory(plan)
    _validate_generated_inventory(plan, generated, actual_inventory)
    payload = plan.to_dict(data_root, inventory=actual_inventory)
    payload["manifest_path"] = _relative_path(generated.manifest_path, data_root)
    payload["files"] = _generated_files_payload(plan, generated, data_root)
    return payload


def _validate_generated_inventory(
    plan: LanguageSplitVersionPlan,
    generated: Any,
    actual_inventory: LanguageInventory,
) -> None:
    generated_inventory = getattr(generated, "inventory", None)
    if not isinstance(generated_inventory, LanguageInventory):
        raise LanguageSplitReleaseError(
            f"{plan.version.value} generated result is missing its source inventory"
        )
    if actual_inventory.artifact_fingerprint != plan.inventory.artifact_fingerprint:
        raise LanguageSplitReleaseError(
            f"{plan.version.value} source artifact fingerprint changed after planning: "
            f"planned={plan.inventory.artifact_fingerprint}, "
            f"actual={actual_inventory.artifact_fingerprint}"
        )
    if actual_inventory.artifact_fingerprint != generated_inventory.artifact_fingerprint:
        raise LanguageSplitReleaseError(
            f"{plan.version.value} generated result fingerprint does not match source inputs: "
            f"generated={generated_inventory.artifact_fingerprint}, "
            f"actual={actual_inventory.artifact_fingerprint}"
        )


def _generated_files_payload(
    plan: LanguageSplitVersionPlan,
    generated: Any,
    data_root: Path,
) -> list[dict[str, object]]:
    if plan.version is LanguageSplitVersion.V1:
        files = [
            {
                **file.to_dict(generated.output_root),
                "path": _relative_path(file.path, data_root),
            }
            for file in generated.files
        ]
    else:
        files = [
            {
                **file.to_dict(),
                "path": _relative_path(generated.processed_root / file.path, data_root),
            }
            for file in generated.files
        ]
    return sorted(files, key=_generated_file_sort_key)


def _recompute_inventory(plan: LanguageSplitVersionPlan) -> LanguageInventory:
    contract = DatasetContract.V1 if plan.version is LanguageSplitVersion.V1 else DatasetContract.V2
    try:
        return build_language_inventory(
            plan.processed_root,
            contract,
            manifest_path=plan.processed_root / plan.inventory.source_manifest,
        )
    except LanguageInventoryError as error:
        raise LanguageSplitReleaseError(
            f"{plan.version.value} generated metadata fingerprint check failed: {error}"
        ) from error


def _generated_file_sort_key(record: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(record["table"]),
        str(record.get("language", "") == "unknown"),
        str(record.get("language", "")),
        str(record["path"]),
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise LanguageSplitReleaseError(
            f"Language split path is outside the release data root: {path}"
        ) from error


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "LanguageSplitReleaseError",
    "LanguageSplitReleasePlan",
    "LanguageSplitReleaseResult",
    "LanguageSplitVersion",
    "plan_language_split_release",
    "run_language_split_release",
]
