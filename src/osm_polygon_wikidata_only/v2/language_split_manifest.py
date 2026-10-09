"""Validate, reconcile, and describe generated V2 language split artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import cast

import pyarrow as pa

from osm_polygon_wikidata_only.hf.language_splits import (
    DATASET_V2_ID,
    LANGUAGE_SPLIT_CONTRACT_VERSION,
    UNKNOWN_LANGUAGE,
    DatasetContract,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    LanguageTableSpec,
    language_split_name,
)
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.parquet_scan import open_parquet
from osm_polygon_wikidata_only.io.staged_install import (
    CrossFilesystemInstallError,
)
from osm_polygon_wikidata_only.io.staged_install import (
    install_staged_files as install_artifact_set,
)
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.language_split_models import (
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    V2_LANGUAGE_SPLIT_CONTRACT_VERSION,
    LanguageShardWriteState,
    V2LanguageSplitError,
    V2LanguageSplitFile,
)


def validate_output_file(
    root: Path,
    spec: LanguageTableSpec,
    shard: LanguageShardWriteState,
    expected_schema: pa.Schema,
) -> V2LanguageSplitFile:
    try:
        with open_parquet(shard.staged_path) as parquet_file:
            actual_schema = parquet_file.schema_arrow
            metadata = parquet_file.metadata
            actual_rows = 0 if metadata is None else metadata.num_rows
    except Exception as error:
        raise V2LanguageSplitError(
            f"Could not validate output {shard.staged_path}: {error}"
        ) from error
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise V2LanguageSplitError(f"schema mismatch for generated output {shard.staged_path}")
    if actual_rows != shard.row_count:
        raise V2LanguageSplitError(
            f"row count mismatch for generated output {shard.staged_path}: "
            f"expected={shard.row_count}, observed={actual_rows}"
        )
    return V2LanguageSplitFile(
        table=spec.table,
        configuration=spec.configuration,
        language=shard.language,
        split=language_split_name(shard.language),
        source_files=tuple(shard.source_files),
        path=relative_path(shard.final_path, root),
        row_count=actual_rows,
        sha256=sha256_file(shard.staged_path),
    )


def shard_count(row_count: int, max_rows_per_shard: int) -> int:
    return (row_count + max_rows_per_shard - 1) // max_rows_per_shard


def validate_conservation(
    inventory: LanguageInventory,
    files: tuple[V2LanguageSplitFile, ...],
) -> None:
    observed = _observed_counts(files)
    for table_inventory in inventory.tables:
        _validate_table_conservation(table_inventory, observed)


def install_staged_files(
    root: Path,
    destination: Path,
    staged: dict[Path, Path],
) -> None:
    previous = _previous_partition_paths(root, destination)
    final_paths = set(staged)
    try:
        install_artifact_set(
            staged,
            stale=previous - final_paths,
            manifest_path=root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
        )
    except CrossFilesystemInstallError as error:
        raise V2LanguageSplitError(
            "V2 language split publication cannot cross filesystems (EXDEV): "
            f"{error.source} -> {error.destination}"
        ) from error
    _remove_empty_output_directories(destination, previous | final_paths)


def _previous_partition_paths(root: Path, destination: Path) -> set[Path]:  # noqa: ARG001 -- seam patched by tests with this signature
    payload = _read_previous_manifest(root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH)
    previous_destination = _manifest_output_root(payload, root)
    if previous_destination is None or payload is None:
        return set()
    return _manifest_partition_paths(payload, root, previous_destination)


def _read_previous_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_output_root(payload: dict[str, object] | None, root: Path) -> Path | None:
    if payload is None:
        return None
    output_root = payload.get("output_root")
    if not isinstance(output_root, str):
        return None
    candidate = (root / output_root).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _manifest_partition_paths(
    payload: dict[str, object], root: Path, destination: Path
) -> set[Path]:
    paths: set[Path] = set()
    for table in _manifest_records(payload.get("tables")):
        for bucket in _manifest_records(table.get("buckets")):
            for file in _manifest_records(bucket.get("files")):
                candidate = _manifest_partition_path(root, destination, file.get("path"))
                if candidate is not None:
                    paths.add(candidate)
    return paths


def _manifest_records(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(cast(dict[str, object], record) for record in value if isinstance(record, dict))


def _manifest_partition_path(root: Path, destination: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str):
        return None
    candidate = (root / raw_path).resolve()
    relative = _relative_to_output(candidate, destination)
    if relative is None or len(relative.parts) != 3 or relative.suffix != ".parquet":
        return None
    return candidate


def _relative_to_output(candidate: Path, destination: Path) -> Path | None:
    try:
        return candidate.relative_to(destination)
    except ValueError:
        return None


def _remove_empty_output_directories(destination: Path, owned_paths: set[Path]) -> None:
    """Prune empty directories left by removed generated shards."""
    directories = sorted(
        _output_directories_for_paths(destination, owned_paths),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def _output_directories_for_paths(destination: Path, owned_paths: set[Path]) -> set[Path]:
    directories: set[Path] = set()
    for path in owned_paths:
        try:
            path.relative_to(destination)
        except ValueError:
            continue
        for directory in path.parents:
            if directory == destination:
                break
            directories.add(directory)
    return directories


def _observed_counts(
    files: tuple[V2LanguageSplitFile, ...],
) -> dict[tuple[LanguageTable, str], int]:
    observed: dict[tuple[LanguageTable, str], int] = defaultdict(int)
    for file in files:
        observed[(file.table, file.language)] += file.row_count
    return observed


def _validate_table_conservation(
    table_inventory: LanguageTableInventory,
    observed: dict[tuple[LanguageTable, str], int],
) -> None:
    expected = {bucket.language: bucket.row_count for bucket in table_inventory.buckets}
    actual = {language: observed[(table_inventory.table, language)] for language in expected}
    if actual != expected:
        raise V2LanguageSplitError(
            f"row conservation failed for {table_inventory.table.value}: "
            f"expected={expected}, observed={actual}"
        )


def manifest_payload(
    root: Path,
    destination: Path,
    inventory: LanguageInventory,
    files: tuple[V2LanguageSplitFile, ...],
) -> dict[str, object]:
    by_table_language: dict[tuple[LanguageTable, str], list[V2LanguageSplitFile]] = defaultdict(
        list
    )
    for file in files:
        by_table_language[(file.table, file.language)].append(file)

    tables: list[dict[str, object]] = []
    for table_inventory in inventory.tables:
        buckets: list[dict[str, object]] = []
        for bucket in table_inventory.buckets:
            bucket_files = by_table_language[(table_inventory.table, bucket.language)]
            bucket_payload = bucket.to_dict()
            bucket_payload["files"] = [
                file.to_dict() for file in sorted(bucket_files, key=file_sort_key)
            ]
            buckets.append(bucket_payload)
        tables.append(
            {
                "table": table_inventory.table.value,
                "configuration": table_inventory.configuration,
                "language_column": table_inventory.language_column,
                "identity_columns": list(table_inventory.identity_columns),
                "source_files": list(table_inventory.source_files),
                "row_count": table_inventory.row_count,
                "buckets": buckets,
            }
        )

    return {
        "contract_version": V2_LANGUAGE_SPLIT_CONTRACT_VERSION,
        "dataset_contract": DatasetContract.V2.value,
        "dataset_id": DATASET_V2_ID,
        "v2_contract_version": V2_CONTRACT_VERSION,
        "language_split_contract_version": LANGUAGE_SPLIT_CONTRACT_VERSION,
        "source_manifest": inventory.source_manifest,
        "source_manifest_sha256": inventory.source_manifest_sha256,
        "artifact_fingerprint": inventory.artifact_fingerprint,
        "output_root": relative_path(destination, root),
        "tables": tables,
    }


def file_sort_key(file: V2LanguageSplitFile) -> tuple[str, bool, str, str]:
    return (
        file.table.value,
        file.language == UNKNOWN_LANGUAGE,
        file.language,
        file.path,
    )


def _language_sort_key(language: str) -> tuple[bool, str]:
    return language == UNKNOWN_LANGUAGE, language


def ensure_source_is_under_root(source_path: Path, root: Path) -> None:
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise V2LanguageSplitError(
            f"Source artifact is outside processed root: {source_path}"
        ) from error


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise V2LanguageSplitError(f"Path is outside processed root: {path}") from error
