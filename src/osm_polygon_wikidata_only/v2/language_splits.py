"""Bounded, deterministic row-level language partitions for V2 artifacts.

The generator consumes the validated V2 inventory from the shared language
contract, then streams complete Parquet rows through one source file at a
time.  It intentionally does not inspect or route the polygon table: a row's
own ``language`` value is the only partition key.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.language_splits import (
    DATASET_V2_ID,
    LANGUAGE_SPLIT_CONTRACT_VERSION,
    UNKNOWN_LANGUAGE,
    DatasetContract,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    LanguageTableSpec,
    build_language_inventory,
    language_split_name,
    language_table_specs,
    normalize_language,
)
from osm_polygon_wikidata_only.io.atomic import atomic_replacement, atomic_write_json
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION

LANGUAGE_SPLITS_DIRNAME = "language_splits"
LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH = Path("manifests/language_splits.json")
V2_LANGUAGE_SPLIT_CONTRACT_VERSION = "v2-language-splits-v1"
DEFAULT_BATCH_SIZE = 65_536


class V2LanguageSplitError(ValueError):
    """Raised when a V2 partition cannot conserve or validate source rows."""


@dataclass(frozen=True, slots=True)
class V2LanguageSplitFile:
    """One deterministic output shard and its source-row accounting."""

    table: LanguageTable
    configuration: str
    language: str
    split: str
    source_file: str
    path: str
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable manifest representation of this shard."""
        return {
            "table": self.table.value,
            "configuration": self.configuration,
            "language": self.language,
            "split": self.split,
            "source_file": self.source_file,
            "path": self.path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class V2LanguageSplitResult:
    """Published V2 language partitions and their deterministic manifest."""

    processed_root: Path
    output_root: Path
    manifest_path: Path
    files: tuple[V2LanguageSplitFile, ...]


def build_v2_language_splits(
    processed_root: Path,
    *,
    output_root: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> V2LanguageSplitResult:
    """Build V2 language partitions from validated local Parquet artifacts.

    Every source file is read in bounded record batches and every complete
    source row is written exactly once.  Source files and their rows retain
    their deterministic order; one output shard is produced per source file
    and normalized language.  The output manifest is published last.
    """
    _validate_batch_size(batch_size)
    root = Path(processed_root).resolve()
    destination = (
        Path(output_root).resolve() if output_root is not None else root / LANGUAGE_SPLITS_DIRNAME
    )
    _ensure_output_is_under_root(destination, root)

    inventory = build_language_inventory(root, DatasetContract.V2)
    specs = language_table_specs(DatasetContract.V2)
    files: list[V2LanguageSplitFile] = []
    for spec in specs:
        table_inventory = inventory.table(spec.table)
        if not isinstance(table_inventory, LanguageTableInventory):
            raise V2LanguageSplitError(f"Expected language inventory for {spec.table.value}")
        for source_file in table_inventory.source_files:
            files.extend(
                _write_source_file(
                    root,
                    destination,
                    spec,
                    source_file,
                    batch_size,
                )
            )

    ordered_files = tuple(sorted(files, key=_file_sort_key))
    _validate_conservation(inventory, ordered_files)
    manifest_path = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    manifest = _manifest_payload(root, destination, inventory, ordered_files)
    atomic_write_json(manifest_path, manifest)
    return V2LanguageSplitResult(
        processed_root=root,
        output_root=destination,
        manifest_path=manifest_path,
        files=ordered_files,
    )


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _ensure_output_is_under_root(destination: Path, root: Path) -> None:
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V2 language split output must be under {root}: {destination}") from error


def _write_source_file(
    root: Path,
    destination: Path,
    spec: LanguageTableSpec,
    source_file: str,
    batch_size: int,
) -> list[V2LanguageSplitFile]:
    source_path = (root / source_file).resolve()
    _ensure_source_is_under_root(source_path, root)
    expected_schema = spec.schema_factory()
    writers: dict[str, pq.ParquetWriter] = {}
    final_paths: dict[str, Path] = {}
    row_counts: dict[str, int] = defaultdict(int)
    source_row_count = 0
    with ExitStack() as stack:
        try:
            with pq.ParquetFile(source_path) as parquet_file:
                language_index = parquet_file.schema_arrow.get_field_index(spec.language_column)
                if language_index < 0:
                    raise V2LanguageSplitError(
                        f"Language column {spec.language_column!r} is missing from {source_path}"
                    )
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    source_row_count += batch.num_rows
                    partitions = _partition_batch(batch, language_index)
                    for language, indices in partitions.items():
                        writer = writers.get(language)
                        if writer is None:
                            final_path = _output_path(
                                destination,
                                spec,
                                language,
                                source_path.stem,
                            )
                            temporary = stack.enter_context(atomic_replacement(final_path))
                            writer = pq.ParquetWriter(
                                temporary,
                                expected_schema,
                                compression="snappy",
                            )
                            writers[language] = writer
                            final_paths[language] = final_path
                        selected = batch.take(pa.array(indices, type=pa.int64()))
                        writer.write_batch(selected)
                        row_counts[language] += len(indices)
        finally:
            for writer in writers.values():
                writer.close()

    expected_rows = sum(row_counts.values())
    if source_row_count != expected_rows:
        raise V2LanguageSplitError(
            f"Rows were not assigned exactly once for {source_file}: "
            f"source={source_row_count}, partitions={expected_rows}"
        )
    return [
        _validated_output_file(
            root,
            spec,
            language,
            source_file,
            final_paths[language],
            row_counts[language],
            expected_schema,
        )
        for language in sorted(row_counts, key=_language_sort_key)
    ]


def _partition_batch(batch: pa.RecordBatch, language_index: int) -> dict[str, list[int]]:
    partitions: dict[str, list[int]] = defaultdict(list)
    for row_index, value in enumerate(batch.column(language_index).to_pylist()):
        partitions[normalize_language(value).partition].append(row_index)
    return dict(partitions)


def _output_path(
    destination: Path,
    spec: LanguageTableSpec,
    language: str,
    source_stem: str,
) -> Path:
    return (
        destination / spec.configuration / language_split_name(language) / f"{source_stem}.parquet"
    )


def _validated_output_file(
    root: Path,
    spec: LanguageTableSpec,
    language: str,
    source_file: str,
    path: Path,
    expected_rows: int,
    expected_schema: pa.Schema,
) -> V2LanguageSplitFile:
    try:
        with pq.ParquetFile(path) as parquet_file:
            actual_schema = parquet_file.schema_arrow
            metadata = parquet_file.metadata
            actual_rows = 0 if metadata is None else metadata.num_rows
    except Exception as error:
        raise V2LanguageSplitError(f"Could not validate output {path}: {error}") from error
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise V2LanguageSplitError(f"schema mismatch for generated output {path}")
    if actual_rows != expected_rows:
        raise V2LanguageSplitError(
            f"row count mismatch for generated output {path}: "
            f"expected={expected_rows}, observed={actual_rows}"
        )
    return V2LanguageSplitFile(
        table=spec.table,
        configuration=spec.configuration,
        language=language,
        split=language_split_name(language),
        source_file=source_file,
        path=_relative_path(path, root),
        row_count=actual_rows,
        sha256=sha256_file(path),
    )


def _validate_conservation(
    inventory: LanguageInventory,
    files: tuple[V2LanguageSplitFile, ...],
) -> None:
    observed: dict[tuple[LanguageTable, str], int] = defaultdict(int)
    for file in files:
        observed[(file.table, file.language)] += file.row_count
    for table_inventory in inventory.tables:
        expected = {bucket.language: bucket.row_count for bucket in table_inventory.buckets}
        actual = {language: observed[(table_inventory.table, language)] for language in expected}
        if actual != expected:
            raise V2LanguageSplitError(
                f"row conservation failed for {table_inventory.table.value}: "
                f"expected={expected}, observed={actual}"
            )


def _manifest_payload(
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
            if bucket.row_count == 0:
                continue
            bucket_payload = bucket.to_dict()
            bucket_payload["files"] = [
                file.to_dict() for file in sorted(bucket_files, key=_file_sort_key)
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
        "output_root": _relative_path(destination, root),
        "tables": tables,
    }


def _file_sort_key(file: V2LanguageSplitFile) -> tuple[str, bool, str, str]:
    return (
        file.table.value,
        file.language == UNKNOWN_LANGUAGE,
        file.language,
        file.source_file,
    )


def _language_sort_key(language: str) -> tuple[bool, str]:
    return language == UNKNOWN_LANGUAGE, language


def _ensure_source_is_under_root(source_path: Path, root: Path) -> None:
    if not source_path.is_file():
        raise V2LanguageSplitError(f"Source artifact is missing: {source_path}")
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise V2LanguageSplitError(
            f"Source artifact is outside processed root: {source_path}"
        ) from error


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise V2LanguageSplitError(f"Path is outside processed root: {path}") from error


def main(argv: list[str] | None = None) -> int:
    """Run the local V2 language split generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    result = build_v2_language_splits(args.processed_root, batch_size=args.batch_size)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module command
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "LANGUAGE_SPLITS_DIRNAME",
    "LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH",
    "V2_LANGUAGE_SPLIT_CONTRACT_VERSION",
    "V2LanguageSplitError",
    "V2LanguageSplitFile",
    "V2LanguageSplitResult",
    "build_v2_language_splits",
    "main",
]
