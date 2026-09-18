"""Bounded, deterministic row-level language partitions for V2 artifacts.

The generator consumes the validated V2 inventory from the shared language
contract, then streams complete Parquet rows through one source file at a
time.  It intentionally does not inspect or route the polygon table: a row's
own ``language`` value is the only partition key.
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION

LANGUAGE_SPLITS_DIRNAME = "language_splits"
LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH = Path("manifests/language_splits.json")
V2_LANGUAGE_SPLIT_CONTRACT_VERSION = "v2-language-splits-v2"
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_MAX_ROWS_PER_SHARD = 100_000


class V2LanguageSplitError(ValueError):
    """Raised when a V2 partition cannot conserve or validate source rows."""


@dataclass(frozen=True, slots=True)
class V2LanguageSplitFile:
    """One deterministic output shard and its source-row accounting."""

    table: LanguageTable
    configuration: str
    language: str
    split: str
    source_files: tuple[str, ...]
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
            "source_files": list(self.source_files),
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
    inventory: LanguageInventory
    files: tuple[V2LanguageSplitFile, ...]


@dataclass(slots=True)
class _ShardWriteState:
    """Mutable state for one staged language shard."""

    language: str
    shard_index: int
    writer: pq.ParquetWriter
    final_path: Path
    staged_path: Path
    row_count: int
    source_files: list[str]


@dataclass(slots=True)
class _TableWriteState:
    """Persistent per-language writers for one language-bearing table."""

    current: dict[str, _ShardWriteState]
    next_shard_index: dict[str, int]
    shards: list[_ShardWriteState]
    staged_paths: dict[Path, Path]


def build_v2_language_splits(
    processed_root: Path,
    *,
    output_root: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rows_per_shard: int = DEFAULT_MAX_ROWS_PER_SHARD,
) -> V2LanguageSplitResult:
    """Build V2 language partitions from validated local Parquet artifacts.

    Every source file is read in bounded record batches and every complete
    source row is written exactly once. Source files and their rows retain
    their deterministic order; each normalized language is written to
    deterministic bounded ``part-*`` shards. The output manifest is published
    last.
    """
    root, destination, inventory = _prepare_v2_split_request(
        processed_root, output_root, batch_size, max_rows_per_shard
    )
    manifest_path = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    inventory, ordered_files = _stage_and_install_v2_release(
        root, destination, inventory, batch_size, max_rows_per_shard, manifest_path
    )

    return V2LanguageSplitResult(
        processed_root=root,
        output_root=destination,
        manifest_path=manifest_path,
        inventory=inventory,
        files=ordered_files,
    )


def _prepare_v2_split_request(
    processed_root: Path,
    output_root: Path | None,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[Path, Path, LanguageInventory]:
    _validate_batch_size(batch_size)
    _validate_max_rows_per_shard(max_rows_per_shard)
    root = Path(processed_root).resolve()
    destination = (
        Path(output_root).resolve() if output_root is not None else root / LANGUAGE_SPLITS_DIRNAME
    )
    _ensure_output_is_under_root(destination, root)
    if destination.exists() and not destination.is_dir():
        raise V2LanguageSplitError(f"V2 language split output is not a directory: {destination}")
    inventory = build_language_inventory(root, DatasetContract.V2)
    _ensure_output_does_not_overlap_source(destination, root, inventory)
    return root, destination, inventory


def _stage_and_install_v2_release(
    root: Path,
    destination: Path,
    inventory: LanguageInventory,
    batch_size: int,
    max_rows_per_shard: int,
    manifest_path: Path,
) -> tuple[LanguageInventory, tuple[V2LanguageSplitFile, ...]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        files, staged_paths = _stage_v2_files(
            root, destination, stage_root, inventory, batch_size, max_rows_per_shard
        )
        ordered_files = tuple(sorted(files, key=_file_sort_key))
        actual_inventory = _verify_source_inventory(root, inventory)
        _validate_conservation(actual_inventory, ordered_files)
        manifest_stage = stage_root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
        atomic_write_json(
            manifest_stage, _manifest_payload(root, destination, actual_inventory, ordered_files)
        )
        staged_paths[manifest_path] = manifest_stage
        _install_staged_files(root, destination, staged_paths)
        return actual_inventory, ordered_files
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _verify_source_inventory(root: Path, expected: LanguageInventory) -> LanguageInventory:
    actual = build_language_inventory(root, DatasetContract.V2)
    if actual.artifact_fingerprint != expected.artifact_fingerprint:
        raise V2LanguageSplitError(
            "V2 source artifact fingerprint changed during generation: "
            f"expected={expected.artifact_fingerprint}, actual={actual.artifact_fingerprint}"
        )
    return actual


def _stage_v2_files(
    root: Path,
    destination: Path,
    stage_root: Path,
    inventory: LanguageInventory,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
    files: list[V2LanguageSplitFile] = []
    staged_paths: dict[Path, Path] = {}
    for spec in language_table_specs(DatasetContract.V2):
        table_inventory = cast(LanguageTableInventory, inventory.table(spec.table))
        if not isinstance(table_inventory, LanguageTableInventory):
            raise V2LanguageSplitError(
                f"V2 inventory entry is not a language table: {spec.table.value}"
            )
        generated, table_staged_paths = _write_table(
            root,
            destination,
            stage_root,
            spec,
            table_inventory,
            batch_size,
            max_rows_per_shard,
        )
        files.extend(generated)
        staged_paths.update(table_staged_paths)
    return files, staged_paths


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")


def _validate_max_rows_per_shard(max_rows_per_shard: int) -> None:
    if max_rows_per_shard < 1:
        raise ValueError(f"max_rows_per_shard must be positive, got {max_rows_per_shard}")


def _ensure_output_is_under_root(destination: Path, root: Path) -> None:
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V2 language split output must be under {root}: {destination}") from error


def _ensure_output_does_not_overlap_source(
    destination: Path,
    root: Path,
    inventory: LanguageInventory,
) -> None:
    source_paths = [root / inventory.source_manifest]
    source_paths.extend(
        root / source_file for table in inventory.tables for source_file in table.source_files
    )
    for source_path in source_paths:
        try:
            source_path.resolve().relative_to(destination)
        except ValueError:
            continue
        raise V2LanguageSplitError(
            "V2 language split output must not overlap source artifacts: "
            f"{destination} contains {source_path.resolve()}"
        )


def _write_table(
    root: Path,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    table_inventory: LanguageTableInventory,
    batch_size: int,
    max_rows_per_shard: int,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
    expected_schema = spec.schema_factory()
    shard_counts = _table_shard_counts(table_inventory, max_rows_per_shard)
    state = _TableWriteState({}, defaultdict(int), [], {})
    _stream_table_sources(
        root,
        destination,
        stage_root,
        spec,
        table_inventory,
        batch_size,
        max_rows_per_shard,
        shard_counts,
        expected_schema,
        state,
    )

    files = [_validated_output_file(root, spec, shard, expected_schema) for shard in state.shards]
    return files, state.staged_paths


def _table_shard_counts(
    table_inventory: LanguageTableInventory, max_rows_per_shard: int
) -> dict[str, int]:
    shard_counts: dict[str, int] = {}
    for bucket in table_inventory.buckets:
        if bucket.row_count > 0:
            shard_counts[bucket.language] = _shard_count(bucket.row_count, max_rows_per_shard)
    return shard_counts


def _stream_table_sources(
    root: Path,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    table_inventory: LanguageTableInventory,
    batch_size: int,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
) -> None:
    """Stream all source files for one language-bearing table."""
    with ExitStack() as stack:
        try:
            _stream_source_files(
                root,
                destination,
                stage_root,
                spec,
                table_inventory,
                batch_size,
                max_rows_per_shard,
                shard_counts,
                expected_schema,
                state,
                stack,
            )
        finally:
            _close_current_writers(state)


def _stream_source_files(
    root: Path,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    table_inventory: LanguageTableInventory,
    batch_size: int,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
    stack: ExitStack,
) -> None:
    for source_file in table_inventory.source_files:
        source_path = (root / source_file).resolve()
        _ensure_source_is_under_root(source_path, root)
        _stream_source_file(
            source_path,
            destination,
            stage_root,
            spec,
            source_file,
            batch_size,
            max_rows_per_shard,
            shard_counts,
            expected_schema,
            state,
            stack,
        )


def _close_current_writers(state: _TableWriteState) -> None:
    for shard in state.current.values():
        shard.writer.close()


def _stream_source_file(
    source_path: Path,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    source_file: str,
    batch_size: int,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
    stack: ExitStack,
) -> None:
    """Stream one validated source file into its language writers."""
    with pq.ParquetFile(source_path) as parquet_file:
        language_index = parquet_file.schema_arrow.get_field_index(spec.language_column)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            _write_batch(
                batch,
                language_index,
                destination,
                stage_root,
                spec,
                source_file,
                max_rows_per_shard,
                shard_counts,
                expected_schema,
                state,
                stack,
            )


def _write_batch(
    batch: pa.RecordBatch,
    language_index: int,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    source_file: str,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
    stack: ExitStack,
) -> None:
    for language, indices in _partition_batch(batch, language_index).items():
        _write_language_indices(
            language,
            indices,
            batch,
            destination,
            stage_root,
            spec,
            source_file,
            max_rows_per_shard,
            shard_counts,
            expected_schema,
            state,
            stack,
        )


def _write_language_indices(
    language: str,
    indices: list[int],
    batch: pa.RecordBatch,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    source_file: str,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
    stack: ExitStack,
) -> None:
    offset = 0
    while offset < len(indices):
        shard = _writer_for_language(
            language,
            destination,
            stage_root,
            spec,
            max_rows_per_shard,
            shard_counts,
            expected_schema,
            state,
            stack,
        )
        _record_source_file(shard, source_file)
        available = max_rows_per_shard - shard.row_count
        count = min(available, len(indices) - offset)
        selected_indices = indices[offset : offset + count]
        shard.writer.write_batch(batch.take(pa.array(selected_indices, type=pa.int64())))
        shard.row_count += count
        offset += count
        _close_full_shard(shard, language, max_rows_per_shard, state)


def _record_source_file(shard: _ShardWriteState, source_file: str) -> None:
    if not shard.source_files or shard.source_files[-1] != source_file:
        shard.source_files.append(source_file)


def _close_full_shard(
    shard: _ShardWriteState,
    language: str,
    max_rows_per_shard: int,
    state: _TableWriteState,
) -> None:
    if shard.row_count == max_rows_per_shard:
        shard.writer.close()
        state.current.pop(language)


def _writer_for_language(
    language: str,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: _TableWriteState,
    stack: ExitStack,
) -> _ShardWriteState:
    current = state.current.get(language)
    if current is not None:
        return current
    shard_index = state.next_shard_index[language]
    state.next_shard_index[language] += 1
    final_path = _output_path(destination, spec, language, shard_index, shard_counts[language])
    staged_path = _output_path(stage_root, spec, language, shard_index, shard_counts[language])
    temporary = stack.enter_context(atomic_replacement(staged_path))
    shard = _ShardWriteState(
        language=language,
        shard_index=shard_index,
        writer=pq.ParquetWriter(temporary, expected_schema, compression="snappy"),
        final_path=final_path,
        staged_path=staged_path,
        row_count=0,
        source_files=[],
    )
    state.current[language] = shard
    state.shards.append(shard)
    state.staged_paths[final_path] = staged_path
    return shard


def _partition_batch(batch: pa.RecordBatch, language_index: int) -> dict[str, list[int]]:
    partitions: dict[str, list[int]] = defaultdict(list)
    for row_index, value in enumerate(batch.column(language_index).to_pylist()):
        partitions[normalize_language(value).partition].append(row_index)
    return dict(partitions)


def _output_path(
    destination: Path,
    spec: LanguageTableSpec,
    language: str,
    shard_index: int,
    shard_count: int,
) -> Path:
    return (
        destination
        / spec.configuration
        / language_split_name(language)
        / f"part-{shard_index:05d}-of-{shard_count:05d}.parquet"
    )


def _validated_output_file(
    root: Path,
    spec: LanguageTableSpec,
    shard: _ShardWriteState,
    expected_schema: pa.Schema,
) -> V2LanguageSplitFile:
    try:
        with pq.ParquetFile(shard.staged_path) as parquet_file:
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
        path=_relative_path(shard.final_path, root),
        row_count=actual_rows,
        sha256=sha256_file(shard.staged_path),
    )


def _shard_count(row_count: int, max_rows_per_shard: int) -> int:
    return (row_count + max_rows_per_shard - 1) // max_rows_per_shard


def _validate_conservation(
    inventory: LanguageInventory,
    files: tuple[V2LanguageSplitFile, ...],
) -> None:
    observed = _observed_counts(files)
    for table_inventory in inventory.tables:
        _validate_table_conservation(table_inventory, observed)


def _install_staged_files(
    root: Path,
    destination: Path,
    staged: dict[Path, Path],
) -> None:
    previous = _previous_partition_paths(root, destination)
    final_paths = set(staged)
    stale = previous - final_paths
    targets = sorted(final_paths | stale, key=lambda path: path.as_posix())
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        _backup_targets(targets, backups)
        _install_files(
            staged,
            installed,
            manifest_path=root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
        )
    except BaseException:
        _restore_files(installed, backups)
        raise
    finally:
        _cleanup_transaction(staged, backups)
    _remove_empty_output_directories(destination, previous | final_paths)


def _backup_targets(targets: list[Path], backups: dict[Path, Path]) -> None:
    try:
        for target in targets:
            if target.exists():
                backups[target] = _backup_existing(target)
    except BaseException:
        _restore_files([], backups)
        raise


def _install_files(
    staged: dict[Path, Path],
    installed: list[Path],
    *,
    manifest_path: Path | None = None,
) -> None:
    ordered = sorted(
        staged.items(),
        key=lambda item: (
            manifest_path is not None and item[0] == manifest_path,
            item[0].as_posix(),
        ),
    )
    for final, temporary in ordered:
        final.parent.mkdir(parents=True, exist_ok=True)
        _replace_path(temporary, final)
        installed.append(final)


def _restore_files(installed: list[Path], backups: dict[Path, Path]) -> None:
    for final in installed:
        final.unlink(missing_ok=True)
    for final, backup in sorted(backups.items(), key=lambda item: item[0].as_posix()):
        if backup.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            _replace_path(backup, final)


def _cleanup_transaction(staged: dict[Path, Path], backups: dict[Path, Path]) -> None:
    for temporary in staged.values():
        temporary.unlink(missing_ok=True)
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def _backup_existing(path: Path) -> Path:
    descriptor, raw_backup = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".backup", dir=path.parent
    )
    os.close(descriptor)
    backup = Path(raw_backup)
    _replace_path(path, backup)
    return backup


def _replace_path(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise V2LanguageSplitError(
                "V2 language split publication cannot cross filesystems (EXDEV): "
                f"{source} -> {destination}"
            ) from error
        raise


def _previous_partition_paths(root: Path, destination: Path) -> set[Path]:
    payload = _read_previous_manifest(root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH)
    previous_destination = _manifest_output_root(payload, root)
    if previous_destination is None:
        return set()
    assert payload is not None
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
        file.path,
    )


def _language_sort_key(language: str) -> tuple[bool, str]:
    return language == UNKNOWN_LANGUAGE, language


def _ensure_source_is_under_root(source_path: Path, root: Path) -> None:
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
    "DEFAULT_MAX_ROWS_PER_SHARD",
    "LANGUAGE_SPLITS_DIRNAME",
    "LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH",
    "V2_LANGUAGE_SPLIT_CONTRACT_VERSION",
    "V2LanguageSplitError",
    "V2LanguageSplitFile",
    "V2LanguageSplitResult",
    "build_v2_language_splits",
    "main",
]
