"""Generate deterministic, row-level language partitions for the V1 dataset.

The generator consumes the validated V1 inventory from
:mod:`osm_polygon_wikidata_only.hf.language_splits`. It writes only the five
language-bearing V1 tables; the default polygons, facts, and undifferentiated
text artifacts remain unchanged. Every source row is routed using its own
``language`` value, never a polygon-level preference.

The generated tree is suitable for a Hugging Face dataset repository::

    data/<configuration>/lang-<language>-00000-of-00001.parquet
    manifests/language_splits_v1.json

The work is staged beside the requested output and installed as one file
transaction. Source files are read in the inventory's sorted order and only
bounded Arrow batches are retained while partition files are written.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.language_splits import (
    DATASET_V1_ID,
    LANGUAGE_SPLIT_CONTRACT_VERSION,
    DatasetContract,
    LanguageInventory,
    LanguageTableInventory,
    LanguageTableSpec,
    build_language_inventory,
    language_table_specs,
    normalize_language,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads

V1_LANGUAGE_SPLIT_MANIFEST = "manifests/language_splits_v1.json"
V1_LANGUAGE_DATA_DIR = "data"
V1_LANGUAGE_FILE_SUFFIX = "-00000-of-00001.parquet"
DEFAULT_BATCH_SIZE = 65_536
_PARQUET_COMPRESSION = "snappy"


class V1LanguageSplitError(ValueError):
    """Raised when a V1 partition release cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class V1PartitionFile:
    """One generated language-partition Parquet file."""

    table: str
    configuration: str
    split: str
    path: Path
    row_count: int
    sha256: str
    columns: tuple[str, ...]

    def to_dict(self, output_root: Path) -> dict[str, object]:
        """Serialize the file record with a path relative to the release."""
        return {
            "table": self.table,
            "configuration": self.configuration,
            "split": self.split,
            "path": self.path.resolve().relative_to(output_root.resolve()).as_posix(),
            "row_count": self.row_count,
            "sha256": self.sha256,
            "columns": list(self.columns),
        }


@dataclass(frozen=True, slots=True)
class V1LanguageSplitRelease:
    """Result of one complete V1 language-partition generation run."""

    output_root: Path
    manifest_path: Path
    inventory: LanguageInventory
    files: tuple[V1PartitionFile, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic release metadata payload."""
        return _manifest_payload(self.output_root, self.inventory, self.files)


@dataclass(frozen=True, slots=True)
class _StagedPartition:
    """A generated file before it is installed at its final path."""

    artifact: V1PartitionFile
    staged_path: Path


def generate_v1_language_splits(
    processed_root: Path,
    output_root: Path,
    *,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> V1LanguageSplitRelease:
    """Generate all discovered V1 language partitions from validated input.

    ``processed_root`` is the complete validated V1 ``processed/`` directory.
    The shared contract validates the manifest, required neutral artifacts,
    schemas, and row-level language inventory before any output is staged.
    ``batch_size`` bounds each in-memory Arrow batch and is exposed to make
    small deterministic fixtures exercise the same streaming path.
    """
    source_root = Path(processed_root).resolve()
    release_root = Path(output_root).resolve()
    _validate_generation_request(source_root, release_root, batch_size)

    inventory = build_language_inventory(
        source_root,
        DatasetContract.V1,
        manifest_path=manifest_path,
    )
    release_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{release_root.name}-", dir=release_root.parent))
    try:
        files = _stage_and_install(
            source_root,
            release_root,
            stage_root,
            inventory,
            batch_size=batch_size,
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    final_files = tuple(
        V1PartitionFile(
            table=item.table,
            configuration=item.configuration,
            split=item.split,
            path=item.path,
            row_count=item.row_count,
            sha256=item.sha256,
            columns=item.columns,
        )
        for item in files
    )
    return V1LanguageSplitRelease(
        output_root=release_root,
        manifest_path=release_root / V1_LANGUAGE_SPLIT_MANIFEST,
        inventory=inventory,
        files=final_files,
    )


def _validate_generation_request(
    source_root: Path,
    release_root: Path,
    batch_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if source_root == release_root:
        raise V1LanguageSplitError(
            "V1 language output must use a release root separate from processed input"
        )
    if release_root.exists() and not release_root.is_dir():
        raise V1LanguageSplitError(f"V1 language output is not a directory: {release_root}")


def _stage_and_install(
    source_root: Path,
    release_root: Path,
    stage_root: Path,
    inventory: LanguageInventory,
    *,
    batch_size: int,
) -> tuple[V1PartitionFile, ...]:
    staged = _stage_partitions(
        source_root,
        release_root,
        stage_root,
        inventory,
        batch_size=batch_size,
    )
    files = tuple(item.artifact for item in staged)
    manifest_stage = stage_root / V1_LANGUAGE_SPLIT_MANIFEST
    atomic_write_text(
        manifest_stage,
        json_dumps(_manifest_payload(release_root, inventory, files)) + "\n",
    )
    staged_paths = {item.artifact.path: item.staged_path for item in staged}
    staged_paths[release_root / V1_LANGUAGE_SPLIT_MANIFEST] = manifest_stage
    _install_staged_files(release_root, staged_paths)
    return files


def _stage_partitions(
    source_root: Path,
    release_root: Path,
    stage_root: Path,
    inventory: LanguageInventory,
    *,
    batch_size: int,
) -> tuple[_StagedPartition, ...]:
    specs = {spec.table: spec for spec in language_table_specs(DatasetContract.V1)}
    staged: list[_StagedPartition] = []
    for table_inventory in inventory.tables:
        spec = specs[table_inventory.table]
        staged.extend(
            _stage_table_partitions(
                source_root,
                release_root,
                stage_root,
                spec,
                table_inventory,
                batch_size=batch_size,
            )
        )
    return tuple(staged)


def _stage_table_partitions(
    source_root: Path,
    release_root: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    table_inventory: LanguageTableInventory,
    *,
    batch_size: int,
) -> tuple[_StagedPartition, ...]:
    schema = spec.schema_factory()
    source_paths = tuple(source_root / relative for relative in table_inventory.source_files)
    observed_rows, staged_paths, written_counts = _partition_table_sources(
        source_paths,
        spec,
        schema,
        stage_root,
        batch_size,
    )
    _validate_partition_counts(table_inventory, observed_rows, written_counts)
    return _build_staged_partitions(
        release_root,
        stage_root,
        spec,
        schema,
        staged_paths,
        written_counts,
    )


def _partition_table_sources(
    source_paths: tuple[Path, ...],
    spec: LanguageTableSpec,
    schema: pa.Schema,
    stage_root: Path,
    batch_size: int,
) -> tuple[int, dict[str, Path], dict[str, int]]:
    staged_paths: dict[str, Path] = {}
    writers: dict[str, Any] = {}
    written_counts: dict[str, int] = {}
    observed_rows = 0
    try:
        for source_path in source_paths:
            observed_rows += _stream_source_file(
                source_path,
                spec,
                schema,
                stage_root,
                batch_size,
                writers,
                staged_paths,
                written_counts,
            )
    finally:
        _close_writers(writers)
    return observed_rows, staged_paths, written_counts


def _validate_partition_counts(
    table_inventory: LanguageTableInventory,
    observed_rows: int,
    written_counts: dict[str, int],
) -> None:
    expected_counts = {bucket.language: bucket.row_count for bucket in table_inventory.buckets}
    _validate_observed_row_count(table_inventory, observed_rows)
    _validate_written_counts(table_inventory, expected_counts, written_counts)


def _validate_observed_row_count(
    table_inventory: LanguageTableInventory,
    observed_rows: int,
) -> None:
    if observed_rows != table_inventory.row_count:
        raise V1LanguageSplitError(
            f"row count changed while partitioning {table_inventory.table.value}: "
            f"inventory={table_inventory.row_count}, observed={observed_rows}"
        )


def _validate_written_counts(
    table_inventory: LanguageTableInventory,
    expected_counts: dict[str, int],
    written_counts: dict[str, int],
) -> None:
    expected_written = {key: value for key, value in expected_counts.items() if value}
    if written_counts != expected_written:
        raise V1LanguageSplitError(
            f"partition counts changed for {table_inventory.table.value}: "
            f"expected={expected_counts}, observed={written_counts}"
        )


def _build_staged_partitions(
    release_root: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    schema: pa.Schema,
    staged_paths: dict[str, Path],
    written_counts: dict[str, int],
) -> tuple[_StagedPartition, ...]:
    results: list[_StagedPartition] = []
    for language in sorted(written_counts):
        staged_path = staged_paths[language]
        _validate_staged_schema(staged_path, schema, spec.table.value, language)
        split = f"lang-{language}"
        final_path = release_root / _relative_partition_path(spec.configuration, split)
        results.append(
            _StagedPartition(
                artifact=V1PartitionFile(
                    table=spec.table.value,
                    configuration=spec.configuration,
                    split=split,
                    path=final_path,
                    row_count=written_counts[language],
                    sha256=sha256_file(staged_path),
                    columns=tuple(schema.names),
                ),
                staged_path=staged_path,
            )
        )
    return tuple(results)


def _stream_source_file(
    source_path: Path,
    spec: LanguageTableSpec,
    schema: pa.Schema,
    stage_root: Path,
    batch_size: int,
    writers: dict[str, Any],
    staged_paths: dict[str, Path],
    written_counts: dict[str, int],
) -> int:
    try:
        with pq.ParquetFile(source_path) as parquet_file:
            _validate_source_schema(parquet_file, schema, source_path)
            return _stream_batches(
                parquet_file,
                spec,
                schema,
                batch_size,
                stage_root,
                writers,
                staged_paths,
                written_counts,
            )
    except V1LanguageSplitError:
        raise
    except Exception as error:
        raise V1LanguageSplitError(
            f"Could not partition language column {spec.language_column!r} in {source_path}: {error}"
        ) from error


def _validate_source_schema(
    parquet_file: pq.ParquetFile,
    schema: pa.Schema,
    source_path: Path,
) -> None:
    if not parquet_file.schema_arrow.equals(schema, check_metadata=True):
        raise V1LanguageSplitError(f"schema mismatch for source artifact {source_path}")


def _stream_batches(
    parquet_file: pq.ParquetFile,
    spec: LanguageTableSpec,
    schema: pa.Schema,
    batch_size: int,
    stage_root: Path,
    writers: dict[str, Any],
    staged_paths: dict[str, Path],
    written_counts: dict[str, int],
) -> int:
    observed_rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        observed_rows += _stream_batch(
            batch,
            spec,
            schema,
            stage_root,
            writers,
            staged_paths,
            written_counts,
        )
    return observed_rows


def _stream_batch(
    batch: pa.RecordBatch,
    spec: LanguageTableSpec,
    schema: pa.Schema,
    stage_root: Path,
    writers: dict[str, Any],
    staged_paths: dict[str, Path],
    written_counts: dict[str, int],
) -> int:
    language_index = batch.schema.get_field_index(spec.language_column)
    if language_index < 0:
        raise V1LanguageSplitError(f"source batch has no {spec.language_column!r} column")
    groups = _row_indices_by_language(batch.column(language_index).to_pylist())
    for language in sorted(groups):
        writer = _writer_for_language(
            language,
            spec,
            schema,
            stage_root,
            writers,
            staged_paths,
        )
        selected = batch.take(pa.array(groups[language], type=pa.int64()))
        writer.write_batch(selected)
        written_counts[language] = written_counts.get(language, 0) + len(groups[language])
    return batch.num_rows


def _writer_for_language(
    language: str,
    spec: LanguageTableSpec,
    schema: pa.Schema,
    stage_root: Path,
    writers: dict[str, Any],
    staged_paths: dict[str, Path],
) -> Any:
    writer = writers.get(language)
    if writer is not None:
        return writer
    staged_path = stage_root / _relative_partition_path(spec.configuration, f"lang-{language}")
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _new_writer(staged_path, schema)
    writers[language] = writer
    staged_paths[language] = staged_path
    return writer


def _row_indices_by_language(values: list[object]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        language = normalize_language(value).partition
        groups.setdefault(language, []).append(index)
    return groups


def _new_writer(path: Path, schema: pa.Schema) -> Any:
    return pq.ParquetWriter(
        str(path),
        schema,
        compression=_PARQUET_COMPRESSION,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=True,
        write_statistics=True,
    )


def _close_writers(writers: dict[str, Any]) -> None:
    for language in sorted(writers):
        writers[language].close()


def _validate_staged_schema(
    path: Path,
    schema: pa.Schema,
    table: str,
    language: str,
) -> None:
    actual = _read_staged_schema(path)
    if not actual.equals(schema, check_metadata=True):
        raise V1LanguageSplitError(
            f"generated {table} {language} artifact has a schema mismatch: {path}"
        )


def _read_staged_schema(path: Path) -> pa.Schema:
    try:
        return pq.read_schema(path)
    except Exception as error:
        raise V1LanguageSplitError(
            f"Could not validate generated artifact {path}: {error}"
        ) from error


def _relative_partition_path(configuration: str, split: str) -> Path:
    return Path(V1_LANGUAGE_DATA_DIR) / configuration / f"{split}{V1_LANGUAGE_FILE_SUFFIX}"


def _manifest_payload(
    release_root: Path,
    inventory: LanguageInventory,
    files: tuple[V1PartitionFile, ...],
) -> dict[str, object]:
    return {
        "contract": DatasetContract.V1.value,
        "dataset_id": DATASET_V1_ID,
        "contract_version": LANGUAGE_SPLIT_CONTRACT_VERSION,
        "manifest_path": V1_LANGUAGE_SPLIT_MANIFEST,
        "files": [
            item.to_dict(release_root)
            for item in sorted(files, key=lambda item: item.path.as_posix())
        ],
        "inventory": inventory.to_dict(),
    }


def _install_staged_files(release_root: Path, staged: dict[Path, Path]) -> None:
    previous = _previous_partition_paths(release_root)
    final_paths = set(staged)
    stale = sorted(previous - final_paths, key=lambda path: path.as_posix())
    targets = sorted((*final_paths, *stale), key=lambda path: path.as_posix())
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        _backup_targets(targets, backups)
        _install_files(staged, installed)
    except BaseException:
        _restore_files(installed, backups)
        raise
    finally:
        _cleanup_transaction(staged, backups)


def _backup_targets(targets: list[Path], backups: dict[Path, Path]) -> None:
    for target in targets:
        if target.exists():
            backups[target] = _backup_existing(target)


def _install_files(staged: dict[Path, Path], installed: list[Path]) -> list[Path]:
    for final, temporary in sorted(staged.items(), key=lambda item: item[0].as_posix()):
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        installed.append(final)
    return installed


def _restore_files(installed: list[Path], backups: dict[Path, Path]) -> None:
    _remove_installed_files(installed)
    _restore_backups(backups)


def _remove_installed_files(installed: list[Path]) -> None:
    for final in installed:
        final.unlink(missing_ok=True)


def _restore_backups(backups: dict[Path, Path]) -> None:
    for final, backup in sorted(backups.items(), key=lambda item: item[0].as_posix()):
        if backup.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, final)


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
    os.replace(path, backup)
    return backup


def _previous_partition_paths(output_root: Path) -> set[Path]:
    files = _read_previous_files(output_root / V1_LANGUAGE_SPLIT_MANIFEST)
    return {
        candidate
        for item in files or ()
        if (candidate := _previous_partition_path(output_root, item)) is not None
    }


def _read_previous_files(manifest: Path) -> list[object] | None:
    if not manifest.is_file():
        return None
    try:
        payload = json_loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    files = payload.get("files")
    return files if isinstance(files, list) else None


def _previous_partition_path(output_root: Path, item: object) -> Path | None:
    raw_path = _previous_raw_path(item)
    if raw_path is None:
        return None
    candidate = (output_root / raw_path).resolve()
    relative = _relative_to_output(candidate, output_root)
    if relative is None:
        return None
    if not _is_partition_path(candidate, relative):
        return None
    return candidate


def _previous_raw_path(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    raw_path = item.get("path")
    return raw_path if isinstance(raw_path, str) else None


def _relative_to_output(candidate: Path, output_root: Path) -> Path | None:
    try:
        return candidate.relative_to(output_root.resolve())
    except ValueError:
        return None


def _is_partition_path(candidate: Path, relative: Path) -> bool:
    return relative.parts[:1] == (V1_LANGUAGE_DATA_DIR,) and candidate.suffix == ".parquet"


def main(argv: list[str] | None = None) -> int:
    """Run the V1 generator as a local release command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    release = generate_v1_language_splits(
        args.processed_root,
        args.output_root,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
    )
    print(f"Generated {len(release.files)} V1 language files and {release.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "V1_LANGUAGE_DATA_DIR",
    "V1_LANGUAGE_FILE_SUFFIX",
    "V1_LANGUAGE_SPLIT_MANIFEST",
    "V1LanguageSplitError",
    "V1LanguageSplitRelease",
    "V1PartitionFile",
    "generate_v1_language_splits",
    "main",
]
