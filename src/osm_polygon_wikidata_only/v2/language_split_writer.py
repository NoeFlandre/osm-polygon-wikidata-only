"""Stream V2 source rows into bounded, deterministic language partition shards."""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.language_splits import (
    LanguageTableInventory,
    LanguageTableSpec,
    language_split_name,
    partition_row_indices,
)
from osm_polygon_wikidata_only.io.atomic import atomic_replacement
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    ensure_source_is_under_root as _ensure_source_is_under_root,
)
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    shard_count as _shard_count,
)
from osm_polygon_wikidata_only.v2.language_split_manifest import (
    validate_output_file as _validated_output_file,
)
from osm_polygon_wikidata_only.v2.language_split_models import (
    SHARD_FLUSH_BYTES,
    TOTAL_FLUSH_BYTES,
    LanguageShardWriteState,
    LanguageTableWriteState,
    V2LanguageSplitFile,
)


def write_table(
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
    state = LanguageTableWriteState({}, defaultdict(int), [], {})
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
    state: LanguageTableWriteState,
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
    state: LanguageTableWriteState,
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


def _close_current_writers(state: LanguageTableWriteState) -> None:
    for shard in state.current.values():
        _flush_shard(state, shard)
        shard.writer.close()


def _buffer_rows(
    state: LanguageTableWriteState, shard: LanguageShardWriteState, rows: pa.RecordBatch
) -> None:
    """Hold ``rows`` until the shard has enough to justify one large write.

    A source batch fans out across every language it mentions, so writing each
    slice straight through produced one tiny row group per language per batch --
    hundreds of interleaved small writes across hundreds of open files. Batches
    are accumulated per shard instead and written once they are worth a seek.
    """
    shard.pending.append(rows)
    shard.pending_bytes += rows.nbytes
    state.pending_bytes += rows.nbytes
    if shard.pending_bytes >= SHARD_FLUSH_BYTES:
        _flush_shard(state, shard)
        return
    if state.pending_bytes >= TOTAL_FLUSH_BYTES:
        _flush_all_shards(state)


def _flush_shard(state: LanguageTableWriteState, shard: LanguageShardWriteState) -> None:
    """Write one shard's buffered batches as a single row group."""
    if not shard.pending:
        return
    table = pa.Table.from_batches(shard.pending)
    shard.writer.write_table(table)
    state.pending_bytes -= shard.pending_bytes
    shard.pending = []
    shard.pending_bytes = 0


def _flush_all_shards(state: LanguageTableWriteState) -> None:
    """Drain every open shard so buffered rows stay inside a fixed budget."""
    for shard in list(state.current.values()):
        _flush_shard(state, shard)


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
    state: LanguageTableWriteState,
    stack: ExitStack,
) -> None:
    """Stream one validated source file into its language writers."""
    with open_parquet(source_path) as parquet_file:
        language_index = parquet_file.schema_arrow.get_field_index(spec.language_column)
        for batch in iter_record_batches(parquet_file, batch_size=batch_size):
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
    state: LanguageTableWriteState,
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
    indices: pa.Array,
    batch: pa.RecordBatch,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    source_file: str,
    max_rows_per_shard: int,
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: LanguageTableWriteState,
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
        _buffer_rows(state, shard, batch.take(indices.slice(offset, count)))
        shard.row_count += count
        offset += count
        _close_full_shard(shard, language, max_rows_per_shard, state)


def _record_source_file(shard: LanguageShardWriteState, source_file: str) -> None:
    if not shard.source_files or shard.source_files[-1] != source_file:
        shard.source_files.append(source_file)


def _close_full_shard(
    shard: LanguageShardWriteState,
    language: str,
    max_rows_per_shard: int,
    state: LanguageTableWriteState,
) -> None:
    if shard.row_count == max_rows_per_shard:
        _flush_shard(state, shard)
        shard.writer.close()
        state.current.pop(language)


def _writer_for_language(
    language: str,
    destination: Path,
    stage_root: Path,
    spec: LanguageTableSpec,
    max_rows_per_shard: int,  # noqa: ARG001 -- writer factory signature kept uniform
    shard_counts: dict[str, int],
    expected_schema: pa.Schema,
    state: LanguageTableWriteState,
    stack: ExitStack,
) -> LanguageShardWriteState:
    current = state.current.get(language)
    if current is not None:
        return current
    shard_index = state.next_shard_index[language]
    state.next_shard_index[language] += 1
    final_path = _output_path(destination, spec, language, shard_index, shard_counts[language])
    staged_path = _output_path(stage_root, spec, language, shard_index, shard_counts[language])
    temporary = stack.enter_context(atomic_replacement(staged_path))
    shard = LanguageShardWriteState(
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


def _partition_batch(batch: pa.RecordBatch, language_index: int) -> dict[str, pa.Array]:
    """Group row indices by language partition for this batch.

    The grouping itself is shared with the V1 splitter; see
    :func:`osm_polygon_wikidata_only.hf.language_splits.partition_row_indices`.
    """
    return partition_row_indices(batch, language_index)


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
