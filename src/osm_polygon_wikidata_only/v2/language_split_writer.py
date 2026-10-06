"""Stream V2 source rows into bounded, deterministic language partition shards."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

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


@dataclass(frozen=True, slots=True)
class TableWriteContext:
    """Immutable configuration shared by every shard in one table write."""

    destination: Path
    stage_root: Path
    spec: LanguageTableSpec
    max_rows_per_shard: int
    shard_counts: Mapping[str, int]
    expected_schema: pa.Schema

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_counts", MappingProxyType(dict(self.shard_counts)))

    @classmethod
    def for_inventory(
        cls,
        destination: Path,
        stage_root: Path,
        spec: LanguageTableSpec,
        table_inventory: LanguageTableInventory,
        max_rows_per_shard: int,
    ) -> TableWriteContext:
        """Build the per-table settings once, before opening staged writers."""
        expected_schema = spec.schema_factory()
        shard_counts = _table_shard_counts(table_inventory, max_rows_per_shard)
        return cls(
            destination=destination,
            stage_root=stage_root,
            spec=spec,
            max_rows_per_shard=max_rows_per_shard,
            shard_counts=shard_counts,
            expected_schema=expected_schema,
        )


def write_table(
    root: Path,
    context: TableWriteContext,
    table_inventory: LanguageTableInventory,
    batch_size: int,
) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
    writer = _LanguageTableWriter(context, batch_size)
    writer.stream_sources(root, table_inventory)

    files = [
        _validated_output_file(root, context.spec, shard, context.expected_schema)
        for shard in writer.state.shards
    ]
    return files, writer.state.staged_paths


def _table_shard_counts(
    table_inventory: LanguageTableInventory, max_rows_per_shard: int
) -> dict[str, int]:
    shard_counts: dict[str, int] = {}
    for bucket in table_inventory.buckets:
        if bucket.row_count > 0:
            shard_counts[bucket.language] = _shard_count(bucket.row_count, max_rows_per_shard)
    return shard_counts


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


class _LanguageTableWriter:
    """Own the buffers and staged writers for one language-bearing table."""

    def __init__(self, context: TableWriteContext, batch_size: int) -> None:
        self.context = context
        self.batch_size = batch_size
        self.state = LanguageTableWriteState({}, defaultdict(int), [], {})
        self._stack = ExitStack()

    def stream_sources(self, root: Path, table_inventory: LanguageTableInventory) -> None:
        """Stream one table and close its writers before staging contexts exit."""
        with self._stack:
            try:
                for source_file in table_inventory.source_files:
                    source_path = (root / source_file).resolve()
                    _ensure_source_is_under_root(source_path, root)
                    self.stream_source_file(source_path, source_file)
            finally:
                _close_current_writers(self.state)

    def stream_source_file(self, source_path: Path, source_file: str) -> None:
        """Stream one validated source file into this table's language writers."""
        with open_parquet(source_path) as parquet_file:
            language_index = parquet_file.schema_arrow.get_field_index(
                self.context.spec.language_column
            )
            for batch in iter_record_batches(parquet_file, batch_size=self.batch_size):
                self.write_batch(batch, language_index, source_file)

    def write_batch(self, batch: pa.RecordBatch, language_index: int, source_file: str) -> None:
        for language, indices in _partition_batch(batch, language_index).items():
            self._write_language_indices(language, indices, batch, source_file)

    def _write_language_indices(
        self,
        language: str,
        indices: pa.Array,
        batch: pa.RecordBatch,
        source_file: str,
    ) -> None:
        offset = 0
        while offset < len(indices):
            shard = self.writer_for_language(language)
            _record_source_file(shard, source_file)
            available = self.context.max_rows_per_shard - shard.row_count
            count = min(available, len(indices) - offset)
            _buffer_rows(self.state, shard, batch.take(indices.slice(offset, count)))
            shard.row_count += count
            offset += count
            _close_full_shard(shard, language, self.context.max_rows_per_shard, self.state)

    def writer_for_language(self, language: str) -> LanguageShardWriteState:
        current = self.state.current.get(language)
        if current is not None:
            return current
        shard_index = self.state.next_shard_index[language]
        self.state.next_shard_index[language] += 1
        final_path = _output_path(
            self.context.destination,
            self.context.spec,
            language,
            shard_index,
            self.context.shard_counts[language],
        )
        staged_path = _output_path(
            self.context.stage_root,
            self.context.spec,
            language,
            shard_index,
            self.context.shard_counts[language],
        )
        temporary = self._stack.enter_context(atomic_replacement(staged_path))
        shard = LanguageShardWriteState(
            language=language,
            shard_index=shard_index,
            writer=pq.ParquetWriter(temporary, self.context.expected_schema, compression="snappy"),
            final_path=final_path,
            staged_path=staged_path,
            row_count=0,
            source_files=[],
        )
        self.state.current[language] = shard
        self.state.shards.append(shard)
        self.state.staged_paths[final_path] = staged_path
        return shard


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
