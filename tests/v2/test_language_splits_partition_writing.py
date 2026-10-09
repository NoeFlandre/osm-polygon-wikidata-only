"""Tests for language-split partitioning and bounded Parquet writing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageBucket,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.v2 import (
    language_split_models,
    language_split_writer,
    language_splits,
)
from osm_polygon_wikidata_only.v2.language_splits import (
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
)


def _shard_state(
    writer: object,
    *,
    row_count: int = 0,
    source_files: list[str] | None = None,
) -> language_split_models.LanguageShardWriteState:
    return language_split_models.LanguageShardWriteState(
        language="en",
        shard_index=0,
        writer=cast(pq.ParquetWriter, writer),
        final_path=Path("final.parquet"),
        staged_path=Path("staged.parquet"),
        row_count=row_count,
        source_files=[] if source_files is None else source_files,
    )


def test_v2_partition_batch_returns_explicit_wide_row_indices() -> None:
    """Partition indices carry an explicit 64-bit integer type.

    ``RecordBatch.take`` is driven by these arrays. A narrow or inferred index
    type would silently truncate on batches larger than that type can address,
    so the width is part of the contract rather than an accident of inference.
    """
    batch = pa.record_batch([pa.array(["en", "fr", "en", None])], names=["language"])
    partitions = language_split_writer._partition_batch(batch, 0)
    assert partitions
    for indices in partitions.values():
        assert indices.type in (pa.int64(), pa.uint64())


def test_v2_partition_batch_groups_rows_in_first_occurrence_order() -> None:
    """Partitions are ordered by first appearance and indices stay ascending."""
    batch = pa.record_batch([pa.array(["fr", "en", "fr", "en", "fr"])], names=["language"])
    partitions = language_split_writer._partition_batch(batch, 0)
    assert list(partitions) == ["fr", "en"]
    assert partitions["fr"].to_pylist() == [0, 2, 4]
    assert partitions["en"].to_pylist() == [1, 3]


def test_v2_table_shard_counts_omit_empty_buckets() -> None:
    bucket = LanguageBucket(
        language="en",
        row_count=0,
        canonical_rows=0,
        legacy_alias_rows=0,
        missing_rows=0,
        blank_rows=0,
        malformed_rows=0,
        legacy_unusable_rows=0,
    )
    nonempty = LanguageBucket(
        language="fr",
        row_count=1,
        canonical_rows=1,
        legacy_alias_rows=0,
        missing_rows=0,
        blank_rows=0,
        malformed_rows=0,
        legacy_unusable_rows=0,
    )
    inventory = LanguageTableInventory(
        table=LanguageTable.WIKIPEDIA_DOCUMENTS,
        configuration="wikipedia_documents_by_language",
        language_column="language",
        identity_columns=("document_id",),
        source_files=(),
        row_count=1,
        buckets=(bucket, nonempty),
    )

    assert language_split_writer._table_shard_counts(inventory, 100_000) == {"fr": 1}


def test_v2_table_write_context_is_frozen(tmp_path: Path) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    shard_counts = {"en": 2}
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=8,
        shard_counts=shard_counts,
        expected_schema=spec.schema_factory(),
    )

    assert context.destination == tmp_path / "destination"
    assert context.shard_counts == {"en": 2}
    shard_counts["en"] = 3
    assert context.shard_counts == {"en": 2}
    with pytest.raises(TypeError):
        cast(dict[str, int], context.shard_counts)["en"] = 4
    with pytest.raises(FrozenInstanceError):
        setattr(context, "max_rows_per_shard", 9)


def test_v2_write_language_indices_respects_existing_shard_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"] * 8)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=10,
        shard_counts={"en": 2},
        expected_schema=batch.schema,
    )
    writer = language_split_writer._LanguageTableWriter(context, batch_size=8)
    state = writer.state

    class FakeWriter:
        def __init__(self) -> None:
            self.rows: list[list[dict[str, object]]] = []

        def write_table(self, selected: pa.Table) -> None:
            self.rows.append(selected.to_pylist())

        def close(self) -> None:
            return None

    first_writer = FakeWriter()
    second_writer = FakeWriter()
    first = _shard_state(first_writer, row_count=3)
    second = _shard_state(second_writer)
    state.current["en"] = first
    writers = iter((first, second))

    def fake_writer_for_language(_language: str) -> language_split_models.LanguageShardWriteState:
        return next(writers)

    monkeypatch.setattr(writer, "writer_for_language", fake_writer_for_language)
    writer.write_batch(batch, language_index=0, source_file="source")

    # The filled shard is flushed when it closes; the partial one stays buffered
    # until the table finishes, which is what keeps writes large and sequential.
    assert [len(rows) for rows in first_writer.rows] == [7]
    assert second_writer.rows == []
    assert [b.num_rows for b in second.pending] == [1]
    language_split_writer._flush_shard(state, second)
    assert [len(rows) for rows in second_writer.rows] == [1]
    assert second.pending == []
    assert state.pending_bytes == 0
    assert first.row_count == 10
    assert second.row_count == 1
    assert "en" not in state.current


def test_v2_write_language_indices_advances_offset_across_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"] * 3)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=2,
        shard_counts={"en": 2},
        expected_schema=batch.schema,
    )
    writer = language_split_writer._LanguageTableWriter(context, batch_size=3)

    class FakeWriter:
        def write_table(self, selected: pa.Table) -> None:
            return None

        def close(self) -> None:
            return None

    first = _shard_state(FakeWriter())
    second = _shard_state(FakeWriter())
    writer.state.current["en"] = first
    writers = iter((first, second))

    def fake_writer_for_language(_language: str) -> language_split_models.LanguageShardWriteState:
        try:
            return next(writers)
        except StopIteration as error:
            raise AssertionError("offset did not advance across shards") from error

    monkeypatch.setattr(writer, "writer_for_language", fake_writer_for_language)
    writer.write_batch(batch, language_index=0, source_file="source")

    assert first.row_count == 2
    assert second.row_count == 1


class _BoundedIndices:
    """Row indices that fail on the first read past their end instead of looping."""

    def __init__(self, values: pa.Array) -> None:
        self.values = values
        self.offsets: list[int] = []

    def __len__(self) -> int:
        return len(self.values)

    def slice(self, offset: int, length: int) -> pa.Array:
        self.offsets.append(offset)
        if offset >= len(self.values):
            raise AssertionError(f"partition indices read past their end at offset {offset}")
        return self.values.slice(offset, length)


def test_v2_write_language_indices_stops_at_the_last_partition_row(tmp_path: Path) -> None:
    """Each partition row is taken once, and the writer never reads past the last one.

    A loop that treats the end offset as in range asks for an empty slice on
    every pass and never returns, so the read past the end must fail here.
    """
    batch = pa.record_batch([pa.array(["en", "en", "en"])], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=10,
        shard_counts={"en": 1},
        expected_schema=batch.schema,
    )
    writer = language_split_writer._LanguageTableWriter(context, batch_size=3)
    shard = _shard_state(object())
    writer.state.current["en"] = shard
    indices = _BoundedIndices(pa.array([0, 1, 2], type=pa.int64()))

    writer._write_language_indices("en", cast(pa.Array, indices), batch, "source")

    assert shard.row_count == 3
    assert shard.source_files == ["source"]
    assert [rows.num_rows for rows in shard.pending] == [3]
    assert indices.offsets == [0]


def test_v2_record_source_file_deduplicates_and_records_transitions() -> None:
    shard = _shard_state(object(), source_files=["source-a.parquet"])

    language_split_writer._record_source_file(shard, "source-a.parquet")
    language_split_writer._record_source_file(shard, "source-b.parquet")

    assert shard.source_files == ["source-a.parquet", "source-b.parquet"]


def test_v2_writer_uses_snappy_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=10,
        shard_counts={"en": 1},
        expected_schema=spec.schema_factory(),
    )
    writer = language_split_writer._LanguageTableWriter(context, batch_size=10)
    observed: dict[str, object] = {}

    @contextmanager
    def fake_atomic(path: Path):
        observed["temporary"] = path
        yield path

    class FakeWriter:
        def __init__(self, path: Path, schema: pa.Schema, *, compression: str) -> None:
            observed["writer"] = (path, schema, compression)

    monkeypatch.setattr(language_split_writer, "atomic_replacement", fake_atomic)
    monkeypatch.setattr(language_split_writer.pq, "ParquetWriter", FakeWriter)

    with writer._stack:
        shard = writer.writer_for_language("en")

    assert shard is writer.state.current["en"]
    assert shard.shard_index == 0
    assert observed["writer"] == (
        tmp_path / "stage" / spec.configuration / "lang-en" / "part-00000-of-00001.parquet",
        spec.schema_factory(),
        "snappy",
    )


def test_v2_writer_discards_staged_file_when_source_stream_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"])], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    root = tmp_path / "root"
    root.mkdir()
    context = language_split_writer.TableWriteContext(
        destination=tmp_path / "destination",
        stage_root=tmp_path / "stage",
        spec=spec,
        max_rows_per_shard=10,
        shard_counts={"en": 1},
        expected_schema=batch.schema,
    )
    inventory = LanguageTableInventory(
        table=spec.table,
        configuration=spec.configuration,
        language_column=spec.language_column,
        identity_columns=spec.identity_columns,
        source_files=("source.parquet",),
        row_count=1,
        buckets=(),
    )
    staged = context.stage_root / spec.configuration / "lang-en" / "part-00000-of-00001.parquet"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"previous staged output")
    events: list[str] = []

    @contextmanager
    def fake_open_parquet(_path: Path):
        yield SimpleNamespace(schema_arrow=batch.schema)

    def failing_batches(_parquet_file: object, *, batch_size: int):
        assert batch_size == 1
        yield batch
        raise RuntimeError("source stream failed")

    class FakeWriter:
        def write_table(self, selected: pa.Table) -> None:
            events.append(f"write:{selected.to_pylist()}")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(language_split_writer, "open_parquet", fake_open_parquet)
    monkeypatch.setattr(language_split_writer, "iter_record_batches", failing_batches)
    monkeypatch.setattr(language_split_writer.pq, "ParquetWriter", lambda *_a, **_kw: FakeWriter())
    writer = language_split_writer._LanguageTableWriter(context, batch_size=1)

    with pytest.raises(RuntimeError, match="source stream failed"):
        writer.stream_sources(root, inventory)

    final = context.destination / spec.configuration / "lang-en" / staged.name
    assert events == ["write:[{'language': 'en'}]", "close"]
    assert staged.read_bytes() == b"previous staged output"
    assert not final.exists()
    assert not list(staged.parent.glob(f".{staged.name}.*.tmp"))


def test_v2_staging_reuses_an_existing_stage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second release must adopt the staging tree an interrupted run left.

    This is what makes resuming possible: the staging root already exists on
    the retry, so creating it must tolerate that rather than fail.
    """
    root = tmp_path / "processed_v2"
    root.mkdir()
    destination = root / "nested/language_splits"
    stage_root = destination.parent / ".language_splits-staging"
    stage_root.mkdir(parents=True)
    (stage_root / "left-over.parquet").write_bytes(b"from the interrupted run")

    monkeypatch.setattr(language_splits, "_stage_v2_files", lambda *args: ([], {}))
    monkeypatch.setattr(language_splits, "_validate_conservation", lambda *args: None)
    monkeypatch.setattr(language_splits, "_manifest_payload", lambda *args: {})
    monkeypatch.setattr(language_splits, "atomic_write_json", lambda *args: None)
    monkeypatch.setattr(language_splits, "_install_staged_files", lambda *args: None)
    monkeypatch.setattr(
        language_splits, "_verify_source_inventory", lambda root, expected: expected
    )
    monkeypatch.setattr(language_splits.shutil, "rmtree", lambda path, **_kw: None)

    inventory = cast(LanguageInventory, object())
    result = language_splits._stage_and_install_v2_release(
        root,
        destination,
        inventory,
        1,
        100_000,
        root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    )

    assert result == (inventory, ())
    assert (stage_root / "left-over.parquet").is_file()
