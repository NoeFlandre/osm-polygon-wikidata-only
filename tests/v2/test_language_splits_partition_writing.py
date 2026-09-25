"""Tests for language-split partitioning and bounded Parquet writing."""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageBucket,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.v2 import language_splits
from osm_polygon_wikidata_only.v2.language_splits import (
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
)


def test_v2_partition_batch_returns_explicit_wide_row_indices() -> None:
    """Partition indices carry an explicit 64-bit integer type.

    ``RecordBatch.take`` is driven by these arrays. A narrow or inferred index
    type would silently truncate on batches larger than that type can address,
    so the width is part of the contract rather than an accident of inference.
    """
    batch = pa.record_batch([pa.array(["en", "fr", "en", None])], names=["language"])
    partitions = language_splits._partition_batch(batch, 0)
    assert partitions
    for indices in partitions.values():
        assert indices.type in (pa.int64(), pa.uint64())


def test_v2_partition_batch_groups_rows_in_first_occurrence_order() -> None:
    """Partitions are ordered by first appearance and indices stay ascending."""
    batch = pa.record_batch([pa.array(["fr", "en", "fr", "en", "fr"])], names=["language"])
    partitions = language_splits._partition_batch(batch, 0)
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

    assert language_splits._table_shard_counts(inventory, 100_000) == {"fr": 1}


def test_v2_write_language_indices_respects_existing_shard_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"] * 8)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})

    class FakeWriter:
        def __init__(self) -> None:
            self.rows: list[list[dict[str, object]]] = []

        def write_table(self, selected: pa.Table) -> None:
            self.rows.append(selected.to_pylist())

        def close(self) -> None:
            return None

    first = SimpleNamespace(
        source_files=[], row_count=3, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    second = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    state.current["en"] = first
    writers = iter((first, second))
    monkeypatch.setattr(language_splits, "_writer_for_language", lambda *args: next(writers))

    with ExitStack() as stack:
        language_splits._write_language_indices(
            "en",
            pa.array(range(8), type=pa.int64()),
            batch,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source",
            10,
            {"en": 2},
            batch.schema,
            state,
            stack,
        )

    # The filled shard is flushed when it closes; the partial one stays buffered
    # until the table finishes, which is what keeps writes large and sequential.
    assert [len(rows) for rows in first.writer.rows] == [7]
    assert second.writer.rows == []
    assert [b.num_rows for b in second.pending] == [1]
    language_splits._flush_shard(state, second)
    assert [len(rows) for rows in second.writer.rows] == [1]
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
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})

    class FakeWriter:
        def write_table(self, selected: pa.Table) -> None:
            return None

        def close(self) -> None:
            return None

    first = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    second = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    state.current["en"] = first
    writers = iter((first, second))

    def fake_writer_for_language(*args: object) -> SimpleNamespace:
        try:
            return next(writers)
        except StopIteration as error:
            raise AssertionError("offset did not advance across shards") from error

    monkeypatch.setattr(language_splits, "_writer_for_language", fake_writer_for_language)

    with ExitStack() as stack:
        language_splits._write_language_indices(
            "en",
            pa.array(range(3), type=pa.int64()),
            batch,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source",
            2,
            {"en": 2},
            batch.schema,
            state,
            stack,
        )

    assert first.row_count == 2
    assert second.row_count == 1


def test_v2_record_source_file_deduplicates_and_records_transitions() -> None:
    shard = SimpleNamespace(source_files=["source-a.parquet"])

    language_splits._record_source_file(shard, "source-a.parquet")
    language_splits._record_source_file(shard, "source-b.parquet")

    assert shard.source_files == ["source-a.parquet", "source-b.parquet"]


def test_v2_writer_uses_snappy_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})
    observed: dict[str, object] = {}

    @contextmanager
    def fake_atomic(path: Path):
        observed["temporary"] = path
        yield path

    class FakeWriter:
        def __init__(self, path: Path, schema: pa.Schema, *, compression: str) -> None:
            observed["writer"] = (path, schema, compression)

    monkeypatch.setattr(language_splits, "atomic_replacement", fake_atomic)
    monkeypatch.setattr(language_splits.pq, "ParquetWriter", FakeWriter)

    with ExitStack() as stack:
        writer = language_splits._writer_for_language(
            "en",
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            10,
            {"en": 1},
            spec.schema_factory(),
            state,
            stack,
        )

    assert writer is state.current["en"]
    assert writer.shard_index == 0
    assert observed["writer"] == (
        tmp_path / "stage" / spec.configuration / "lang-en" / "part-00000-of-00001.parquet",
        spec.schema_factory(),
        "snappy",
    )


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
