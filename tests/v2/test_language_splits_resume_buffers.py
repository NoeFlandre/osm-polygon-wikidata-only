"""Tests for language-split resume markers, buffering, and partition helpers."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.hf import language_splits as hf_language_splits
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageTableInventory,
    language_table_specs,
    normalize_language,
)
from osm_polygon_wikidata_only.io.hashing import sha256_file_uncached
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.v2 import (
    language_split_models,
    language_split_resume,
    language_split_writer,
    language_splits,
)
from osm_polygon_wikidata_only.v2.language_splits import (
    V2LanguageSplitFile,
)
from tests.v2.language_splits_support import (
    _resume_file_record,
    _resume_inventory,
    _write_resume_source,
)


def test_v2_partition_ids_are_int32_for_compactness() -> None:
    """Partition ids stay int32: one value per row, and int32 addresses any batch."""
    batch = pa.record_batch([pa.array(["fr", "en", "fr", None])], names=["language"])
    names_by_code, codes = hf_language_splits._partition_names_by_code(batch.column(0))
    ids_by_name: dict[str, int] = {}
    id_of_code = [ids_by_name.setdefault(name, len(ids_by_name)) for name in names_by_code]

    assert hf_language_splits._partition_ids(id_of_code, codes).type == pa.int32()


def test_v2_partition_row_indices_groups_in_first_occurrence_order() -> None:
    """Partitions come back by first appearance with ascending indices."""
    batch = pa.record_batch([pa.array(["fr", "en", "fr", "en", "fr"])], names=["language"])
    partitions = hf_language_splits.partition_row_indices(batch, 0)

    assert list(partitions) == ["fr", "en"]
    assert partitions["fr"].to_pylist() == [0, 2, 4]
    assert partitions["en"].to_pylist() == [1, 3]


def test_v2_partition_row_indices_folds_null_languages_into_one_partition() -> None:
    """Null language values resolve like any other value, keeping row order."""
    batch = pa.record_batch([pa.array([None, "fr", None, "fr"])], names=["language"])
    partitions = hf_language_splits.partition_row_indices(batch, 0)
    unknown = normalize_language(None).partition

    assert partitions[unknown].to_pylist() == [0, 2]
    assert partitions["fr"].to_pylist() == [1, 3]


def test_v2_partition_row_indices_returns_nothing_for_an_empty_batch() -> None:
    """An empty batch has no partitions at all."""
    batch = pa.record_batch([pa.array([], type=pa.string())], names=["language"])

    assert hf_language_splits.partition_row_indices(batch, 0) == {}


def _buffer_shard() -> tuple[Any, Any, list[pa.Table]]:
    """A shard wired to a writer that records every flushed table."""
    written: list[pa.Table] = []

    class Writer:
        def write_table(self, table: pa.Table) -> None:
            written.append(table)

    state = language_split_models.LanguageTableWriteState({}, defaultdict(int), [], {})
    shard = SimpleNamespace(
        source_files=[], row_count=0, writer=Writer(), pending=[], pending_bytes=0
    )
    return state, shard, written


def _rows(count: int) -> pa.RecordBatch:
    return pa.record_batch([pa.array(["en"] * count)], names=["language"])


def test_v2_buffer_rows_accumulates_bytes_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each buffered batch adds to the shard and to the global total."""
    monkeypatch.setattr(language_split_writer, "SHARD_FLUSH_BYTES", 10**9)
    monkeypatch.setattr(language_split_writer, "TOTAL_FLUSH_BYTES", 10**9)
    state, shard, written = _buffer_shard()
    first, second = _rows(2), _rows(3)

    language_split_writer._buffer_rows(state, shard, first)
    language_split_writer._buffer_rows(state, shard, second)

    assert shard.pending_bytes == first.nbytes + second.nbytes
    assert state.pending_bytes == first.nbytes + second.nbytes
    assert [b.num_rows for b in shard.pending] == [2, 3]
    assert written == []


def test_v2_buffer_rows_tracks_every_shard_in_the_global_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global budget sums all open shards, not just the last one."""
    monkeypatch.setattr(language_split_writer, "SHARD_FLUSH_BYTES", 10**9)
    monkeypatch.setattr(language_split_writer, "TOTAL_FLUSH_BYTES", 10**9)
    state, first_shard, _ = _buffer_shard()
    _, second_shard, _ = _buffer_shard()
    batch = _rows(2)

    language_split_writer._buffer_rows(state, first_shard, batch)
    language_split_writer._buffer_rows(state, second_shard, batch)

    assert state.pending_bytes == batch.nbytes * 2


def test_v2_buffer_rows_flushes_when_the_shard_reaches_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reaching the threshold exactly is enough to flush."""
    state, shard, written = _buffer_shard()
    batch = _rows(2)
    monkeypatch.setattr(language_split_writer, "SHARD_FLUSH_BYTES", batch.nbytes)
    monkeypatch.setattr(language_split_writer, "TOTAL_FLUSH_BYTES", 10**9)

    language_split_writer._buffer_rows(state, shard, batch)

    assert [t.num_rows for t in written] == [2]
    assert shard.pending == []
    assert shard.pending_bytes == 0
    assert state.pending_bytes == 0


def test_v2_buffer_rows_drains_every_shard_when_the_global_budget_is_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global ceiling flushes open shards, bounding memory."""
    state, first_shard, first_written = _buffer_shard()
    _, second_shard, second_written = _buffer_shard()
    batch = _rows(2)
    monkeypatch.setattr(language_split_writer, "SHARD_FLUSH_BYTES", 10**9)
    monkeypatch.setattr(language_split_writer, "TOTAL_FLUSH_BYTES", batch.nbytes * 2)
    state.current["a"] = first_shard
    state.current["b"] = second_shard

    language_split_writer._buffer_rows(state, first_shard, batch)
    assert first_written == []
    language_split_writer._buffer_rows(state, second_shard, batch)

    assert [t.num_rows for t in first_written] == [2]
    assert [t.num_rows for t in second_written] == [2]
    assert state.pending_bytes == 0


def test_v2_flush_shard_resets_the_buffer_and_is_a_no_op_when_empty() -> None:
    """Flushing clears the shard's buffer; flushing again writes nothing."""
    state, shard, written = _buffer_shard()
    batch = _rows(4)
    shard.pending.append(batch)
    shard.pending_bytes = batch.nbytes
    state.pending_bytes = batch.nbytes

    language_split_writer._flush_shard(state, shard)
    assert shard.pending == []
    assert shard.pending_bytes == 0
    assert state.pending_bytes == 0
    assert [t.num_rows for t in written] == [4]

    language_split_writer._flush_shard(state, shard)
    assert [t.num_rows for t in written] == [4]


def _fingerprint_inventory(
    source_files: tuple[str, ...] = ("polygon_document_links/a.parquet",),
    row_count: int = 4,
) -> LanguageTableInventory:
    spec = language_table_specs(DatasetContract.V2)[0]
    return LanguageTableInventory(
        table=spec.table,
        configuration=spec.configuration,
        language_column=spec.language_column,
        identity_columns=spec.identity_columns,
        source_files=source_files,
        row_count=row_count,
        buckets=(),
    )


def _write_fingerprint_source(source_root: Path, name: str, content: bytes) -> None:
    path = source_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_table_fingerprint_is_stable_for_identical_inputs(tmp_path: Path) -> None:
    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v1")
    inventory = _fingerprint_inventory()

    first = language_split_resume.table_fingerprint(tmp_path, inventory, 10)
    second = language_split_resume.table_fingerprint(tmp_path, inventory, 10)

    assert first == second
    assert len(first) == 64


def test_table_fingerprint_changes_when_same_size_source_bytes_change(tmp_path: Path) -> None:
    inventory = _fingerprint_inventory()
    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v1")
    before = language_split_resume.table_fingerprint(tmp_path, inventory, 10)

    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v2")

    assert language_split_resume.table_fingerprint(tmp_path, inventory, 10) != before


def test_table_fingerprint_changes_when_max_rows_per_shard_changes(tmp_path: Path) -> None:
    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v1")
    inventory = _fingerprint_inventory()

    assert language_split_resume.table_fingerprint(
        tmp_path, inventory, 10
    ) != language_split_resume.table_fingerprint(tmp_path, inventory, 1)


def test_table_fingerprint_changes_when_the_inventory_changes(tmp_path: Path) -> None:
    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v1")

    assert language_split_resume.table_fingerprint(
        tmp_path, _fingerprint_inventory(row_count=4), 10
    ) != language_split_resume.table_fingerprint(tmp_path, _fingerprint_inventory(row_count=5), 10)


def test_table_fingerprint_covers_every_listed_source_file(tmp_path: Path) -> None:
    """Editing any one of several sources invalidates the table, not only the first."""
    names = ("polygon_document_links/a.parquet", "polygon_document_links/b.parquet")
    _write_fingerprint_source(tmp_path, names[0], b"a-v1")
    _write_fingerprint_source(tmp_path, names[1], b"b-v1")
    inventory = _fingerprint_inventory(source_files=names)
    before = language_split_resume.table_fingerprint(tmp_path, inventory, 10)

    _write_fingerprint_source(tmp_path, names[1], b"b-v2")

    assert language_split_resume.table_fingerprint(tmp_path, inventory, 10) != before


def test_table_fingerprint_reads_sources_from_the_given_root(tmp_path: Path) -> None:
    """The same bytes under the same relative names give the same fingerprint in any root."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        _write_fingerprint_source(root, "polygon_document_links/a.parquet", b"source-v1")
    inventory = _fingerprint_inventory()

    assert language_split_resume.table_fingerprint(
        first_root, inventory, 10
    ) == language_split_resume.table_fingerprint(second_root, inventory, 10)


def test_table_fingerprint_pins_the_persisted_payload_layout(tmp_path: Path) -> None:
    """Resume markers persist this digest, so its payload layout is a compatibility contract."""
    _write_fingerprint_source(tmp_path, "polygon_document_links/a.parquet", b"source-v1")
    inventory = _fingerprint_inventory()
    payload = {
        "inventory": inventory.to_dict(),
        "max_rows_per_shard": 10,
        "source_sha256": {
            "polygon_document_links/a.parquet": sha256_file_uncached(
                tmp_path / "polygon_document_links/a.parquet"
            ),
        },
    }
    expected = hashlib.sha256(json_dumps(payload).encode()).hexdigest()

    assert language_split_resume.table_fingerprint(tmp_path, inventory, 10) == expected


def test_v2_resume_marker_lives_at_a_stable_path() -> None:
    """The marker path is part of the on-disk resume contract."""
    spec = language_table_specs(DatasetContract.V2)[0]
    marker = language_split_resume._resume_marker_path(Path("/stage"), spec)

    assert marker == Path("/stage/.resume") / f"{spec.table.value}.json"


def test_v2_staged_table_reuses_a_completed_table_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded table is returned as-is and the writer is never invoked."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    inventory = _resume_inventory()
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    record = _resume_file_record("language_splits/lang-fr.parquet")
    _write_resume_source(tmp_path)
    language_split_resume.record_completed_table(
        stage_root,
        spec,
        language_split_resume.table_fingerprint(tmp_path, inventory, 10),
        [record],
        {final: staged},
    )

    def fail(*_args: object) -> None:
        raise AssertionError("_write_table must not run for a completed table")

    monkeypatch.setattr(language_splits, "_write_table", fail)

    files, staged_paths = language_splits._staged_table(
        tmp_path, tmp_path / "language_splits", stage_root, spec, inventory, 1, 10
    )

    assert [file.to_dict() for file in files] == [record.to_dict()]
    assert staged_paths == {final: staged}


def test_v2_staged_table_rebuilds_when_the_inventory_no_longer_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker from other sources is ignored and the table is staged again."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    _write_resume_source(tmp_path)
    language_split_resume.record_completed_table(
        stage_root,
        spec,
        language_split_resume.table_fingerprint(tmp_path, _resume_inventory(row_count=4), 10),
        [_resume_file_record("language_splits/lang-fr.parquet")],
        {final: staged},
    )
    calls: list[str] = []

    def rebuilt(*_args: object) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
        calls.append("write")
        return [], {}

    monkeypatch.setattr(language_splits, "_write_table", rebuilt)

    language_splits._staged_table(
        tmp_path,
        tmp_path / "language_splits",
        stage_root,
        spec,
        _resume_inventory(row_count=5),
        1,
        10,
    )

    assert calls == ["write"]


def test_v2_resume_rejects_a_shard_record_missing_its_row_count(tmp_path: Path) -> None:
    """Every required field must be present and correctly typed."""
    spec = language_table_specs(DatasetContract.V2)[0]
    entry = _resume_file_record("language_splits/lang-fr.parquet").to_dict()
    entry["row_count"] = "four"

    assert language_split_resume._resume_file(entry, spec) is None


def test_v2_resume_rejects_a_staged_path_map_with_a_non_string_key(tmp_path: Path) -> None:
    """A malformed key is rejected even when the staged file really exists.

    The staged file is real here on purpose: a map that fails only because the
    path is missing would not prove the key is validated at all.
    """
    staged = tmp_path / "lang-fr.parquet"
    staged.write_bytes(b"staged")

    assert language_split_resume._resume_staged_paths({4: str(staged)}) is None
    assert language_split_resume._resume_staged_paths({str(tmp_path / "f.parquet"): 4}) is None
    assert language_split_resume._resume_staged_paths(
        {str(tmp_path / "f.parquet"): str(staged)}
    ) == {tmp_path / "f.parquet": staged}


def test_v2_resume_payload_rejects_a_marker_that_is_not_an_object(tmp_path: Path) -> None:
    """A marker holding a JSON array is not a payload."""
    spec = language_table_specs(DatasetContract.V2)[0]
    marker = language_split_resume._resume_marker_path(tmp_path, spec)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("[1, 2, 3]", encoding="utf-8")

    assert language_split_resume._resume_payload(tmp_path, spec) is None


def test_v2_resume_payload_is_none_when_no_marker_was_written(tmp_path: Path) -> None:
    """A table that never completed has no marker to resume from."""
    spec = language_table_specs(DatasetContract.V2)[0]

    assert language_split_resume._resume_payload(tmp_path, spec) is None


def test_v2_resume_file_rows_requires_a_list_and_an_integer() -> None:
    """The row accounting of a shard record is validated field by field."""
    assert (
        language_split_resume._resume_file_rows({"source_files": "a.parquet", "row_count": 1})
        is None
    )
    assert (
        language_split_resume._resume_file_rows({"source_files": ["a"], "row_count": "1"}) is None
    )
    assert language_split_resume._resume_file_rows({"source_files": ["a"], "row_count": 2}) == (
        ("a",),
        2,
    )


def test_v2_resume_file_texts_requires_every_field_to_be_text() -> None:
    """A shard record missing any text field is rejected."""
    complete: dict[str, object] = {
        "configuration": "c",
        "language": "fr",
        "split": "lang-fr",
        "path": "p",
        "sha256": "d",
    }
    assert language_split_resume._resume_file_texts(complete) == ("c", "fr", "lang-fr", "p", "d")
    assert language_split_resume._resume_file_texts({**complete, "sha256": None}) is None
    assert language_split_resume._resume_file_texts({}) is None


def test_v2_resume_staged_entry_validates_one_pair(tmp_path: Path) -> None:
    """A staged entry needs string paths and a staged file that exists."""
    staged = tmp_path / "s.parquet"
    staged.write_bytes(b"x")

    assert language_split_resume._resume_staged_entry("final", str(staged)) == (
        Path("final"),
        staged,
    )
    assert language_split_resume._resume_staged_entry(4, str(staged)) is None
    assert language_split_resume._resume_staged_entry("final", 4) is None
    assert language_split_resume._resume_staged_entry("final", str(tmp_path / "gone")) is None


def test_v2_resume_files_rejects_a_non_list_and_a_bad_entry() -> None:
    """The shard list must be a list of well-formed records."""
    spec = language_table_specs(DatasetContract.V2)[0]

    assert language_split_resume._resume_files({"not": "a list"}, spec) is None
    assert language_split_resume._resume_files([{"bad": "entry"}], spec) is None
    assert language_split_resume._resume_files([], spec) == []


def test_v2_staged_table_reports_the_resumed_table_and_shard_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Resuming is announced: operators need to see what was skipped and why."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    inventory = _resume_inventory()
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    _write_resume_source(tmp_path)
    language_split_resume.record_completed_table(
        stage_root,
        spec,
        language_split_resume.table_fingerprint(tmp_path, inventory, 10),
        [_resume_file_record("language_splits/lang-fr.parquet")],
        {final: staged},
    )
    monkeypatch.setattr(language_splits, "_write_table", lambda *a: ([], {}))

    with caplog.at_level(logging.INFO, logger=language_splits.LOGGER.name):
        language_splits._staged_table(
            tmp_path, tmp_path / "language_splits", stage_root, spec, inventory, 1, 10
        )

    # Exact equality, not containment: a padded or re-cased format string is a
    # different operator-facing message and must not pass.
    assert [record.getMessage() for record in caplog.records] == [
        f"Resuming: reusing 1 staged shards for {spec.table.value}"
    ]
