from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_wikidata_only.v2.sentence_checkpoints as checkpoint_module
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.sentence_checkpoints import (
    SentenceCheckpoint,
    _load_checkpoint_rows,
    _requested_batch_count,
    _validate_component,
    _validated_batch_indexes,
)
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema


def test_sentence_checkpoint_reuses_rows_and_empty_batches(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    rows = [{"sentence_id": "sentence-1", "text": "First."}]

    checkpoint.write_batch(0, rows)
    checkpoint.write_batch(1, [])
    checkpoint.mark_complete(batch_count=2, row_count=1)

    recovered = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )

    assert recovered.complete
    assert recovered.completed_batches == (0, 1)
    loaded = recovered.load_batch(0)
    assert loaded is not None
    assert loaded[0]["sentence_id"] == "sentence-1"
    assert loaded[0]["text"] == "First."
    assert recovered.load_batch(1) == []
    assert len(recovered.load_rows()) == 1


def test_sentence_checkpoint_resets_when_contract_changes(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    kwargs = {
        "input_fingerprint": "input-a",
        "model_id": "segment-any-text/sat-3l-sm",
        "model_revision": "model-a",
        "batch_size": 2,
    }
    checkpoint = SentenceCheckpoint(root, "region-latest", "wikipedia", **kwargs)
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1"}])
    checkpoint.mark_complete(batch_count=1, row_count=1)

    changed = SentenceCheckpoint(
        root,
        "region-latest",
        "wikipedia",
        **(kwargs | {"input_fingerprint": "input-b"}),
    )

    assert not changed.complete
    assert changed.completed_batches == ()


@pytest.mark.parametrize("value", ["", ".", "..", "nested/name", "nested\\name"])
def test_sentence_checkpoint_rejects_path_components(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid sentence checkpoint stem"):
        _validate_component(value, "stem")


def test_sentence_checkpoint_finalization_keeps_only_restart_metadata(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1"}])
    checkpoint.mark_complete(batch_count=1, row_count=1)
    output = tmp_path / "sentences.parquet"
    output.write_bytes(b"final output")
    output_hash = sha256_file(output)

    checkpoint.finalize(output, output_hash=output_hash, summary={"sentence_rows": 1})

    assert checkpoint.complete
    assert checkpoint.completed_batches == ()
    assert checkpoint.output_matches(output, output_hash=output_hash)
    assert checkpoint.metadata["summary"] == {"sentence_rows": 1}

    reloaded = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    assert reloaded.metadata["output_hash"] == output_hash
    assert reloaded.metadata["summary"] == {"sentence_rows": 1}


def test_sentence_checkpoint_rejects_non_contiguous_completion(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(1, [])

    with pytest.raises(ValueError, match="Sentence checkpoint batches are not contiguous"):
        checkpoint.mark_complete(batch_count=1, row_count=0)


def test_sentence_checkpoint_rejects_wrong_completion_row_count(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1"}])

    with pytest.raises(ValueError, match="row accounting is invalid"):
        checkpoint.mark_complete(batch_count=1, row_count=2)


def test_sentence_checkpoint_loads_validated_arrow_tables_without_row_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])

    table = checkpoint.load_batch_table(0)

    assert isinstance(table, pa.Table)
    assert table is not None
    assert table.schema.equals(sentence_schema(), check_metadata=True)
    assert table.to_pylist()[0]["text"] == "First."

    monkeypatch.setattr(
        checkpoint,
        "load_batch",
        lambda index: pytest.fail(f"Python rows were materialized for batch {index}"),
    )
    assert checkpoint.completed_batches == (0,)


def test_sentence_checkpoint_validates_batch_metadata_without_loading_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])

    monkeypatch.setattr(
        checkpoint,
        "load_batch_table",
        lambda index: pytest.fail(f"Rows were loaded for batch {index}"),
    )

    assert checkpoint.batch_row_count(0) == 1
    assert checkpoint.completed_batches == (0,)


def test_requested_batch_count_uses_next_partial_batch_index() -> None:
    assert _requested_batch_count(None, complete=False, metadata={}, indexes=(0, 1)) == 2
    assert _requested_batch_count(None, complete=False, metadata={}, indexes=()) == 0


def test_validated_batch_indexes_rejects_a_negative_requested_count() -> None:
    with pytest.raises(ValueError, match="Invalid sentence checkpoint batch count"):
        _validated_batch_indexes((0,), -1)

    assert _validated_batch_indexes((), 0) == ()


def test_validated_batch_indexes_preserves_the_public_error_message() -> None:
    with pytest.raises(ValueError) as error:
        _validated_batch_indexes((0,), -1)

    assert str(error.value) == "Invalid sentence checkpoint batch count"


def test_load_checkpoint_rows_preserves_batch_order(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])

    expected = {field.name: None for field in sentence_schema()}
    expected.update({"sentence_id": "sentence-1", "text": "First."})

    assert _load_checkpoint_rows(checkpoint, (0,)) == [expected]


def test_sentence_checkpoint_rejects_non_contiguous_batches(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(1, [])

    with pytest.raises(ValueError, match="Sentence checkpoint batches are not contiguous"):
        checkpoint.load_rows()


def test_sentence_checkpoint_rejects_invalid_requested_batch_count(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )

    with pytest.raises(ValueError, match="Invalid sentence checkpoint batch count"):
        checkpoint.load_rows(batch_count=-1)


@pytest.mark.parametrize(
    ("complete", "stored_path", "stored_hash", "file_exists", "expected"),
    [
        (False, "{output}", "hash", True, False),
        (True, "other.parquet", "hash", True, False),
        (True, "{output}", "different", True, False),
        (True, "{output}", "hash", False, False),
    ],
)
def test_output_matches_requires_every_recorded_output_precondition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
    stored_path: str,
    stored_hash: str,
    file_exists: bool,
    expected: bool,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    output = tmp_path / "sentences.parquet"
    if file_exists:
        output.write_bytes(b"final output")
    checkpoint._metadata = {
        **checkpoint._metadata,
        "complete": complete,
        "output_path": stored_path.format(output=output),
        "output_hash": stored_hash,
    }
    monkeypatch.setattr(checkpoint_module, "sha256_file", lambda _path: stored_hash)

    assert checkpoint.output_matches(output, output_hash="hash") is expected


def test_sentence_checkpoint_rejects_missing_and_schema_invalid_batch_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )

    assert checkpoint.batch_row_count(0) is None

    invalid_path = checkpoint._batch_path(0)
    pq.write_table(pa.table({"unexpected": ["value"]}), invalid_path)

    assert checkpoint.batch_row_count(0) is None
    assert checkpoint.completed_batches == ()
