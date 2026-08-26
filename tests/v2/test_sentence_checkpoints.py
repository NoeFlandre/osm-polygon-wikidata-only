from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.sentence_checkpoints import SentenceCheckpoint


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
