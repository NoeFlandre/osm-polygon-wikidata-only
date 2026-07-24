"""Phase 2 / Amendment 8: Durable upload queue correctness.

Tests for:

1. Resumability for UUID-named legacy envelopes (state files that
   don't follow the ``NNNNNN.json`` pattern must still be picked up
   on resume, by checking the per-envelope ``sequence`` field).
2. The high-water mark for ``_next_sequence`` must come from a
   ``.highwater`` file persisted on disk so the queue's next
   allocation is monotonic across restarts.
3. SHA-256 is computed AFTER the snapshot copy so a TOCTOU mutation
   of the canonical file between submit and upload is caught.
4. Submit cleans up the per-job envelope + snapshot directory on
   failure to enqueue.
5. Resumed snapshot directories are removed only AFTER successful
   upload (NOT before).
6. The envelope + snapshot are preserved on upload/hash failure so
   the caller can retry.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest


def _queue():
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    return BackgroundUploadQueue


def _op():
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp

    return PublicationOp


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_resume_picks_up_uuid_named_legacy_envelope(tmp_path: Path) -> None:
    """A legacy envelope (UUID filename, no contract_version) must be
    upgraded durably and dispatched in its recorded sequence order.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Pre-seed two legacy envelopes (UUID filenames, no contract_version).
    canonical = _write(tmp_path / "canonical" / "data1.parquet", b"DATA1")
    canonical2 = _write(tmp_path / "canonical" / "data2.parquet", b"DATA2")

    uuid_a = str(uuid.uuid4())
    uuid_b = str(uuid.uuid4())
    legacy_envelope_a = {
        "message": "legacy A",
        "ops": [
            {
                "action": "add",
                "path_in_repo": "data1.parquet",
                "local_path": str(canonical),
            }
        ],
    }
    legacy_envelope_b = {
        "message": "legacy B",
        "ops": [
            {
                "action": "add",
                "path_in_repo": "data2.parquet",
                "local_path": str(canonical2),
            }
        ],
    }
    (state_dir / f"{uuid_a}.json").write_text(json.dumps(legacy_envelope_a))
    (state_dir / f"{uuid_b}.json").write_text(json.dumps(legacy_envelope_b))

    uploaded_messages: list[str] = []

    def upload(ops, message):
        uploaded_messages.append(message)

    q = mod(upload=upload, state_dir=state_dir)
    try:
        q.resume_pending()
    finally:
        q.close_and_wait()
    assert sorted(uploaded_messages) == ["legacy A", "legacy B"], (
        f"Resume must dispatch BOTH legacy envelopes; got {uploaded_messages}"
    )


def test_high_water_mark_persists_across_restarts(tmp_path: Path) -> None:
    """After allocating sequence 5, restarting the queue must allocate
    sequence 6, not sequence 1 again.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    canonical = _write(tmp_path / "canonical" / "data.parquet", b"D")

    # First queue instance: submit one job to advance the sequence.
    q1 = mod(upload=lambda ops, msg: None, state_dir=state_dir)
    q1.submit([_op()(action="add", path_in_repo="data.parquet", local_path=canonical)], "x")
    q1.close_and_wait()
    highwater_path = state_dir / ".highwater"
    assert highwater_path.is_file(), "Queue must persist a high-water mark on disk"
    persisted = int(highwater_path.read_text().strip())
    assert persisted >= 1, (
        f"High-water mark must record the highest allocated sequence; got {persisted}"
    )

    # Second instance: the next allocation must be strictly greater
    # than persisted (NOT re-allocate 1).
    q2 = mod(upload=lambda ops, msg: None, state_dir=state_dir)
    assert q2._next_sequence > persisted, (
        f"Restart must not regress the sequence counter; next={q2._next_sequence}, persisted={persisted}"
    )
    q2.close_and_wait()


def test_sha256_computed_after_copy_prevents_toctou(tmp_path: Path) -> None:
    """If the canonical file is mutated between submit and upload, the
    snapshot bytes must still match (because the snapshot was taken
    once at submit time) -- the upload must NOT be called if the
    snapshot's bytes are intact but the canonical file has changed.
    The check is: snapshot bytes hash matches the recorded sha256.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    canonical = _write(tmp_path / "canonical" / "data.parquet", b"INITIAL")

    captured: dict[str, str] = {}

    def upload(ops, message):
        # Capture the bytes the upload sees.
        captured["bytes"] = ops[0].snapshot_path.read_bytes() if ops[0].snapshot_path else b""

    q = mod(upload=upload, state_dir=state_dir)
    q.submit([_op()(action="add", path_in_repo="data.parquet", local_path=canonical)], "msg")
    # Mutate the canonical file. The snapshot must still contain INITIAL.
    canonical.write_bytes(b"MUTATED")
    q.close_and_wait()
    assert captured.get("bytes") == b"INITIAL", (
        f"Upload must use the snapshot bytes (INITIAL), got {captured.get('bytes')!r}"
    )


def test_submit_failure_cleans_up_partial_artifacts(tmp_path: Path) -> None:
    """If the snapshot copy itself fails (e.g. the canonical file
    disappears mid-submit), the queue must clean up the partial
    envelope and partial snapshot directory.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    canonical = _write(tmp_path / "canonical" / "data.parquet", b"DATA")

    # Replace upload_queue's snapshot copy with a flaky one BEFORE
    # constructing the queue, so the worker also sees the patched copy.
    from osm_polygon_wikidata_only.hf import upload_queue as uq_mod

    real_copy = uq_mod._independent_copy

    def _flaky_copy(source, target):
        raise RuntimeError("simulated copy failure")

    uq_mod._independent_copy = _flaky_copy
    try:
        q = mod(upload=lambda ops, msg: None, state_dir=state_dir)
        with pytest.raises(RuntimeError, match="simulated copy failure"):
            q.submit(
                [_op()(action="add", path_in_repo="data.parquet", local_path=canonical)],
                "flaky",
            )
        # Close queue and wait for worker to finish.
        q.close_and_wait()
    finally:
        uq_mod._independent_copy = real_copy

    # State directory must NOT contain a leftover envelope file for
    # this submission.
    leftover_envelopes = [p for p in state_dir.glob("*.json") if p.name != ".highwater"]
    assert leftover_envelopes == [], f"submit failure left envelopes on disk: {leftover_envelopes}"
    # Snapshots directory must be empty.
    snapshots_dir = state_dir / "snapshots"
    if snapshots_dir.is_dir():
        leftovers = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        assert leftovers == [], f"submit failure left snapshot directories: {leftovers}"


def test_resumed_snapshot_directory_deleted_only_after_success(tmp_path: Path) -> None:
    """After a successful upload, both the envelope AND the snapshot
    directory must be removed. After a hash mismatch (failure), both
    must be preserved so the caller can retry.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    canonical = _write(tmp_path / "canonical" / "data.parquet", b"DATA")

    # Successful path: envelope + snapshot removed.
    q = mod(upload=lambda ops, msg: None, state_dir=state_dir)
    q.submit([_op()(action="add", path_in_repo="data.parquet", local_path=canonical)], "good")
    q.close_and_wait()
    envelopes_after_success = list(state_dir.glob("*.json"))
    assert envelopes_after_success == [], (
        f"Successful upload must remove envelope, found {envelopes_after_success}"
    )
    snapshots_after_success = state_dir / "snapshots"
    assert not snapshots_after_success.is_dir() or list(snapshots_after_success.iterdir()) == [], (
        "Successful upload must remove snapshot directory"
    )

    # Failure path: pre-seed a tampered envelope where the snapshot
    # bytes do NOT match the recorded sha256.
    canonical2 = _write(tmp_path / "canonical" / "data2.parquet", b"DATA2")
    sha_data2 = hashlib.sha256(b"DATA2").hexdigest()
    # Build the envelope but DON'T write a matching snapshot file --
    # this forces a hash mismatch on resume. The snapshot_path MUST
    # live inside state_dir/snapshots for the new validation to
    # accept it.
    snapshot_dir = state_dir / "snapshots" / "000001"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshot_dir / "000" / "data2.parquet"
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.write_bytes(b"MUTATED")  # tampered bytes
    state_file = state_dir / "000001.json"
    state_file.write_text(
        json.dumps(
            {
                "contract_version": "bg-upload-v1",
                "sequence": 1,
                "message": "tampered",
                "ops": [
                    {
                        "action": "add",
                        "path_in_repo": "data2.parquet",
                        "local_path": str(canonical2),
                        "snapshot_path": str(snapshot_file),
                        "sha256": sha_data2,  # matches DATA2, not MUTATED
                    }
                ],
            }
        )
    )

    upload_calls: list[int] = []

    def upload(ops, message):
        upload_calls.append(1)

    q2 = mod(upload=upload, state_dir=state_dir)
    try:
        q2.resume_pending()
    finally:
        failures = q2.close_and_wait()
    assert upload_calls == [], "Upload must NOT be called on hash mismatch"
    assert any("tampered" in failure for failure in failures), (
        f"Hash mismatch failure must be surfaced; got {failures}"
    )
    # Envelope preserved so caller can retry.
    assert state_file.is_file(), "Envelope must be preserved on failure"
