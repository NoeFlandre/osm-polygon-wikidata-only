"""Durable background upload queue: snapshots, sequencing, resume, and cleanup.

Key invariants:

* Submit snapshots each local file into an independent copy, so later
  mutation of the canonical file never changes what gets uploaded, and
  the recorded SHA-256 describes the snapshot bytes.
* Sequence allocation is monotonic across queue restarts; resume
  ordering follows the envelope sequence, not the filename.
* Malformed, non-UTF-8 or duplicate-sequence envelopes fail before any
  upload; a snapshot hash mismatch fails loudly and keeps the envelope.
* Envelopes and snapshot directories are removed only after success.
* Delete ops carry no snapshot or hash.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest


def _queue():
    try:
        from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
    except ImportError as exc:
        pytest.fail(f"BackgroundUploadQueue import failed: {exc}")
    return BackgroundUploadQueue


def _op():
    try:
        from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
    except ImportError as exc:
        pytest.fail(f"PublicationOp import failed: {exc}")
    return PublicationOp


def _block_until(release: threading.Event):
    def upload(_ops, _message):
        release.wait(timeout=30)

    return upload


def _make_op(action: str, path_in_repo: str, local_path: Path | None = None) -> object:
    return _op()(action=action, path_in_repo=path_in_repo, local_path=local_path)


def test_local_file_copied_to_snapshot_directory_at_submit(tmp_path: Path) -> None:
    """An ``add`` op must copy its local_path into a queue-owned
    snapshots/ directory."""
    BgUploadQueue = _queue()
    canonical = tmp_path / "canonical.txt"
    canonical.write_text("ORIGINAL")
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit(
            [_make_op("add", "polygon_articles/x", canonical)],
            "msg-1",
        )
        snapshots_dir = tmp_path / "snapshots"
        snapshot_files = list(snapshots_dir.rglob("*.txt"))
        assert snapshot_files, (
            f"Snapshot copy must be created at submit time under state_dir/snapshots/; "
            f"found nothing in {snapshots_dir}"
        )
        assert snapshot_files[0].read_text() == "ORIGINAL"
    finally:
        release.set()
        queue.close_and_wait()


def test_snapshot_is_independent_copy_not_a_hard_link(tmp_path: Path) -> None:
    """The snapshot must be an independent copy, NOT a hard link sharing
    the canonical inode. A hard link would mutate when the canonical
    file is modified, which would defeat the snapshot's purpose."""
    BgUploadQueue = _queue()
    canonical = tmp_path / "canonical.txt"
    canonical.write_text("ORIGINAL")
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit(
            [_make_op("add", "polygon_articles/x", canonical)],
            "msg-1",
        )
        snapshots_dir = tmp_path / "snapshots"
        snapshot_files = list(snapshots_dir.rglob("*.txt"))
        assert snapshot_files
        snap = snapshot_files[0]
        # Confirm this is NOT a hard link: the snapshot must have a
        # different inode from the canonical file.
        canonical_inode = canonical.stat().st_ino
        snapshot_inode = snap.stat().st_ino
        if snapshot_inode == canonical_inode:
            # Same inode -> hard link. Refuse.
            pytest.fail(
                "Snapshot shares the canonical inode (hard link). "
                "The snapshot must be an independent copy or reflink."
            )
    finally:
        release.set()
        queue.close_and_wait()


def test_canonical_mutation_after_submit_does_not_affect_pending_job(
    tmp_path: Path,
) -> None:
    """Submitting a job must snapshot the file; mutating canonical
    afterward must NOT change what gets uploaded."""
    BgUploadQueue = _queue()
    canonical = tmp_path / "canonical.txt"
    canonical.write_text("ORIGINAL")
    release = threading.Event()
    seen_bytes: list[bytes] = []
    queue = BgUploadQueue(
        upload=lambda ops, _msg: (
            release.wait(timeout=30),
            # The upload reads from the op's *effective* file:
            # snapshot_path when present, else local_path. The
            # canonical local_path is preserved on the op for the
            # downstream retirement check.
            seen_bytes.append(Path(ops[0].snapshot_path or ops[0].local_path).read_bytes()),
        ),
        max_pending=2,
        state_dir=tmp_path,
    )
    try:
        queue.submit(
            [_make_op("add", "polygon_articles/x", canonical)],
            "msg-1",
        )
        canonical.write_text("MUTATED_AFTER_SUBMIT")
        release.set()
        queue.close_and_wait()
        assert seen_bytes == [b"ORIGINAL"], (
            f"Upload must use snapshot, not canonical; got {seen_bytes!r}"
        )
    except Exception:
        release.set()
        queue.close_and_wait()
        raise


def test_two_ops_with_same_basename_get_distinct_snapshots(tmp_path: Path) -> None:
    """Two ops with the same ``local_path.basename`` must not collide
    inside the snapshots directory."""
    BgUploadQueue = _queue()
    canonical_a = tmp_path / "canonical_a" / "data.txt"
    canonical_b = tmp_path / "canonical_b" / "data.txt"
    canonical_a.parent.mkdir(parents=True)
    canonical_b.parent.mkdir(parents=True)
    canonical_a.write_text("A")
    canonical_b.write_text("B")
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=3, state_dir=tmp_path)
    try:
        queue.submit(
            [
                _make_op("add", "polygon_articles/a", canonical_a),
                _make_op("add", "polygon_articles/b", canonical_b),
            ],
            "msg-1",
        )
        snapshots_dir = tmp_path / "snapshots"
        # The snapshots directory exists and contains both files.
        snapshot_files = list(snapshots_dir.rglob("data.txt"))
        assert len(snapshot_files) == 2, (
            f"Two snapshots must coexist (one per op), got {snapshot_files}"
        )
        contents = sorted(p.read_text() for p in snapshot_files)
        assert contents == ["A", "B"], f"Both snapshots must be intact, got {contents}"
    finally:
        release.set()
        queue.close_and_wait()


def test_snapshot_sha256_recorded_in_envelope(tmp_path: Path) -> None:
    BgUploadQueue = _queue()
    canonical = tmp_path / "canonical.txt"
    canonical.write_text("CONTENT")
    expected_sha = hashlib.sha256(b"CONTENT").hexdigest()
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit(
            [_make_op("add", "polygon_articles/x", canonical)],
            "msg-1",
        )
        state_files = sorted(tmp_path.glob("*.json"))
        envelope = json.loads(state_files[0].read_text())
        op = envelope["ops"][0]
        assert "sha256" in op, f"Op must record sha256 of snapshot; got keys: {list(op.keys())}"
        assert op["sha256"] == expected_sha, (
            f"Recorded sha256 must match snapshot content hash; "
            f"expected {expected_sha}, got {op.get('sha256')}"
        )
    finally:
        release.set()
        queue.close_and_wait()


def test_delete_op_persists_without_local_file_or_hash(tmp_path: Path) -> None:
    """Delete ops carry no local_path, no sha256, and no snapshot."""
    BgUploadQueue = _queue()
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit([_make_op("delete", "articles/old")], "msg-delete")
        state_files = sorted(tmp_path.glob("*.json"))
        envelope = json.loads(state_files[0].read_text())
        op = envelope["ops"][0]
        assert op.get("local_path") is None
        assert op.get("sha256") is None or "sha256" not in op
        # No snapshot directory must be created for delete-only submits.
        snapshots_dir = tmp_path / "snapshots"
        assert not snapshots_dir.exists() or not any(snapshots_dir.rglob("*")), (
            f"Delete-only submit must not create a snapshot directory, "
            f"got contents: {list(snapshots_dir.rglob('*'))}"
        )
    finally:
        release.set()
        queue.close_and_wait()


def test_sequence_allocation_is_monotonic_across_restart(tmp_path: Path) -> None:
    """A restarted queue allocates sequences strictly after the previous
    queue's highest one, even after its envelopes were removed."""
    BgUploadQueue = _queue()

    def submitted_sequence(queue_state: Path, message: str) -> int:
        release = threading.Event()
        queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=queue_state)
        try:
            queue.submit([_make_op("delete", f"articles/{message}")], message)
            (state_file,) = queue_state.glob("*.json")
            return json.loads(state_file.read_text())["sequence"]
        finally:
            release.set()
            assert queue.close_and_wait() == []

    first = submitted_sequence(tmp_path, "first")
    assert list(tmp_path.glob("*.json")) == []
    second = submitted_sequence(tmp_path, "second")

    assert second > first


def test_resume_pending_rejects_duplicate_sequence_ids(tmp_path: Path) -> None:
    """Two envelopes with the SAME sequence number must be rejected at
    resume time, BEFORE any upload is attempted."""
    BgUploadQueue = _queue()
    state_dir = tmp_path / "queue"
    state_dir.mkdir()
    for name in ("000001.json", "000001-dup.json"):
        (state_dir / name).write_text(
            json.dumps(
                {
                    "contract_version": "bg-upload-v1",
                    "sequence": 1,
                    "message": "dup",
                    "ops": [
                        {
                            "action": "delete",
                            "path_in_repo": "articles/x",
                            "local_path": None,
                            "sha256": None,
                        }
                    ],
                }
            )
        )
    started = threading.Event()
    upload_called = threading.Event()

    def upload(_ops, _msg):
        upload_called.set()
        started.set()

    queue = BgUploadQueue(upload=upload, max_pending=2, state_dir=state_dir)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            queue.resume_pending()
        failed = queue.close_and_wait()
        assert not upload_called.is_set(), (
            f"Duplicate-sequence envelopes must NOT trigger upload; failed={failed}"
        )
    finally:
        queue.close_and_wait()


def test_resume_pending_rejects_malformed_envelope_before_upload(tmp_path: Path) -> None:
    """A malformed envelope must fail validation BEFORE the upload is attempted."""
    BgUploadQueue = _queue()
    state_dir = tmp_path / "queue"
    state_dir.mkdir()
    (state_dir / "000001.json").write_text("not json {{{")
    upload_called = threading.Event()

    def upload(_ops, _msg):
        upload_called.set()

    queue = BgUploadQueue(upload=upload, max_pending=2, state_dir=state_dir)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            queue.resume_pending()
        assert not upload_called.is_set(), (
            "Malformed envelope must be rejected before upload is attempted"
        )
    finally:
        queue.close_and_wait()


def test_resume_pending_rejects_non_utf8_envelope_before_upload(tmp_path: Path) -> None:
    """A corrupt UTF-8 state file must fail closed, not crash queue setup."""
    BgUploadQueue = _queue()
    state_dir = tmp_path / "queue"
    state_dir.mkdir()
    (state_dir / "000001.json").write_bytes(b"\xff\xfe")
    upload_called = threading.Event()

    def upload(_ops, _msg):
        upload_called.set()

    queue = BgUploadQueue(upload=upload, max_pending=2, state_dir=state_dir)
    try:
        with pytest.raises(ValueError, match="Malformed envelope"):
            queue.resume_pending()
        assert not upload_called.is_set()
    finally:
        queue.close_and_wait()


def test_resume_sorts_by_sequence_not_filename(tmp_path: Path) -> None:
    """Hand-crafted envelopes with sequence=2 named 'a' and sequence=1
    named 'b' must still process in sequence order [1, 2]."""
    BgUploadQueue = _queue()
    state_dir = tmp_path / "queue"
    state_dir.mkdir()
    (state_dir / "b.json").write_text(
        json.dumps(
            {
                "contract_version": "bg-upload-v1",
                "sequence": 1,
                "message": "msg-1",
                "ops": [
                    {
                        "action": "delete",
                        "path_in_repo": "articles/1",
                        "local_path": None,
                        "sha256": None,
                    }
                ],
            }
        )
    )
    (state_dir / "a.json").write_text(
        json.dumps(
            {
                "contract_version": "bg-upload-v1",
                "sequence": 2,
                "message": "msg-2",
                "ops": [
                    {
                        "action": "delete",
                        "path_in_repo": "articles/2",
                        "local_path": None,
                        "sha256": None,
                    }
                ],
            }
        )
    )

    order: list[str] = []
    done = threading.Event()

    def upload(ops, _msg):
        order.append(ops[0].path_in_repo.rsplit("/", 1)[-1])
        if len(order) == 2:
            done.set()

    queue = BgUploadQueue(upload=upload, max_pending=4, state_dir=state_dir)
    queue.resume_pending()
    assert done.wait(timeout=5), f"Expected both jobs to upload; got {order}"
    queue.close_and_wait()
    assert order == ["1", "2"], f"Expected sequence-ordered resume [1, 2]; got {order}"


def test_snapshot_sha256_mismatch_on_resume_fails_loudly(tmp_path: Path) -> None:
    """A snapshot whose sha256 doesn't match envelope must NOT be uploaded.

    The failure must be exposed by ``close_and_wait``; the job envelope
    and snapshot must remain on disk for retry.
    """
    BgUploadQueue = _queue()
    state_dir = tmp_path / "queue"
    snapshots = state_dir / "snapshots" / "000001"
    snapshots.mkdir(parents=True)
    snap = snapshots / "data.txt"
    snap.write_text("TAMPERED")
    wrong_sha = hashlib.sha256(b"ORIGINAL").hexdigest()
    (state_dir / "000001.json").write_text(
        json.dumps(
            {
                "contract_version": "bg-upload-v1",
                "sequence": 1,
                "message": "msg",
                "ops": [
                    {
                        "action": "add",
                        "path_in_repo": "x",
                        "local_path": str(snap),
                        "sha256": wrong_sha,
                    }
                ],
            }
        )
    )

    called = threading.Event()
    uploaded = []

    def upload(ops, _msg):
        uploaded.append(ops)
        called.set()

    queue = BgUploadQueue(upload=upload, max_pending=2, state_dir=state_dir)
    try:
        queue.resume_pending()
        assert not called.wait(timeout=2), (
            "Tampered snapshot must NOT be uploaded; the upload callback was called"
        )
        failures = queue.close_and_wait()
        assert failures, (
            f"SHA mismatch must be exposed as a failure by close_and_wait; got {failures!r}"
        )
        # Envelope and snapshot must remain for retry.
        assert (state_dir / "000001.json").is_file(), (
            "Envelope must remain on disk for retry after a SHA mismatch"
        )
        assert snap.is_file(), "Snapshot must remain on disk for retry after a SHA mismatch"
    finally:
        queue.close_and_wait()


def test_successful_upload_removes_envelope_and_snapshot_directory(
    tmp_path: Path,
) -> None:
    BgUploadQueue = _queue()
    canonical = tmp_path / "canonical.txt"
    canonical.write_text("OK")
    queue = BgUploadQueue(upload=lambda *_: None, max_pending=2, state_dir=tmp_path)
    queue.submit([_make_op("add", "polygon_articles/x", canonical)], "msg-ok")
    failures = queue.close_and_wait()
    assert failures == [], f"Expected no failures, got {failures}"
    # Envelope must be removed.
    state_files = list(tmp_path.glob("*.json"))
    assert state_files == [], f"Envelope must be removed after success, got {state_files}"
    # Snapshot directory must be removed.
    snapshots_dir = tmp_path / "snapshots"
    assert not snapshots_dir.exists() or list(snapshots_dir.rglob("*")) == [], (
        f"Snapshot directory must be removed after success, got {list(snapshots_dir.rglob('*'))}"
    )


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


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


def test_snapshot_hash_is_computed_after_copy_during_race(tmp_path: Path) -> None:
    """Mutate the canonical file DURING the snapshot copy. The
    recorded hash in the envelope must describe the snapshot bytes
    (post-copy), not the source bytes (pre-copy).
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"INITIAL")

    # Patch ``_independent_copy`` to mutate the source during the
    # copy, simulating a race where another process writes to the
    # canonical file while we are snapshotting.
    from osm_polygon_wikidata_only.hf import upload_queue as uq_mod

    real_copy = uq_mod._independent_copy

    snapshot_seen: dict[str, bytes] = {}

    def _racy_copy(source, target):
        # Start the copy, mutate the source mid-stream.
        target.parent.mkdir(parents=True, exist_ok=True)
        # Read-then-write with a race window.
        with source.open("rb") as src, target.open("wb") as dst:
            chunk = src.read(4)  # Read just part of the file
            source.write_bytes(b"RACED")  # Mutate source mid-copy
            dst.write(chunk)
            dst.write(src.read())  # Rest of source after mutation
        snapshot_seen["bytes"] = target.read_bytes()

    uq_mod._independent_copy = _racy_copy
    captured_envelope: dict[str, Any] = {}

    def upload(ops, message):
        # The envelope file on disk is what the worker sees; copy it
        # before the worker deletes it.
        envelope_files = [p for p in state_dir.glob("*.json") if p.name != ".highwater"]
        if envelope_files:
            captured_envelope["payload"] = json.loads(envelope_files[0].read_text())

    try:
        q = mod(upload=upload, state_dir=state_dir)
        q.submit(
            [_op()(action="add", path_in_repo="data.parquet", local_path=canonical)],
            "race",
        )
        q.close_and_wait()
    finally:
        uq_mod._independent_copy = real_copy

    assert "payload" in captured_envelope, (
        "Upload callback must be invoked; the test failed to capture the envelope"
    )
    envelope = captured_envelope["payload"]
    recorded_sha = envelope["ops"][0]["sha256"]
    # The recorded sha must equal the hash of the SNAPSHOT bytes, not
    # the canonical source bytes.
    snapshot_bytes = snapshot_seen["bytes"]
    snapshot_sha = hashlib.sha256(snapshot_bytes).hexdigest()
    assert recorded_sha == snapshot_sha, (
        f"Recorded sha must describe snapshot bytes; recorded={recorded_sha}, snapshot_sha={snapshot_sha}"
    )
    # And the recorded sha must NOT equal the (mutated) canonical bytes.
    canonical_sha = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert recorded_sha != canonical_sha, (
        f"Recorded sha must NOT describe canonical bytes; recorded={recorded_sha}, canonical={canonical_sha}"
    )


def test_resumed_snapshot_directory_is_removed_after_success(tmp_path: Path) -> None:
    """A successful resume must remove BOTH the envelope AND the
    queue-owned snapshot directory derived from the envelope's
    sequence.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"DATA")

    # Pre-seed a fully-formed bg-upload-v1 envelope that resume()
    # picks up -- the snapshot directory will be derived from the
    # envelope's sequence.
    sequence = 42
    envelope_path = state_dir / f"{sequence:06d}.json"
    snapshot_dir = state_dir / "snapshots" / f"{sequence:06d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshot_dir / "000" / "canonical.parquet"
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.write_bytes(b"DATA")

    envelope = {
        "contract_version": "bg-upload-v1",
        "sequence": sequence,
        "message": "resume success",
        "ops": [
            {
                "action": "add",
                "path_in_repo": "data.parquet",
                "local_path": str(canonical),
                "snapshot_path": str(snapshot_file),
                "sha256": hashlib.sha256(b"DATA").hexdigest(),
            }
        ],
    }
    envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")

    q = mod(upload=lambda ops, msg: None, state_dir=state_dir)
    try:
        q.resume_pending()
    finally:
        q.close_and_wait()

    assert not envelope_path.is_file(), "Successful resume must remove the envelope"
    assert not snapshot_dir.is_dir(), (
        f"Successful resume must remove the snapshot directory; still at {snapshot_dir}"
    )


def test_resume_rejects_snapshot_path_outside_state_dir(tmp_path: Path) -> None:
    """A resume envelope that records a snapshot_path outside
    ``state_dir/snapshots`` must be rejected -- never trust an
    envelope's path to delete an arbitrary location.
    """
    mod = _queue()
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Try to get the resume to delete a path outside state_dir.
    evil = tmp_path / "evil.parquet"
    evil.write_bytes(b"EVIL")

    envelope = {
        "contract_version": "bg-upload-v1",
        "sequence": 1,
        "message": "evil",
        "ops": [
            {
                "action": "add",
                "path_in_repo": "data.parquet",
                "local_path": str(evil),
                "snapshot_path": str(evil),
                "sha256": hashlib.sha256(b"EVIL").hexdigest(),
            }
        ],
    }
    envelope_path = state_dir / "000001.json"
    envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")

    q = mod(upload=lambda ops, msg: None, state_dir=state_dir)
    try:
        q.resume_pending()
    finally:
        q.close_and_wait()
    failures = q._failures

    # The evil file must NOT be deleted.
    assert evil.is_file(), "Resume must not delete files outside state_dir/snapshots"
    assert any(
        "snapshot" in failure.lower() or "outside" in failure.lower() for failure in failures
    ), f"Outside-state-dir snapshot path must be reported; got {failures}"


def test_tampered_snapshot_is_not_uploaded_and_its_envelope_is_kept(tmp_path: Path) -> None:
    """A queue-owned snapshot whose bytes no longer match the recorded
    SHA-256 must fail the job before upload and keep the envelope for retry."""
    BgUploadQueue = _queue()
    state_dir = tmp_path / "state"
    canonical = _write(tmp_path / "canonical" / "data.parquet", b"DATA")
    snapshot_file = _write(state_dir / "snapshots" / "000001" / "000" / "data.parquet", b"MUTATED")
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
                        "path_in_repo": "data.parquet",
                        "local_path": str(canonical),
                        "snapshot_path": str(snapshot_file),
                        "sha256": hashlib.sha256(b"DATA").hexdigest(),
                    }
                ],
            }
        )
    )
    uploads: list[str] = []

    queue = BgUploadQueue(upload=lambda _ops, message: uploads.append(message), state_dir=state_dir)
    try:
        queue.resume_pending()
    finally:
        failures = queue.close_and_wait()

    assert uploads == []
    assert any("tampered" in failure and "SHA mismatch" in failure for failure in failures)
    assert state_file.is_file()
    assert snapshot_file.is_file()
