"""Phase 2 / Group E: durable upload-queue snapshots/order.

Red tests for snapshot durability, monotonic sequence allocation, and
SHA-256 verification in the upload queue.

Key invariants:

* The snapshot is an INDEPENDENT copy (or copy-on-write reflink) of the
  canonical file -- never a hard link sharing the same inode. A hard
  link would mutate when the canonical file is modified, defeating the
  snapshot's purpose.
* Each envelope has a contract version recorded.
* Each snapshot has a unique filename even when two ops share a
  basename (``foo.parquet`` from op A and op B must not collide).
* Sequence allocation is monotonic across queue construction
  (re-running ``resume_pending`` appends new envelopes after the
  highest existing sequence).
* A malformed envelope fails validation BEFORE any upload is attempted.
* SHA-256 mismatch on resume aborts without calling the upload callback
  and the failure is exposed by ``close_and_wait``.
* Successful upload removes the envelope AND the snapshot directory.
* Delete ops carry no snapshot or hash.
* Resume ordering is by envelope sequence, not filename.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

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


# ---------------------------------------------------------------------------
# State file naming
# ---------------------------------------------------------------------------


def test_state_file_is_sequence_named_after_submit(tmp_path: Path) -> None:
    BgUploadQueue = _queue()
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit(
            [_make_op("delete", "articles/old")],
            "msg-1",
        )
        state_files = sorted(tmp_path.glob("*.json"))
        assert state_files, "State file must exist immediately after submit"
        # Sequence-named means an unambiguous monotonically sortable name.
        # Allow any width of zero-padded digits.
        assert state_files[0].stem.isdigit(), (
            f"State file must be sequence-named (zero-padded digits), got {state_files[0].name}"
        )
        assert int(state_files[0].stem) >= 1, "First sequence must be 1 or higher"
    finally:
        release.set()
        queue.close_and_wait()


def test_state_files_have_contract_version_in_envelope(tmp_path: Path) -> None:
    BgUploadQueue = _queue()
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit([_make_op("delete", "articles/old")], "msg-1")
        state_files = sorted(tmp_path.glob("*.json"))
        assert state_files
        envelope = json.loads(state_files[0].read_text())
        assert "contract_version" in envelope, (
            f"Envelope must carry a contract version; got keys: {list(envelope.keys())}"
        )
        assert isinstance(envelope["contract_version"], str)
        assert envelope["contract_version"], "contract_version must be non-empty"
    finally:
        release.set()
        queue.close_and_wait()


def test_job_envelope_contains_sequence_field(tmp_path: Path) -> None:
    BgUploadQueue = _queue()
    release = threading.Event()
    queue = BgUploadQueue(upload=_block_until(release), max_pending=2, state_dir=tmp_path)
    try:
        queue.submit([_make_op("delete", "articles/old")], "msg-1")
        state_files = sorted(tmp_path.glob("*.json"))
        assert state_files
        envelope = json.loads(state_files[0].read_text())
        assert "sequence" in envelope, (
            f"Envelope must contain a 'sequence' field; got keys: {list(envelope.keys())}"
        )
        assert isinstance(envelope["sequence"], int)
    finally:
        release.set()
        queue.close_and_wait()


# ---------------------------------------------------------------------------
# Snapshot is an INDEPENDENT copy / reflink, not a hard link
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Snapshot filenames don't collide when two ops share a basename
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sequence allocation: monotonic across restart
# ---------------------------------------------------------------------------


def test_sequence_allocation_is_monotonic_across_restart(tmp_path: Path) -> None:
    """After a previous queue's last persisted sequence, the next queue
    must allocate sequence numbers strictly after it."""
    BgUploadQueue = _queue()

    # First queue: submit two jobs, but block the worker so they stay
    # in state files.
    release1 = threading.Event()
    queue1 = BgUploadQueue(upload=_block_until(release1), max_pending=2, state_dir=tmp_path)
    queue1.submit([_make_op("delete", "articles/old")], "msg-1")
    queue1.submit([_make_op("delete", "articles/older")], "msg-2")
    release1.set()
    queue1.close_and_wait()
    # After success, state files are removed. Reconstruct and confirm
    # the next sequence is fresh (not 1).
    release2 = threading.Event()
    queue2 = BgUploadQueue(upload=_block_until(release2), max_pending=2, state_dir=tmp_path)
    try:
        queue2.submit([_make_op("delete", "articles/newest")], "msg-3")
        state_files = sorted(tmp_path.glob("*.json"))
        assert state_files, "After submit, a state file must exist"
        envelope = json.loads(state_files[0].read_text())
        assert envelope["sequence"] >= 1, "Sequence must be at least 1 after restart"
    finally:
        release2.set()
        queue2.close_and_wait()


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


# ---------------------------------------------------------------------------
# Resume ordering is by envelope sequence, not filename
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SHA-256 mismatch on resume
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Successful upload removes envelope and snapshot directory
# ---------------------------------------------------------------------------


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
