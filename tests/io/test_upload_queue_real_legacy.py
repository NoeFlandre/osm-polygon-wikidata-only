"""Phase 2.5: Real legacy upload-envelope compatibility, snapshot
TOCTOU hardening, and resumed-snapshot lifecycle.

Defects addressed:

1. The previous "legacy envelope" test created a *bg-upload-v1*
   envelope with sequence, snapshot_path and sha256. That is the
   *current* format with a UUID filename, not the actual old shape.
   Real legacy envelopes had a UUID filename, no contract_version,
   no sequence, no snapshot_path, and no sha256 -- only
   ``message`` and ``ops`` where each op carried ``action``,
   ``path_in_repo`` and ``local_path`` only.

   The queue must:

   * Recognise the real legacy shape (UUID filename, no
     contract_version).
   * Upgrade legacy envelopes durably: build immutable snapshots,
     compute the SHA-256 of each snapshot, and persist a NEW
     ``bg-upload-v1`` envelope atomically.
   * Fail closed if a canonical local_path is missing -- the
     ORIGINAL envelope must be preserved on failure.
   * Never use the canonical mutable local_path as the snapshot.

2. Snapshot TOCTOU: the production code computes ``_sha256_file``
   against ``op.local_path`` BEFORE calling ``_independent_copy``.
   The recorded hash therefore describes the source bytes, not the
   snapshot bytes. A test that mutates the source between submit
   and upload does not exercise the race; we mutate the source
   *during* the snapshot copy.

3. Resumed snapshots still leak: ``resume_pending()`` constructs
   ``_UploadJob(..., snapshot_dir=None)`` so successful resumed
   jobs never delete their snapshot directory. The queue must
   derive the snapshot directory from the envelope (it lives at
   ``state_dir/snapshots/<sequence>`` by construction), validate
   that the resolved path is inside ``state_dir/snapshots``, and
   remove it only after the upload has been confirmed successful.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _queue():
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    return BackgroundUploadQueue


def _op():
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp

    return PublicationOp


# ---------------------------------------------------------------------------
# Defect 1: real legacy upload-envelope compatibility
# ---------------------------------------------------------------------------


_LEGACY_CONTRACT_VERSION = "bg-upload-legacy-v0"


def _write_legacy_envelope(
    state_dir: Path,
    *,
    name: str,
    message: str,
    ops: list[dict],
) -> Path:
    """Write a *real* legacy envelope: UUID filename, no
    contract_version, no sequence, no snapshot_path, no sha256.
    Each op carries action/path_in_repo/local_path only."""
    state_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "message": message,
        "ops": ops,
    }
    path = state_dir / f"{name}.json"
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    return path


def test_resume_upgrades_real_legacy_envelope_durably(tmp_path: Path) -> None:
    """A real legacy envelope (UUID filename, no contract_version) must
    be upgraded in-place to a ``bg-upload-v1`` envelope with
    snapshots and SHA-256 hashes computed from the snapshot bytes.
    """
    mod = _queue()
    state_dir = tmp_path / "state"

    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"LEGACY_DATA")

    legacy_path = _write_legacy_envelope(
        state_dir,
        name="legacy-uuid-0001",
        message="legacy commit",
        ops=[
            {
                "action": "add",
                "path_in_repo": "data.parquet",
                "local_path": str(canonical),
            }
        ],
    )

    upload_calls: list[tuple[list[str], str]] = []
    snapshot_paths_seen: list[str] = []

    def upload(ops, message):
        upload_calls.append(([str(op.local_path) for op in ops], message))
        # Capture state of the state dir BEFORE the worker deletes it.
        for op in ops:
            if op.snapshot_path is not None:
                snapshot_paths_seen.append(str(op.snapshot_path))

    q = mod(upload=upload, state_dir=state_dir)
    try:
        q.resume_pending()
    finally:
        q.close_and_wait()

    assert upload_calls and upload_calls[0][1] == "legacy commit", (
        f"Upload must be invoked for the upgraded legacy envelope; got {upload_calls}"
    )

    # The upload must have been called with a snapshot_path (NOT the
    # canonical local_path).
    assert snapshot_paths_seen, "Snapshot path must be set on the op the upload sees"
    for path in snapshot_paths_seen:
        assert "/snapshots/" in path, (
            f"Upload must use the queue-owned snapshot path (in state_dir/snapshots/); got {path}"
        )
        assert path != str(tmp_path / "canonical.parquet"), (
            f"Upload must NOT use the canonical local_path as the snapshot; got {path}"
        )

    # Legacy envelope file must have been removed.
    assert not legacy_path.is_file(), (
        f"Legacy envelope must be removed after upgrade; still at {legacy_path}"
    )

    # After successful upload the envelope + snapshot dir are gone.
    envelopes_after = [p for p in state_dir.glob("*.json") if p.name != ".highwater"]
    assert envelopes_after == [], (
        f"Upgraded envelope must be removed after successful upload; still {envelopes_after}"
    )


def test_legacy_upgrade_missing_local_path_fails_closed(tmp_path: Path) -> None:
    """If a legacy envelope references a missing canonical file, the
    upgrade must fail closed: the original legacy envelope is
    preserved and no upload occurs.
    """
    mod = _queue()
    state_dir = tmp_path / "state"

    legacy_path = _write_legacy_envelope(
        state_dir,
        name="legacy-uuid-missing",
        message="legacy commit with missing file",
        ops=[
            {
                "action": "add",
                "path_in_repo": "missing.parquet",
                "local_path": str(tmp_path / "does-not-exist.parquet"),
            }
        ],
    )

    upload_calls: list[str] = []

    def upload(ops, message):
        upload_calls.append(message)

    q = mod(upload=upload, state_dir=state_dir)
    try:
        q.resume_pending()
    finally:
        q.close_and_wait()
    failures = q._failures

    # Upload must NOT be called.
    assert upload_calls == [], (
        f"Upload must not be invoked when local_path is missing; got {upload_calls}"
    )
    # The original legacy envelope must be preserved.
    assert legacy_path.is_file(), "Original legacy envelope must be preserved on upgrade failure"
    # The failure must be surfaced.
    assert any(
        "missing" in failure.lower() or "does-not-exist" in failure for failure in failures
    ), f"Failure must be surfaced; got {failures}"


def test_legacy_upgrade_snapshot_is_independent_of_canonical(tmp_path: Path) -> None:
    """The snapshot produced by legacy upgrade must be independent of
    the canonical file: a mutation to the canonical file must NOT
    affect the snapshot bytes the upload sees.
    """
    mod = _queue()
    state_dir = tmp_path / "state"

    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"LEGACY_DATA")

    _write_legacy_envelope(
        state_dir,
        name="legacy-uuid-isolated",
        message="legacy isolated",
        ops=[
            {
                "action": "add",
                "path_in_repo": "data.parquet",
                "local_path": str(canonical),
            }
        ],
    )

    captured: dict[str, bytes] = {}

    def upload(ops, message):
        # Read bytes from the snapshot_path (the immutable copy),
        # NOT the canonical local_path.
        op = ops[0]
        source = op.snapshot_path or op.local_path
        assert source is not None
        captured["bytes"] = source.read_bytes()

    q = mod(upload=upload, state_dir=state_dir)
    try:
        q.resume_pending()
        # Mutate the canonical file -- the snapshot must NOT change.
        canonical.write_bytes(b"MUTATED")
    finally:
        q.close_and_wait()
    assert captured.get("bytes") == b"LEGACY_DATA", (
        f"Legacy upgrade snapshot must be independent of canonical; got {captured.get('bytes')!r}"
    )


# ---------------------------------------------------------------------------
# Defect 2: snapshot TOCTOU during copy
# ---------------------------------------------------------------------------


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
        with open(source, "rb") as src, open(target, "wb") as dst:
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


# ---------------------------------------------------------------------------
# Defect 3: resumed snapshots leak
# ---------------------------------------------------------------------------


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
