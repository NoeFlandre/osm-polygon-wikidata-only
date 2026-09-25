"""Resume of legacy upload envelopes (UUID filename, no contract version).

Legacy envelopes carry only ``message`` and ``ops`` with
``action``/``path_in_repo``/``local_path``. Resume must upgrade them
durably to snapshotted ``bg-upload-v1`` envelopes, never upload the
mutable canonical file, and fail closed (keeping the original envelope)
when a referenced file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path


def _queue():
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    return BackgroundUploadQueue


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
