"""Bounded background publication queue with durable snapshots.

The queue persists every submitted job to a sequence-named state file
and copies the canonical local file into a queue-owned immutable
snapshot directory. Independent copies (never hard links) ensure the
upload is durable across any canonical mutation.

Each job envelope carries:

* ``contract_version`` -- the queue's contract version.
* ``sequence`` -- the monotonically increasing sequence number.
* ``message`` -- the user-supplied commit message.
* ``ops`` -- a list of operations; each op records the local path,
  the snapshot path, and the SHA-256 of the snapshot bytes.

Sequence allocation is monotonic across queue reconstruction: the next
sequence is the maximum of all existing sequence-named state files
plus one. Resuming (``resume_pending``) sorts by envelope sequence
(not filename), rejects duplicate sequence IDs, and verifies each
op's SHA-256 against the snapshot bytes before invoking the upload
callback.

Legacy envelopes (UUID filenames, no ``contract_version``,
``sequence``, ``snapshot_path`` or ``sha256``) are recognised during
resume, upgraded DURABLY to the current shape, and atomically
swapped onto the state dir. Missing canonical files fail closed --
the original legacy envelope is preserved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

LOGGER = logging.getLogger(__name__)

UploadOps = list[PublicationOp]
UploadOperation = Callable[[UploadOps, str], None]

QUEUE_CONTRACT_VERSION = "bg-upload-v1"

_SEQUENCE_FILE = re.compile(r"^(\d+)\.json$")
_HIGHWATER_FILENAME = ".highwater"
_SNAPSHOTS_SUBDIR = "snapshots"


def _read_highwater(state_dir: Path) -> int:
    """Read the persisted high-water mark from ``.highwater``.

    Returns 0 if the file is missing or malformed. The high-water mark
    is the highest sequence number that has been allocated (not yet
    necessarily committed) by any queue instance against this
    ``state_dir``.
    """
    path = state_dir / _HIGHWATER_FILENAME
    if not path.is_file():
        return 0
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0
    if value < 0:
        return 0
    return value


def _write_highwater(state_dir: Path, value: int) -> None:
    """Persist the high-water mark atomically."""
    path = state_dir / _HIGHWATER_FILENAME
    atomic_write_text(path, str(int(value)) + "\n")


def _next_sequence_from_state_dir(state_dir: Path) -> int:
    """Return the next monotonically increasing sequence number.

    Prefers the persisted ``.highwater`` file (explicitly written
    after every allocation). Falls back to scanning state files for
    legacy envelopes (UUID-named or numbered) to compute the
    high-water mark if no ``.highwater`` file is present.
    """
    if not state_dir.is_dir():
        return 1
    persisted = _read_highwater(state_dir)
    scanned = 0
    for path in state_dir.glob("*.json"):
        match = _SEQUENCE_FILE.match(path.name)
        if match is None:
            # Legacy UUID-named envelopes -- look up the envelope's
            # own ``sequence`` field if any (legacy envelopes have
            # none -- the high-water mark is then derived from the
            # scan of numbered envelopes plus this UUID envelope
            # counts toward the water mark through the explicit
            # ``_allocate_sequence`` call).
            envelope = _read_legacy_or_current_envelope(path)
            if envelope is None:
                continue
            sequence = envelope.get("sequence")
            if isinstance(sequence, int) and sequence > scanned:
                scanned = sequence
            continue
        value = int(match.group(1))
        if value > scanned:
            scanned = value
    highest = max(persisted, scanned)
    return highest + 1


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot_target_name(op_index: int, sequence: int, original_name: str) -> str:
    """Return a per-op, per-sequence snapshot filename that cannot collide
    when two ops share a basename (e.g. ``data.txt``)."""
    return f"{sequence:06d}_op{op_index:03d}_{original_name}"


def _independent_copy(source: Path, target: Path) -> None:
    """Copy *source* to *target* as an INDEPENDENT file (never a hard link).

    A hard link would share the canonical inode and mutate when the
    canonical file is modified, defeating the snapshot's purpose.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from *path*, returning ``None`` on any
    malformed shape."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _read_envelope(path: Path) -> dict[str, Any] | None:
    """Return the envelope if *path* is a current ``bg-upload-v1``
    envelope, else ``None``."""
    raw = _read_json_object(path)
    if raw is None:
        return None
    if raw.get("contract_version") != QUEUE_CONTRACT_VERSION:
        return None
    return raw


def _read_legacy_or_current_envelope(path: Path) -> dict[str, Any] | None:
    """Return the envelope dict for either a legacy or current shape.

    A *legacy* envelope has a UUID filename and no ``contract_version``
    field. Its shape is::

        {"message": str, "ops": [{"action", "path_in_repo", "local_path"}, ...]}

    Anything that fails to parse as a JSON object returns ``None``.
    """
    return _read_json_object(path)


def _is_legacy_envelope(payload: dict[str, Any]) -> bool:
    """Return True when the envelope is missing ``contract_version``
    (the legacy shape) and otherwise looks well-formed (has
    ``message`` and ``ops``)."""
    if "contract_version" in payload:
        return False
    return isinstance(payload.get("message"), str) and isinstance(payload.get("ops"), list)


def _snapshot_dir_for_sequence(state_dir: Path, sequence: int) -> Path:
    """Return the queue-owned snapshot directory for *sequence*."""
    return state_dir / _SNAPSHOTS_SUBDIR / f"{sequence:06d}"


def _is_inside(child: Path, parent: Path) -> bool:
    """Return True iff ``child`` is the same as or strictly inside
    ``parent`` (resolved). Symlinks are followed."""
    try:
        child_resolved = child.resolve(strict=False)
        parent_resolved = parent.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    if child_resolved == parent_resolved:
        return True
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    return True


def _delete_envelope_with_snapshot(state_path: Path, snapshot_dir: Path | None) -> None:
    """Atomically remove the envelope and its queue-owned snapshot
    directory. The snapshot directory must be inside
    ``state_dir/snapshots`` or the call is rejected."""
    state_path.unlink(missing_ok=True)
    if snapshot_dir is None:
        return
    state_dir = state_path.parent
    snapshots_root = state_dir / _SNAPSHOTS_SUBDIR
    if not _is_inside(snapshot_dir, snapshots_root):
        raise ValueError(
            f"snapshot_dir {snapshot_dir} is outside {snapshots_root}; refusing to delete"
        )
    shutil.rmtree(snapshot_dir, ignore_errors=True)


@dataclass(frozen=True)
class _UploadJob:
    ops: UploadOps
    message: str
    state_path: Path | None
    snapshot_dir: Path | None
    op_shas: tuple[str | None, ...] = ()


class BackgroundUploadQueue:
    """Durable background publication queue."""

    def __init__(
        self,
        *,
        upload: UploadOperation,
        max_pending: int = 2,
        state_dir: Path | None = None,
        attempts: int = 3,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._upload = upload
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self._attempts = attempts
        self._jobs: queue.Queue[_UploadJob | None] = queue.Queue(max_pending)
        self._state_dir = state_dir
        if state_dir is not None:
            state_dir.mkdir(parents=True, exist_ok=True)
        self._next_sequence_lock = threading.Lock()
        self._next_sequence = (
            _next_sequence_from_state_dir(state_dir) if state_dir is not None else 1
        )
        self._failures: list[str] = []
        self._thread = threading.Thread(target=self._worker, name="hf-upload", daemon=False)
        self._closed = False
        self._thread.start()

    def _allocate_sequence(self) -> int:
        with self._next_sequence_lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            if self._state_dir is not None:
                # Persist the high-water mark so a fresh queue
                # instance against the same state_dir never
                # re-allocates a sequence.
                _write_highwater(self._state_dir, sequence)
            return sequence

    def submit(self, ops: UploadOps, message: str) -> None:
        if self._closed:
            raise RuntimeError("upload queue is closed")
        state_path: Path | None = None
        snapshot_dir: Path | None = None
        rewritten_ops: UploadOps = list(ops)
        try:
            if self._state_dir is not None:
                sequence = self._allocate_sequence()
                state_path = self._state_dir / f"{sequence:06d}.json"
                op_entries: list[dict[str, Any]] = []
                new_ops: list[PublicationOp] = []
                has_add = any(op.action == "add" for op in ops)
                if has_add:
                    snapshot_dir = _snapshot_dir_for_sequence(self._state_dir, sequence)
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                for op_index, op in enumerate(ops):
                    entry: dict[str, Any] = {
                        "action": op.action,
                        "path_in_repo": op.path_in_repo,
                        "local_path": str(op.local_path) if op.local_path else None,
                    }
                    if op.action == "add" and op.local_path is not None:
                        if not op.local_path.is_file():
                            raise FileNotFoundError(
                                f"Cannot snapshot missing file: {op.local_path}"
                            )
                        if snapshot_dir is None:
                            raise RuntimeError("snapshot_dir must be set when any op is an add")
                        # Per-op subdirectory: avoids name collisions
                        # when two ops share a basename (e.g. ``data.txt``).
                        op_dir = snapshot_dir / f"{op_index:03d}"
                        op_dir.mkdir(parents=True, exist_ok=True)
                        snapshot_path = op_dir / op.local_path.name
                        _independent_copy(op.local_path, snapshot_path)
                        # Hash the COMPLETED SNAPSHOT bytes, not the
                        # pre-copy source bytes, so the recorded hash
                        # describes the immutable snapshot.
                        sha = _sha256_file(snapshot_path)
                        entry["snapshot_path"] = str(snapshot_path)
                        entry["sha256"] = sha
                        # The op keeps the CANONICAL local_path (so the
                        # paired-retirement check can verify the canonical
                        # location) and carries the snapshot_path on the
                        # side so the upload callback reads from the
                        # immutable snapshot rather than the canonical
                        # file (which may mutate between submit and
                        # upload).
                        new_ops.append(
                            PublicationOp(
                                action=op.action,
                                path_in_repo=op.path_in_repo,
                                local_path=op.local_path,
                                snapshot_path=snapshot_path,
                            )
                        )
                    else:
                        new_ops.append(op)
                    op_entries.append(entry)
                rewritten_ops = new_ops
                op_shas = tuple(entry.get("sha256") for entry in op_entries)
                envelope = {
                    "contract_version": QUEUE_CONTRACT_VERSION,
                    "sequence": sequence,
                    "message": message,
                    "ops": op_entries,
                }
                atomic_write_text(state_path, json.dumps(envelope, indent=2, sort_keys=True) + "\n")
            else:
                op_shas = tuple(None for _ in ops)
        except BaseException:
            # Clean up partial artifacts so the state dir is left
            # consistent (no stale envelope + no stale snapshot dir).
            if state_path is not None:
                state_path.unlink(missing_ok=True)
            if (
                snapshot_dir is not None
                and self._state_dir is not None
                and _is_inside(snapshot_dir, self._state_dir / _SNAPSHOTS_SUBDIR)
            ):
                shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise

        self._jobs.put(
            _UploadJob(rewritten_ops, message, state_path, snapshot_dir, op_shas=op_shas)
        )
        LOGGER.info("Queued background upload: %s", message)

    def resume_pending(self) -> int:
        """Resume all pending envelopes in sequence order.

        * Legacy envelopes (UUID filename, no ``contract_version``)
          are upgraded DURABLY: each add op's canonical local_path is
          snapshotted into the queue-owned snapshot directory, the
          SHA-256 of the snapshot bytes is recorded, and a new
          ``bg-upload-v1`` envelope is written and atomically swapped
          onto the state dir. Missing canonical files fail closed --
          the original legacy envelope is preserved.
        * Current envelopes are validated. ``snapshot_path`` values
          that resolve outside ``state_dir/snapshots`` are rejected.
        """
        if self._state_dir is None:
            return 0
        state_dir = self._state_dir

        # Pre-scan every JSON object (legacy or current) and bucket
        # into current and legacy lists.
        current: list[tuple[int, Path, dict[str, Any]]] = []
        legacy: list[tuple[Path, dict[str, Any]]] = []
        seen_sequences: set[int] = set()

        # Build the bucket for every JSON file under state_dir.
        for path in sorted(state_dir.glob("*.json")):
            payload = _read_legacy_or_current_envelope(path)
            if payload is None:
                raise ValueError(f"Malformed envelope: {path} -- refusing to resume")
            if _is_legacy_envelope(payload):
                legacy.append((path, payload))
            else:
                if payload.get("contract_version") != QUEUE_CONTRACT_VERSION:
                    raise ValueError(f"Unknown contract_version in {path}; refusing to resume")
                sequence = int(payload["sequence"])
                if sequence in seen_sequences:
                    raise ValueError(
                        f"Duplicate sequence {sequence} in {path} -- refusing to upload"
                    )
                seen_sequences.add(sequence)
                current.append((sequence, path, payload))

        # Upgrade legacy envelopes durably first. A failed upgrade
        # leaves the legacy envelope untouched and surfaces the
        # failure; we continue with the remaining envelopes so a
        # single bad envelope does not strand everything else.
        upgraded: list[tuple[int, Path, dict[str, Any], Path]] = []
        for legacy_path, payload in legacy:
            try:
                sequence, env_payload, snapshot_dir = self._upgrade_legacy_envelope(
                    state_dir, legacy_path, payload
                )
            except (FileNotFoundError, OSError, ValueError) as error:
                # Already recorded in self._failures by the upgrade.
                detail = f"legacy envelope {legacy_path.name}: skipped, {error}"
                LOGGER.error("Background upload failed: %s", detail)
                self._failures.append(detail)
                continue
            if sequence in seen_sequences:
                # Roll back the upgrade so we do not strand a
                # snapshot that nothing tracks.
                upgraded_path = state_dir / f"{sequence:06d}.json"
                upgraded_path.unlink(missing_ok=True)
                if snapshot_dir.is_dir() and _is_inside(
                    snapshot_dir, state_dir / _SNAPSHOTS_SUBDIR
                ):
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                raise ValueError(
                    f"Legacy upgrade produced duplicate sequence {sequence}; refusing to upload"
                )
            seen_sequences.add(sequence)
            # Use the UPGRADED envelope path, not the legacy path.
            upgraded_path = state_dir / f"{sequence:06d}.json"
            upgraded.append((sequence, upgraded_path, env_payload, snapshot_dir))

        # Now emit jobs for every current + upgraded envelope in
        # sequence order.
        all_jobs: list[tuple[int, Path, dict[str, Any], Path]] = [
            (sequence, path, payload, _snapshot_dir_for_sequence(state_dir, sequence))
            for (sequence, path, payload) in current
        ]
        all_jobs.extend(upgraded)
        all_jobs.sort(key=lambda item: item[0])

        for _sequence, state_path, envelope, snapshot_dir in all_jobs:
            # Defensive: the snapshot directory must be inside
            # state_dir/snapshots.
            if not _is_inside(snapshot_dir, state_dir / _SNAPSHOTS_SUBDIR):
                detail = (
                    f"Computed snapshot dir {snapshot_dir} is outside state_dir/snapshots; "
                    "refusing to resume"
                )
                LOGGER.error("Background upload failed: %s", detail)
                self._failures.append(detail)
                continue
            # Validate every add op's snapshot_path is also inside.
            try:
                ops = []
                op_shas: list[str | None] = []
                for entry in envelope.get("ops", []):
                    action = entry.get("action")
                    local_path_str = entry.get("local_path")
                    snapshot_path_str = entry.get("snapshot_path")
                    local_path = Path(local_path_str) if local_path_str else None
                    snapshot_path = Path(snapshot_path_str) if snapshot_path_str else None
                    if action == "add":
                        if snapshot_path is None:
                            raise ValueError(f"Add op in {state_path} is missing snapshot_path")
                        if not _is_inside(snapshot_path, snapshot_dir):
                            raise ValueError(
                                f"Add op snapshot {snapshot_path} is outside {snapshot_dir}; "
                                "refusing to upload"
                            )
                    ops.append(
                        PublicationOp(
                            action=action,
                            path_in_repo=entry["path_in_repo"],
                            local_path=local_path,
                            snapshot_path=snapshot_path,
                        )
                    )
                    op_shas.append(entry.get("sha256"))
            except (FileNotFoundError, OSError, ValueError) as error:
                detail = f"resume validation failed for {state_path.name}: {error}"
                LOGGER.error("Background upload failed: %s", detail)
                self._failures.append(detail)
                continue
            self._jobs.put(
                _UploadJob(
                    ops,
                    str(envelope["message"]),
                    state_path,
                    snapshot_dir,
                    op_shas=tuple(op_shas),
                )
            )
        return len(all_jobs)

    def _upgrade_legacy_envelope(
        self, state_dir: Path, legacy_path: Path, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], Path]:
        """Upgrade a single legacy envelope to the current shape.

        Returns ``(sequence, new_envelope_payload, snapshot_dir)``.

        On any failure (missing canonical file, copy error), the
        original legacy envelope is preserved and the failure is
        surfaced via ``self._failures`` by the caller.
        """
        sequence = self._allocate_sequence()
        snapshot_dir = _snapshot_dir_for_sequence(state_dir, sequence)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        op_entries: list[dict[str, Any]] = []
        new_ops: list[PublicationOp] = []

        for op_index, entry in enumerate(payload.get("ops", [])):
            action = entry.get("action")
            path_in_repo = entry.get("path_in_repo")
            local_path_str = entry.get("local_path")
            local_path = Path(local_path_str) if local_path_str else None

            op_entry: dict[str, Any] = {
                "action": action,
                "path_in_repo": path_in_repo,
                "local_path": local_path_str,
            }
            if action == "add" and local_path is not None:
                if not local_path.is_file():
                    raise FileNotFoundError(f"Cannot snapshot missing canonical file: {local_path}")
                op_dir = snapshot_dir / f"{op_index:03d}"
                op_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = op_dir / local_path.name
                _independent_copy(local_path, snapshot_path)
                sha = _sha256_file(snapshot_path)
                op_entry["snapshot_path"] = str(snapshot_path)
                op_entry["sha256"] = sha
                new_ops.append(
                    PublicationOp(
                        action=action,
                        path_in_repo=path_in_repo,
                        local_path=local_path,
                        snapshot_path=snapshot_path,
                    )
                )
            else:
                new_ops.append(
                    PublicationOp(
                        action=action,
                        path_in_repo=path_in_repo,
                        local_path=local_path,
                        snapshot_path=None,
                    )
                )
            op_entries.append(op_entry)

        new_envelope = {
            "contract_version": QUEUE_CONTRACT_VERSION,
            "sequence": sequence,
            "message": str(payload.get("message", "")),
            "ops": op_entries,
        }
        upgraded_path = state_dir / f"{sequence:06d}.json"
        atomic_write_text(
            upgraded_path,
            json.dumps(new_envelope, indent=2, sort_keys=True) + "\n",
        )
        # Only NOW that the upgraded envelope is on disk do we
        # remove the legacy file.
        legacy_path.unlink(missing_ok=True)
        return sequence, new_envelope, snapshot_dir

    def close_and_wait(self) -> list[str]:
        if not self._closed:
            self._closed = True
            self._jobs.put(None)
        self._thread.join()
        return list(self._failures)

    def upload_synchronously(self, ops: list[PublicationOp], commit_message: str) -> None:
        """Run one final upload after the regional queue has drained.

        This deliberately reuses the queue's configured upload
        collaborator (and therefore the same authenticated Hub
        session/test double) while preserving strict ordering.
        """
        if not self._closed:
            raise RuntimeError("Synchronous final upload requires a drained, closed queue")
        self._upload(ops, commit_message)

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                try:
                    # Verify each add op's snapshot against its recorded
                    # sha256 BEFORE the upload is attempted. A mismatch
                    # is recorded as a failure and the envelope + snapshot
                    # are preserved for retry (no upload is called).
                    sha_mismatch = False
                    for op, recorded_sha in zip(job.ops, job.op_shas, strict=True):
                        snapshot = op.snapshot_path or op.local_path
                        if op.action == "add" and snapshot is not None and recorded_sha:
                            actual = _sha256_file(snapshot)
                            if actual != recorded_sha:
                                sha_mismatch = True
                                detail = (
                                    f"{job.message}: SHA mismatch for {snapshot} "
                                    f"(recorded {recorded_sha}, actual {actual})"
                                )
                                LOGGER.error("Background upload failed: %s", detail)
                                self._failures.append(detail)
                                break
                    if sha_mismatch:
                        # Do NOT delete the envelope or snapshot; the
                        # caller can retry after fixing the snapshot.
                        continue
                    # Pass the original ops to the upload callback.
                    # The op's local_path is the canonical (for the
                    # downstream retirement check); the snapshot_path
                    # is set on the side so the upload callback can
                    # prefer the immutable snapshot over the
                    # canonical for the actual bytes.
                    last_error: Exception | None = None
                    for _ in range(self._attempts):
                        try:
                            self._upload(job.ops, job.message)
                            last_error = None
                            break
                        except Exception as error:
                            last_error = error
                    if last_error is not None:
                        raise last_error
                    # Successful upload: remove envelope + snapshot dir.
                    if job.state_path is not None:
                        _delete_envelope_with_snapshot(job.state_path, job.snapshot_dir)
                    LOGGER.info("Background upload complete: %s", job.message)
                except Exception as error:
                    detail = f"{job.message}: {error}"
                    LOGGER.error("Background upload failed: %s", detail)
                    self._failures.append(detail)
            finally:
                self._jobs.task_done()


__all__ = ["QUEUE_CONTRACT_VERSION", "BackgroundUploadQueue", "UploadOperation", "UploadOps"]
