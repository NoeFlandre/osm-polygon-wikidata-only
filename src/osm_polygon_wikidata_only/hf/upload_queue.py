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
    return max(_read_highwater(state_dir), _scanned_sequence(state_dir)) + 1


def _scanned_sequence(state_dir: Path) -> int:
    """Return the highest sequence discoverable from state envelopes."""
    highest = 0
    for path in state_dir.glob("*.json"):
        highest = max(highest, _sequence_from_state_path(path))
    return highest


def _sequence_from_state_path(path: Path) -> int:
    """Read one sequence-named or legacy envelope's sequence."""
    match = _SEQUENCE_FILE.match(path.name)
    if match is not None:
        return int(match.group(1))
    envelope = _read_legacy_or_current_envelope(path)
    sequence = envelope.get("sequence") if envelope is not None else None
    return sequence if isinstance(sequence, int) and sequence > 0 else 0


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _required_string(entry: dict[str, Any], key: str) -> str:
    """Return a required string field from a persisted envelope entry."""
    value = entry.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Envelope operation field {key!r} must be a string")
    return value


def _current_sequence(payload: dict[str, Any], path: Path) -> int:
    """Validate and return a current envelope's positive sequence."""
    raw_sequence = payload.get("sequence")
    if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool) or raw_sequence < 1:
        raise ValueError(f"Current envelope {path} has an invalid sequence")
    return raw_sequence


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
        """Persist and enqueue one upload job."""
        self._ensure_open()
        rewritten_ops, state_path, snapshot_dir, op_shas = self._submission_payload(ops, message)
        self._jobs.put(
            _UploadJob(rewritten_ops, message, state_path, snapshot_dir, op_shas=op_shas)
        )
        LOGGER.info("Queued background upload: %s", message)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("upload queue is closed")

    def _cleanup_failed_submission(
        self,
        state_path: Path,
        snapshot_dir: Path | None,
    ) -> None:
        """Remove partial persisted artifacts after a failed submit."""
        state_path.unlink(missing_ok=True)
        if snapshot_dir is None or self._state_dir is None:
            return
        if _is_inside(snapshot_dir, self._state_dir / _SNAPSHOTS_SUBDIR):
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    def _submission_payload(
        self,
        ops: UploadOps,
        message: str,
    ) -> tuple[UploadOps, Path | None, Path | None, tuple[str | None, ...]]:
        """Build a persisted or in-memory submission payload."""
        if self._state_dir is None:
            return list(ops), None, None, tuple(None for _ in ops)
        return self._persist_submission(ops, message)

    def _persist_submission(
        self,
        ops: UploadOps,
        message: str,
    ) -> tuple[UploadOps, Path, Path | None, tuple[str | None, ...]]:
        """Snapshot operations and durably write their current envelope."""
        if self._state_dir is None:
            raise RuntimeError("state_dir is required for persisted submissions")
        sequence = self._allocate_sequence()
        state_path = self._state_dir / f"{sequence:06d}.json"
        snapshot_dir = _snapshot_dir_for_sequence(self._state_dir, sequence)
        try:
            rewritten_ops, op_entries, snapshot_dir = self._snapshot_operations(ops, sequence)
            envelope = {
                "contract_version": QUEUE_CONTRACT_VERSION,
                "sequence": sequence,
                "message": message,
                "ops": op_entries,
            }
            atomic_write_text(state_path, json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        except BaseException:
            self._cleanup_failed_submission(state_path, snapshot_dir)
            raise
        op_shas = tuple(entry.get("sha256") for entry in op_entries)
        return rewritten_ops, state_path, snapshot_dir, op_shas

    def _snapshot_operations(
        self,
        ops: UploadOps,
        sequence: int,
    ) -> tuple[UploadOps, list[dict[str, Any]], Path | None]:
        """Create immutable snapshots and return rewritten operations."""
        snapshot_dir = self._submission_snapshot_dir(ops, sequence)
        rewritten: UploadOps = []
        entries: list[dict[str, Any]] = []
        for op_index, op in enumerate(ops):
            rewritten_op, entry = self._snapshot_operation(op, op_index, snapshot_dir)
            rewritten.append(rewritten_op)
            entries.append(entry)
        return rewritten, entries, snapshot_dir

    def _submission_snapshot_dir(self, ops: UploadOps, sequence: int) -> Path | None:
        """Create a queue-owned snapshot directory when an add is present."""
        if self._state_dir is None or not any(op.action == "add" for op in ops):
            return None
        snapshot_dir = _snapshot_dir_for_sequence(self._state_dir, sequence)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir

    def _snapshot_operation(
        self,
        op: PublicationOp,
        op_index: int,
        snapshot_dir: Path | None,
    ) -> tuple[PublicationOp, dict[str, Any]]:
        """Snapshot one add operation or preserve a delete operation."""
        entry: dict[str, Any] = {
            "action": op.action,
            "path_in_repo": op.path_in_repo,
            "local_path": str(op.local_path) if op.local_path else None,
        }
        if op.action != "add" or op.local_path is None:
            return op, entry
        return self._snapshot_add_operation(op, op_index, snapshot_dir, entry)

    @staticmethod
    def _snapshot_add_operation(
        op: PublicationOp,
        op_index: int,
        snapshot_dir: Path | None,
        entry: dict[str, Any],
    ) -> tuple[PublicationOp, dict[str, Any]]:
        """Copy one add operation into its immutable queue snapshot."""
        if op.local_path is None:
            raise ValueError("add operation must carry a local path")
        if not op.local_path.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing file: {op.local_path}")
        if snapshot_dir is None:
            raise RuntimeError("snapshot_dir must be set when any op is an add")
        op_dir = snapshot_dir / f"{op_index:03d}"
        op_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = op_dir / op.local_path.name
        _independent_copy(op.local_path, snapshot_path)
        sha = _sha256_file(snapshot_path)
        entry["snapshot_path"] = str(snapshot_path)
        entry["sha256"] = sha
        rewritten = PublicationOp(
            action=op.action,
            path_in_repo=op.path_in_repo,
            local_path=op.local_path,
            snapshot_path=snapshot_path,
        )
        return rewritten, entry

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
        current, legacy, seen_sequences = self._pending_envelopes(state_dir)
        upgraded = self._upgrade_pending_legacy(state_dir, legacy, seen_sequences)
        all_jobs = self._ordered_pending_jobs(state_dir, current, upgraded)
        for _sequence, state_path, envelope, snapshot_dir in all_jobs:
            self._enqueue_pending_job(state_dir, state_path, envelope, snapshot_dir)
        return len(all_jobs)

    def _pending_envelopes(
        self,
        state_dir: Path,
    ) -> tuple[
        list[tuple[int, Path, dict[str, Any]]],
        list[tuple[Path, dict[str, Any]]],
        set[int],
    ]:
        """Read and classify every pending envelope before enqueueing."""
        current: list[tuple[int, Path, dict[str, Any]]] = []
        legacy: list[tuple[Path, dict[str, Any]]] = []
        seen_sequences: set[int] = set()
        for path in sorted(state_dir.glob("*.json")):
            classification = self._classify_pending_envelope(path, seen_sequences)
            if classification[0] == "legacy":
                legacy.append((path, classification[1]))
            else:
                sequence = classification[2]
                if sequence is None:
                    raise ValueError(f"Current envelope {path} has no sequence")
                current.append((sequence, path, classification[1]))
        return current, legacy, seen_sequences

    @staticmethod
    def _classify_pending_envelope(
        path: Path,
        seen_sequences: set[int],
    ) -> tuple[str, dict[str, Any], int | None]:
        """Validate one pending envelope and return its classification."""
        payload = _read_legacy_or_current_envelope(path)
        if payload is None:
            raise ValueError(f"Malformed envelope: {path} -- refusing to resume")
        if _is_legacy_envelope(payload):
            return "legacy", payload, None
        if payload.get("contract_version") != QUEUE_CONTRACT_VERSION:
            raise ValueError(f"Unknown contract_version in {path}; refusing to resume")
        sequence = _current_sequence(payload, path)
        if sequence in seen_sequences:
            raise ValueError(f"Duplicate sequence {sequence} in {path} -- refusing to upload")
        seen_sequences.add(sequence)
        return "current", payload, sequence

    def _upgrade_pending_legacy(
        self,
        state_dir: Path,
        legacy: list[tuple[Path, dict[str, Any]]],
        seen_sequences: set[int],
    ) -> list[tuple[int, Path, dict[str, Any], Path]]:
        """Durably upgrade legacy envelopes and retain valid upgrades."""
        upgraded: list[tuple[int, Path, dict[str, Any], Path]] = []
        for legacy_path, payload in legacy:
            result = self._try_upgrade_legacy(state_dir, legacy_path, payload)
            if result is None:
                continue
            sequence, env_payload, snapshot_dir = result
            if sequence in seen_sequences:
                self._remove_failed_upgrade(state_dir, sequence, snapshot_dir)
                raise ValueError(
                    f"Legacy upgrade produced duplicate sequence {sequence}; refusing to upload"
                )
            seen_sequences.add(sequence)
            upgraded_path = state_dir / f"{sequence:06d}.json"
            upgraded.append((sequence, upgraded_path, env_payload, snapshot_dir))
        return upgraded

    def _try_upgrade_legacy(
        self,
        state_dir: Path,
        legacy_path: Path,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any], Path] | None:
        """Upgrade one legacy envelope, recording recoverable failures."""
        try:
            return self._upgrade_legacy_envelope(state_dir, legacy_path, payload)
        except (FileNotFoundError, OSError, ValueError) as error:
            detail = f"legacy envelope {legacy_path.name}: skipped, {error}"
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
            return None

    @staticmethod
    def _remove_failed_upgrade(state_dir: Path, sequence: int, snapshot_dir: Path) -> None:
        """Remove an upgraded duplicate that cannot be queued."""
        upgraded_path = state_dir / f"{sequence:06d}.json"
        upgraded_path.unlink(missing_ok=True)
        if snapshot_dir.is_dir() and _is_inside(snapshot_dir, state_dir / _SNAPSHOTS_SUBDIR):
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    @staticmethod
    def _ordered_pending_jobs(
        state_dir: Path,
        current: list[tuple[int, Path, dict[str, Any]]],
        upgraded: list[tuple[int, Path, dict[str, Any], Path]],
    ) -> list[tuple[int, Path, dict[str, Any], Path]]:
        """Combine current and upgraded envelopes in sequence order."""
        jobs = [
            (sequence, path, payload, _snapshot_dir_for_sequence(state_dir, sequence))
            for sequence, path, payload in current
        ]
        jobs.extend(upgraded)
        jobs.sort(key=lambda item: item[0])
        return jobs

    def _enqueue_pending_job(
        self,
        state_dir: Path,
        state_path: Path,
        envelope: dict[str, Any],
        snapshot_dir: Path,
    ) -> None:
        """Validate one pending job and enqueue it or record its failure."""
        if not _is_inside(snapshot_dir, state_dir / _SNAPSHOTS_SUBDIR):
            detail = (
                f"Computed snapshot dir {snapshot_dir} is outside state_dir/snapshots; "
                "refusing to resume"
            )
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
            return
        try:
            ops, op_shas = self._resume_operations(envelope, state_path, snapshot_dir)
        except (FileNotFoundError, OSError, ValueError) as error:
            detail = f"resume validation failed for {state_path.name}: {error}"
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
            return
        self._jobs.put(
            _UploadJob(
                ops,
                str(envelope["message"]),
                state_path,
                snapshot_dir,
                op_shas=op_shas,
            )
        )

    @staticmethod
    def _resume_operations(
        envelope: dict[str, Any],
        state_path: Path,
        snapshot_dir: Path,
    ) -> tuple[list[PublicationOp], tuple[str | None, ...]]:
        """Validate and reconstruct operations from one envelope."""
        operations: list[PublicationOp] = []
        op_shas: list[str | None] = []
        for entry in envelope.get("ops", []):
            operation, sha = BackgroundUploadQueue._resume_operation(
                entry, state_path, snapshot_dir
            )
            operations.append(operation)
            op_shas.append(sha)
        return operations, tuple(op_shas)

    @staticmethod
    def _resume_operation(
        entry: dict[str, Any],
        state_path: Path,
        snapshot_dir: Path,
    ) -> tuple[PublicationOp, str | None]:
        """Validate one envelope operation and reconstruct its value."""
        action = _required_string(entry, "action")
        path_in_repo = _required_string(entry, "path_in_repo")
        local_path_str = entry.get("local_path")
        snapshot_path_str = entry.get("snapshot_path")
        local_path = Path(local_path_str) if local_path_str else None
        snapshot_path = Path(snapshot_path_str) if snapshot_path_str else None
        if action == "add":
            BackgroundUploadQueue._validate_resume_snapshot(snapshot_path, state_path, snapshot_dir)
        return (
            PublicationOp(
                action=action,
                path_in_repo=path_in_repo,
                local_path=local_path,
                snapshot_path=snapshot_path,
            ),
            entry.get("sha256"),
        )

    @staticmethod
    def _validate_resume_snapshot(
        snapshot_path: Path | None,
        state_path: Path,
        snapshot_dir: Path,
    ) -> None:
        """Require an add snapshot to stay inside its queue directory."""
        if snapshot_path is None:
            raise ValueError(f"Add op in {state_path} is missing snapshot_path")
        if not _is_inside(snapshot_path, snapshot_dir):
            raise ValueError(
                f"Add op snapshot {snapshot_path} is outside {snapshot_dir}; refusing to upload"
            )

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
        _new_ops, op_entries = self._upgrade_legacy_operations(payload, snapshot_dir)

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

    def _upgrade_legacy_operations(
        self,
        payload: dict[str, Any],
        snapshot_dir: Path,
    ) -> tuple[UploadOps, list[dict[str, Any]]]:
        """Snapshot every operation from a legacy envelope."""
        new_ops: UploadOps = []
        op_entries: list[dict[str, Any]] = []
        for op_index, entry in enumerate(payload.get("ops", [])):
            operation, op_entry = self._upgrade_legacy_operation(entry, op_index, snapshot_dir)
            new_ops.append(operation)
            op_entries.append(op_entry)
        return new_ops, op_entries

    @staticmethod
    def _upgrade_legacy_operation(
        entry: dict[str, Any],
        op_index: int,
        snapshot_dir: Path,
    ) -> tuple[PublicationOp, dict[str, Any]]:
        """Upgrade and snapshot one legacy operation."""
        action = _required_string(entry, "action")
        path_in_repo = _required_string(entry, "path_in_repo")
        local_path_str = entry.get("local_path")
        local_path = Path(local_path_str) if local_path_str else None
        op_entry: dict[str, Any] = {
            "action": action,
            "path_in_repo": path_in_repo,
            "local_path": local_path_str,
        }
        if action != "add" or local_path is None:
            return (
                PublicationOp(
                    action=action,
                    path_in_repo=path_in_repo,
                    local_path=local_path,
                    snapshot_path=None,
                ),
                op_entry,
            )
        if not local_path.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing canonical file: {local_path}")
        op_dir = snapshot_dir / f"{op_index:03d}"
        op_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = op_dir / local_path.name
        _independent_copy(local_path, snapshot_path)
        op_entry["snapshot_path"] = str(snapshot_path)
        op_entry["sha256"] = _sha256_file(snapshot_path)
        return (
            PublicationOp(
                action=action,
                path_in_repo=path_in_repo,
                local_path=local_path,
                snapshot_path=snapshot_path,
            ),
            op_entry,
        )

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
                    self._process_job(job)
                except Exception as error:
                    detail = f"{job.message}: {error}"
                    LOGGER.error("Background upload failed: %s", detail)
                    self._failures.append(detail)
            finally:
                self._jobs.task_done()

    def _process_job(self, job: _UploadJob) -> None:
        """Verify, upload, and clean up one queued job."""
        if not self._verify_job_snapshot(job):
            return
        self._upload_with_retries(job)
        if job.state_path is not None:
            _delete_envelope_with_snapshot(job.state_path, job.snapshot_dir)
        LOGGER.info("Background upload complete: %s", job.message)

    def _verify_job_snapshot(self, job: _UploadJob) -> bool:
        """Verify all recorded snapshot hashes before uploading."""
        for op, recorded_sha in zip(job.ops, job.op_shas, strict=True):
            detail = self._snapshot_mismatch(job.message, op, recorded_sha)
            if detail is None:
                continue
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
            return False
        return True

    @staticmethod
    def _snapshot_mismatch(
        message: str,
        op: PublicationOp,
        recorded_sha: str | None,
    ) -> str | None:
        """Return a failure message when one add snapshot hash is wrong."""
        snapshot = BackgroundUploadQueue._snapshot_path(op)
        if snapshot is None or not recorded_sha:
            return None
        return BackgroundUploadQueue._snapshot_hash_mismatch(message, snapshot, recorded_sha)

    @staticmethod
    def _snapshot_hash_mismatch(
        message: str,
        snapshot: Path,
        recorded_sha: str,
    ) -> str | None:
        actual = _sha256_file(snapshot)
        if actual == recorded_sha:
            return None
        return f"{message}: SHA mismatch for {snapshot} (recorded {recorded_sha}, actual {actual})"

    @staticmethod
    def _snapshot_path(op: PublicationOp) -> Path | None:
        """Return the immutable file used to verify an add operation."""
        if op.action != "add":
            return None
        return op.snapshot_path or op.local_path

    def _upload_with_retries(self, job: _UploadJob) -> None:
        """Attempt one upload up to the configured retry count."""
        last_error: Exception | None = None
        for _ in range(self._attempts):
            try:
                self._upload(job.ops, job.message)
                return
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise last_error


__all__ = ["QUEUE_CONTRACT_VERSION", "BackgroundUploadQueue", "UploadOperation", "UploadOps"]
