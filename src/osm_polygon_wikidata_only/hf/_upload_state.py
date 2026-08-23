"""Durable state boundary for the background upload queue.

This module owns the on-disk upload contract: sequence allocation, immutable
snapshots, envelope serialization, legacy upgrades, and resume validation.
The queue itself only coordinates workers and upload callbacks.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

UploadOps = list[PublicationOp]
CopyFile = Callable[[Path, Path], None]
HashFile = Callable[[Path], str]

QUEUE_CONTRACT_VERSION = "bg-upload-v1"

_SEQUENCE_FILE = re.compile(r"^(\d+)\.json$")
_HIGHWATER_FILENAME = ".highwater"
_SNAPSHOTS_SUBDIR = "snapshots"


@dataclass(frozen=True)
class StoredUpload:
    """One durable upload ready for queue processing."""

    ops: UploadOps
    message: str
    state_path: Path
    snapshot_dir: Path | None
    op_shas: tuple[str | None, ...]


@dataclass(frozen=True)
class ResumeResult:
    """Validated jobs and recoverable failures found during resume."""

    uploads: tuple[StoredUpload, ...]
    failures: tuple[str, ...]
    discovered_count: int


def _read_highwater(state_dir: Path) -> int:
    """Read the allocated sequence high-water mark."""
    path = state_dir / _HIGHWATER_FILENAME
    if not path.is_file():
        return 0
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0
    return max(value, 0)


def _write_highwater(state_dir: Path, value: int) -> None:
    """Persist the allocated sequence high-water mark atomically."""
    atomic_write_text(state_dir / _HIGHWATER_FILENAME, f"{int(value)}\n")


def _next_sequence_from_state_dir(state_dir: Path) -> int:
    """Return the next sequence after persisted state."""
    if not state_dir.is_dir():
        return 1
    return max(_read_highwater(state_dir), _scanned_sequence(state_dir)) + 1


def _scanned_sequence(state_dir: Path) -> int:
    highest = 0
    for path in state_dir.glob("*.json"):
        highest = max(highest, _sequence_from_state_path(path))
    return highest


def _sequence_from_state_path(path: Path) -> int:
    match = _SEQUENCE_FILE.match(path.name)
    if match is not None:
        return int(match.group(1))
    envelope = _read_legacy_or_current_envelope(path)
    sequence = envelope.get("sequence") if envelope is not None else None
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        return 0
    return max(sequence, 0)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _independent_copy(source: Path, target: Path) -> None:
    """Copy a source file without sharing its inode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for malformed input."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_envelope(path: Path) -> dict[str, Any] | None:
    """Read a current upload envelope."""
    raw = _read_json_object(path)
    if raw is None or raw.get("contract_version") != QUEUE_CONTRACT_VERSION:
        return None
    return raw


def _read_legacy_or_current_envelope(path: Path) -> dict[str, Any] | None:
    """Read either the current or legacy envelope shape."""
    return _read_json_object(path)


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Envelope operation field {key!r} must be a string")
    return value


def _current_sequence(payload: dict[str, Any], path: Path) -> int:
    raw_sequence = payload.get("sequence")
    if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool) or raw_sequence < 1:
        raise ValueError(f"Current envelope {path} has an invalid sequence")
    return raw_sequence


def _is_legacy_envelope(payload: dict[str, Any]) -> bool:
    if "contract_version" in payload:
        return False
    return isinstance(payload.get("message"), str) and isinstance(payload.get("ops"), list)


def _snapshot_dir_for_sequence(state_dir: Path, sequence: int) -> Path:
    return state_dir / _SNAPSHOTS_SUBDIR / f"{sequence:06d}"


def _is_inside(child: Path, parent: Path) -> bool:
    """Return whether a resolved path is inside or equal to its parent."""
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


def _remove_failed_upgrade(state_dir: Path, sequence: int, snapshot_dir: Path) -> None:
    """Remove an upgraded duplicate that cannot be queued."""
    (state_dir / f"{sequence:06d}.json").unlink(missing_ok=True)
    if snapshot_dir.is_dir() and _is_inside(snapshot_dir, state_dir / _SNAPSHOTS_SUBDIR):
        shutil.rmtree(snapshot_dir, ignore_errors=True)


class UploadStateStore:
    """Own the durable state contract used by ``BackgroundUploadQueue``."""

    def __init__(
        self,
        state_dir: Path,
        *,
        copy_file: CopyFile = _independent_copy,
        hash_file: HashFile = _sha256_file,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._copy_file = copy_file
        self._hash_file = hash_file
        self._next_sequence_lock = threading.Lock()
        self._next_sequence = _next_sequence_from_state_dir(state_dir)

    @property
    def next_sequence(self) -> int:
        """Return the next sequence that will be allocated."""
        return self._next_sequence

    def _allocate_sequence(self) -> int:
        with self._next_sequence_lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            _write_highwater(self.state_dir, sequence)
            return sequence

    def persist(self, ops: UploadOps, message: str) -> StoredUpload:
        """Snapshot and durably persist one upload submission."""
        sequence = self._allocate_sequence()
        state_path = self.state_dir / f"{sequence:06d}.json"
        snapshot_dir = _snapshot_dir_for_sequence(self.state_dir, sequence)
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
            self.cleanup_failed_submission(state_path, snapshot_dir)
            raise
        return StoredUpload(
            ops=rewritten_ops,
            message=message,
            state_path=state_path,
            snapshot_dir=snapshot_dir,
            op_shas=tuple(entry.get("sha256") for entry in op_entries),
        )

    def cleanup_failed_submission(self, state_path: Path, snapshot_dir: Path | None) -> None:
        """Remove artifacts created by a failed submission."""
        state_path.unlink(missing_ok=True)
        if snapshot_dir is None:
            return
        if _is_inside(snapshot_dir, self.state_dir / _SNAPSHOTS_SUBDIR):
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    def _snapshot_operations(
        self,
        ops: UploadOps,
        sequence: int,
    ) -> tuple[UploadOps, list[dict[str, Any]], Path | None]:
        snapshot_dir = self._submission_snapshot_dir(ops, sequence)
        rewritten: UploadOps = []
        entries: list[dict[str, Any]] = []
        for op_index, op in enumerate(ops):
            rewritten_op, entry = self._snapshot_operation(op, op_index, snapshot_dir)
            rewritten.append(rewritten_op)
            entries.append(entry)
        return rewritten, entries, snapshot_dir

    def _submission_snapshot_dir(self, ops: UploadOps, sequence: int) -> Path | None:
        if not any(op.action == "add" for op in ops):
            return None
        snapshot_dir = _snapshot_dir_for_sequence(self.state_dir, sequence)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir

    def _snapshot_operation(
        self,
        op: PublicationOp,
        op_index: int,
        snapshot_dir: Path | None,
    ) -> tuple[PublicationOp, dict[str, Any]]:
        entry: dict[str, Any] = {
            "action": op.action,
            "path_in_repo": op.path_in_repo,
            "local_path": str(op.local_path) if op.local_path else None,
        }
        if op.action != "add":
            return op, entry
        if snapshot_dir is None:
            raise RuntimeError("snapshot_dir must be set when any op is an add")
        return self._snapshot_add_operation(op, op_index, snapshot_dir, entry)

    def _snapshot_add_operation(
        self,
        op: PublicationOp,
        op_index: int,
        snapshot_dir: Path,
        entry: dict[str, Any],
    ) -> tuple[PublicationOp, dict[str, Any]]:
        if op.local_path is None:
            raise ValueError("add operation must carry a local path")
        if not op.local_path.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing file: {op.local_path}")
        op_dir = snapshot_dir / f"{op_index:03d}"
        op_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = op_dir / op.local_path.name
        self._copy_file(op.local_path, snapshot_path)
        entry["snapshot_path"] = str(snapshot_path)
        entry["sha256"] = self._hash_file(snapshot_path)
        return (
            PublicationOp(
                action=op.action,
                path_in_repo=op.path_in_repo,
                local_path=op.local_path,
                snapshot_path=snapshot_path,
            ),
            entry,
        )

    def resume_pending(self) -> ResumeResult:
        """Validate and reconstruct all pending envelopes in sequence order."""
        current, legacy, seen_sequences = self._pending_envelopes()
        failures: list[str] = []
        upgraded = self._upgrade_pending_legacy(legacy, seen_sequences, failures)
        all_jobs = self._ordered_pending_jobs(current, upgraded)
        uploads: list[StoredUpload] = []
        for _sequence, state_path, envelope, snapshot_dir in all_jobs:
            restored, failure = self._restore_pending_job(
                state_path,
                envelope,
                snapshot_dir,
            )
            if failure is not None:
                failures.append(failure)
            elif restored is not None:
                uploads.append(restored)
        return ResumeResult(tuple(uploads), tuple(failures), len(all_jobs))

    def _pending_envelopes(
        self,
    ) -> tuple[
        list[tuple[int, Path, dict[str, Any]]],
        list[tuple[Path, dict[str, Any]]],
        set[int],
    ]:
        current: list[tuple[int, Path, dict[str, Any]]] = []
        legacy: list[tuple[Path, dict[str, Any]]] = []
        seen_sequences: set[int] = set()
        for path in sorted(self.state_dir.glob("*.json")):
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
        legacy: list[tuple[Path, dict[str, Any]]],
        seen_sequences: set[int],
        failures: list[str],
    ) -> list[tuple[int, Path, dict[str, Any], Path]]:
        upgraded: list[tuple[int, Path, dict[str, Any], Path]] = []
        for legacy_path, payload in legacy:
            try:
                result = self._upgrade_legacy_envelope(legacy_path, payload)
            except (FileNotFoundError, OSError, ValueError) as error:
                failures.append(f"legacy envelope {legacy_path.name}: skipped, {error}")
                continue
            sequence, envelope, snapshot_dir = result
            if sequence in seen_sequences:
                _remove_failed_upgrade(self.state_dir, sequence, snapshot_dir)
                raise ValueError(
                    f"Legacy upgrade produced duplicate sequence {sequence}; refusing to upload"
                )
            seen_sequences.add(sequence)
            upgraded.append(
                (sequence, self.state_dir / f"{sequence:06d}.json", envelope, snapshot_dir)
            )
        return upgraded

    def _ordered_pending_jobs(
        self,
        current: list[tuple[int, Path, dict[str, Any]]],
        upgraded: list[tuple[int, Path, dict[str, Any], Path]],
    ) -> list[tuple[int, Path, dict[str, Any], Path]]:
        jobs = [
            (sequence, path, payload, _snapshot_dir_for_sequence(self.state_dir, sequence))
            for sequence, path, payload in current
        ]
        jobs.extend(upgraded)
        jobs.sort(key=lambda item: item[0])
        return jobs

    def _restore_pending_job(
        self,
        state_path: Path,
        envelope: dict[str, Any],
        snapshot_dir: Path,
    ) -> tuple[StoredUpload | None, str | None]:
        snapshots_root = self.state_dir / _SNAPSHOTS_SUBDIR
        if not _is_inside(snapshot_dir, snapshots_root):
            return None, (
                f"Computed snapshot dir {snapshot_dir} is outside state_dir/snapshots; "
                "refusing to resume"
            )
        try:
            ops, op_shas = self._resume_operations(envelope, state_path, snapshot_dir)
        except (FileNotFoundError, OSError, ValueError) as error:
            return None, f"resume validation failed for {state_path.name}: {error}"
        return (
            StoredUpload(
                ops=ops,
                message=str(envelope["message"]),
                state_path=state_path,
                snapshot_dir=snapshot_dir,
                op_shas=op_shas,
            ),
            None,
        )

    @staticmethod
    def _resume_operations(
        envelope: dict[str, Any],
        state_path: Path,
        snapshot_dir: Path,
    ) -> tuple[UploadOps, tuple[str | None, ...]]:
        operations: UploadOps = []
        op_shas: list[str | None] = []
        for entry in envelope.get("ops", []):
            operation, sha = UploadStateStore._resume_operation(entry, state_path, snapshot_dir)
            operations.append(operation)
            op_shas.append(sha)
        return operations, tuple(op_shas)

    @staticmethod
    def _resume_operation(
        entry: dict[str, Any],
        state_path: Path,
        snapshot_dir: Path,
    ) -> tuple[PublicationOp, str | None]:
        action = _required_string(entry, "action")
        path_in_repo = _required_string(entry, "path_in_repo")
        local_path_str = entry.get("local_path")
        snapshot_path_str = entry.get("snapshot_path")
        local_path = Path(local_path_str) if local_path_str else None
        snapshot_path = Path(snapshot_path_str) if snapshot_path_str else None
        if action == "add":
            UploadStateStore._validate_resume_snapshot(snapshot_path, state_path, snapshot_dir)
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
        if snapshot_path is None:
            raise ValueError(f"Add op in {state_path} is missing snapshot_path")
        if not _is_inside(snapshot_path, snapshot_dir):
            raise ValueError(
                f"Add op snapshot {snapshot_path} is outside {snapshot_dir}; refusing to upload"
            )

    def _upgrade_legacy_envelope(
        self,
        legacy_path: Path,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any], Path]:
        sequence = self._allocate_sequence()
        snapshot_dir = _snapshot_dir_for_sequence(self.state_dir, sequence)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _new_ops, op_entries = self._upgrade_legacy_operations(payload, snapshot_dir)
        new_envelope = {
            "contract_version": QUEUE_CONTRACT_VERSION,
            "sequence": sequence,
            "message": str(payload.get("message", "")),
            "ops": op_entries,
        }
        upgraded_path = self.state_dir / f"{sequence:06d}.json"
        atomic_write_text(upgraded_path, json.dumps(new_envelope, indent=2, sort_keys=True) + "\n")
        legacy_path.unlink(missing_ok=True)
        return sequence, new_envelope, snapshot_dir

    def _upgrade_legacy_operations(
        self,
        payload: dict[str, Any],
        snapshot_dir: Path,
    ) -> tuple[UploadOps, list[dict[str, Any]]]:
        new_ops: UploadOps = []
        op_entries: list[dict[str, Any]] = []
        for op_index, entry in enumerate(payload.get("ops", [])):
            operation, op_entry = self._upgrade_legacy_operation(entry, op_index, snapshot_dir)
            new_ops.append(operation)
            op_entries.append(op_entry)
        return new_ops, op_entries

    def _upgrade_legacy_operation(
        self,
        entry: dict[str, Any],
        op_index: int,
        snapshot_dir: Path,
    ) -> tuple[PublicationOp, dict[str, Any]]:
        action = _required_string(entry, "action")
        path_in_repo = _required_string(entry, "path_in_repo")
        local_path_str = entry.get("local_path")
        local_path = Path(local_path_str) if local_path_str else None
        op_entry: dict[str, Any] = {
            "action": action,
            "path_in_repo": path_in_repo,
            "local_path": local_path_str,
        }
        if action != "add":
            return self._upgrade_legacy_non_add_operation(
                action,
                path_in_repo,
                local_path,
                op_entry,
            )
        return self._upgrade_legacy_add_operation(
            action,
            path_in_repo,
            local_path,
            op_index,
            snapshot_dir,
            op_entry,
        )

    @staticmethod
    def _upgrade_legacy_non_add_operation(
        action: str,
        path_in_repo: str,
        local_path: Path | None,
        op_entry: dict[str, Any],
    ) -> tuple[PublicationOp, dict[str, Any]]:
        if local_path is not None:
            raise ValueError(
                f"PublicationOp(action='delete', path_in_repo={path_in_repo!r}) "
                "must not carry a local_path"
            )
        return PublicationOp(action=action, path_in_repo=path_in_repo), op_entry

    def _upgrade_legacy_add_operation(
        self,
        action: str,
        path_in_repo: str,
        local_path: Path | None,
        op_index: int,
        snapshot_dir: Path,
        op_entry: dict[str, Any],
    ) -> tuple[PublicationOp, dict[str, Any]]:
        if local_path is None:
            raise ValueError(
                f"PublicationOp(action='add', path_in_repo={path_in_repo!r}) requires a local_path"
            )
        if not local_path.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing canonical file: {local_path}")
        op_dir = snapshot_dir / f"{op_index:03d}"
        op_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = op_dir / local_path.name
        self._copy_file(local_path, snapshot_path)
        op_entry["snapshot_path"] = str(snapshot_path)
        op_entry["sha256"] = self._hash_file(snapshot_path)
        return PublicationOp(action, path_in_repo, local_path, snapshot_path), op_entry

    def delete(self, state_path: Path, snapshot_dir: Path | None) -> None:
        """Delete a completed envelope and its queue-owned snapshot."""
        state_path.unlink(missing_ok=True)
        if snapshot_dir is None:
            return
        snapshots_root = self.state_dir / _SNAPSHOTS_SUBDIR
        if not _is_inside(snapshot_dir, snapshots_root):
            raise ValueError(
                f"snapshot_dir {snapshot_dir} is outside {snapshots_root}; refusing to delete"
            )
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    def snapshot_mismatch(
        self,
        message: str,
        op: PublicationOp,
        recorded_sha: str | None,
    ) -> str | None:
        """Return a failure message for a changed immutable snapshot."""
        snapshot = _snapshot_path(op)
        if snapshot is None or not recorded_sha:
            return None
        actual = self._hash_file(snapshot)
        if actual == recorded_sha:
            return None
        return f"{message}: SHA mismatch for {snapshot} (recorded {recorded_sha}, actual {actual})"


def _snapshot_path(op: PublicationOp) -> Path | None:
    if op.action != "add":
        return None
    return op.snapshot_path or op.local_path


__all__ = [
    "QUEUE_CONTRACT_VERSION",
    "ResumeResult",
    "StoredUpload",
    "UploadStateStore",
]
