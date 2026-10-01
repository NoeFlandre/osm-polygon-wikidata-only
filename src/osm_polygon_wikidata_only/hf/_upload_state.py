"""Durable state boundary for the background upload queue.

This module owns the on-disk upload contract: sequence allocation, immutable
snapshots, envelope serialization, legacy upgrades, and resume validation.
The queue itself only coordinates workers and upload callbacks.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.hf._upload_state_files import (
    QUEUE_CONTRACT_VERSION,
    SNAPSHOTS_SUBDIR,
    current_sequence,
    independent_copy,
    is_inside,
    is_legacy_envelope,
    next_sequence_from_state_dir,
    read_legacy_or_current_envelope,
    remove_failed_upgrade,
    required_string,
    snapshot_dir_for_sequence,
    write_highwater,
)
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file as _sha256_file

UploadOps = list[PublicationOp]
CopyFile = Callable[[Path, Path], None]
HashFile = Callable[[Path], str]


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


class UploadStateStore:
    """Own the durable state contract used by ``BackgroundUploadQueue``."""

    def __init__(
        self,
        state_dir: Path,
        *,
        copy_file: CopyFile = independent_copy,
        hash_file: HashFile = _sha256_file,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._copy_file = copy_file
        self._hash_file = hash_file
        self._next_sequence_lock = threading.Lock()
        self._next_sequence = next_sequence_from_state_dir(state_dir)

    @property
    def next_sequence(self) -> int:
        """Return the next sequence that will be allocated."""
        return self._next_sequence

    def _allocate_sequence(self) -> int:
        with self._next_sequence_lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            write_highwater(self.state_dir, sequence)
            return sequence

    def persist(self, ops: UploadOps, message: str) -> StoredUpload:
        """Snapshot and durably persist one upload submission."""
        sequence = self._allocate_sequence()
        state_path = self.state_dir / f"{sequence:06d}.json"
        snapshot_dir = snapshot_dir_for_sequence(self.state_dir, sequence)
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
        if is_inside(snapshot_dir, self.state_dir / SNAPSHOTS_SUBDIR):
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
        snapshot_dir = snapshot_dir_for_sequence(self.state_dir, sequence)
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
        payload = read_legacy_or_current_envelope(path)
        if payload is None:
            raise ValueError(f"Malformed envelope: {path} -- refusing to resume")
        if is_legacy_envelope(payload):
            return "legacy", payload, None
        if payload.get("contract_version") != QUEUE_CONTRACT_VERSION:
            raise ValueError(f"Unknown contract_version in {path}; refusing to resume")
        sequence = current_sequence(payload, path)
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
                remove_failed_upgrade(self.state_dir, sequence, snapshot_dir)
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
            (sequence, path, payload, snapshot_dir_for_sequence(self.state_dir, sequence))
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
        snapshots_root = self.state_dir / SNAPSHOTS_SUBDIR
        if not is_inside(snapshot_dir, snapshots_root):
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
        action = required_string(entry, "action")
        path_in_repo = required_string(entry, "path_in_repo")
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
        if not is_inside(snapshot_path, snapshot_dir):
            raise ValueError(
                f"Add op snapshot {snapshot_path} is outside {snapshot_dir}; refusing to upload"
            )

    def _upgrade_legacy_envelope(
        self,
        legacy_path: Path,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any], Path]:
        sequence = self._allocate_sequence()
        snapshot_dir = snapshot_dir_for_sequence(self.state_dir, sequence)
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
        action = required_string(entry, "action")
        path_in_repo = required_string(entry, "path_in_repo")
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
        snapshots_root = self.state_dir / SNAPSHOTS_SUBDIR
        if not is_inside(snapshot_dir, snapshots_root):
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
