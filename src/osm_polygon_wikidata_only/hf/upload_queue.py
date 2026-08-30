"""Bounded background publication queue.

The queue owns worker lifecycle and upload retry behavior. Durable envelope,
snapshot, and resume semantics live in :mod:`._upload_state` so they can be
validated independently of threading and network callbacks.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_wikidata_only.hf._upload_retry import _run_upload_attempts
from osm_polygon_wikidata_only.hf._upload_state import (
    QUEUE_CONTRACT_VERSION,
    UploadStateStore,
    _independent_copy,
    _sha256_file,
)
from osm_polygon_wikidata_only.hf._upload_state import _read_envelope as _state_read_envelope
from osm_polygon_wikidata_only.hf._upload_state import (
    _remove_failed_upgrade as _remove_failed_upgrade_artifacts,
)
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp

LOGGER = logging.getLogger(__name__)

UploadOps = list[PublicationOp]
UploadOperation = Callable[[UploadOps, str], None]


def _read_envelope(path: Path) -> dict[str, object] | None:
    """Compatibility wrapper for current-envelope parsing."""
    return _state_read_envelope(path)


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
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self._upload = upload
        self._attempts = attempts
        self._jobs: queue.Queue[_UploadJob | None] = queue.Queue(max_pending)
        self._state_dir = state_dir
        self._state_store = (
            UploadStateStore(
                state_dir,
                copy_file=_independent_copy,
                hash_file=_sha256_file,
            )
            if state_dir is not None
            else None
        )
        self._next_sequence = (
            self._state_store.next_sequence if self._state_store is not None else 1
        )
        self._failures: list[str] = []
        self._thread = threading.Thread(target=self._worker, name="hf-upload", daemon=False)
        self._closed = False
        self._thread.start()

    def submit(self, ops: UploadOps, message: str) -> None:
        """Persist and enqueue one upload job."""
        self._ensure_open()
        job = self._build_job(ops, message)
        self._jobs.put(job)
        LOGGER.info("Queued background upload: %s", message)

    def _build_job(self, ops: UploadOps, message: str) -> _UploadJob:
        if self._state_store is None:
            return _UploadJob(
                list(ops),
                message,
                None,
                None,
                op_shas=tuple(None for _ in ops),
            )
        stored = self._state_store.persist(ops, message)
        self._next_sequence = self._state_store.next_sequence
        return _UploadJob(
            stored.ops,
            stored.message,
            stored.state_path,
            stored.snapshot_dir,
            op_shas=stored.op_shas,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("upload queue is closed")

    def resume_pending(self) -> int:
        """Resume all pending envelopes in sequence order."""
        if self._state_store is None:
            return 0
        result = self._state_store.resume_pending()
        self._next_sequence = self._state_store.next_sequence
        for detail in result.failures:
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
        for stored in result.uploads:
            self._jobs.put(
                _UploadJob(
                    stored.ops,
                    stored.message,
                    stored.state_path,
                    stored.snapshot_dir,
                    op_shas=stored.op_shas,
                )
            )
        return result.discovered_count

    @staticmethod
    def _remove_failed_upgrade(state_dir: Path, sequence: int, snapshot_dir: Path) -> None:
        """Compatibility wrapper for cleanup of a failed legacy upgrade."""
        _remove_failed_upgrade_artifacts(state_dir, sequence, snapshot_dir)

    def close_and_wait(self) -> list[str]:
        if not self._closed:
            self._closed = True
            self._jobs.put(None)
        self._thread.join()
        return list(self._failures)

    def upload_synchronously(self, ops: list[PublicationOp], commit_message: str) -> None:
        """Run one final upload after the regional queue has drained."""
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
            if self._state_store is None:
                raise RuntimeError("durable upload job has no state store")
            self._state_store.delete(job.state_path, job.snapshot_dir)
        LOGGER.info("Background upload complete: %s", job.message)

    def _verify_job_snapshot(self, job: _UploadJob) -> bool:
        """Verify all recorded snapshot hashes before uploading."""
        if self._state_store is None:
            return True
        for op, recorded_sha in zip(job.ops, job.op_shas, strict=True):
            detail = self._state_store.snapshot_mismatch(job.message, op, recorded_sha)
            if detail is None:
                continue
            LOGGER.error("Background upload failed: %s", detail)
            self._failures.append(detail)
            return False
        return True

    def _upload_with_retries(self, job: _UploadJob) -> None:
        """Attempt one upload up to the configured retry count."""
        _run_upload_attempts(self._upload, job.ops, job.message, attempts=self._attempts)


__all__ = ["QUEUE_CONTRACT_VERSION", "BackgroundUploadQueue", "UploadOperation", "UploadOps"]
