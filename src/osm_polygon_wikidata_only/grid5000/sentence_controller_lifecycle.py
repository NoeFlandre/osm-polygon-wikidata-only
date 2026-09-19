"""Run lifecycle, publication, cleanup, and interrupt handling."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from .sentence_controller_policy import (
    ACTIVE_STATES,
    ControllerRunError,
    baseline_hashes,
    batch_stems,
    publication_message,
    timestamp,
)


class SentenceControllerLifecycleMixin:
    """Coordinate batches and keep remote cleanup resumable and safe."""

    def run(self) -> dict[str, Any]:
        """Process batches serially until all finalized V2 stems are published."""
        ledger = self.initialize()
        try:
            for batch in ledger["batches"]:
                self._current_batch = batch
                self._process_batch(batch)
            if not all(batch["state"] == "published" for batch in ledger["batches"]):
                raise ControllerRunError("Sentence run is incomplete and remains resumable")
            self._finalize_run()
            return ledger
        except KeyboardInterrupt:
            self._handle_interrupt()
            raise

    def _publish_batch(self, batch: dict[str, Any]) -> None:
        try:
            self._assert_baseline()
        except Exception as error:
            batch["error"] = type(error).__name__
            self._write_ledger()
            raise ControllerRunError(f"Protected publication baseline changed: {error}") from error
        message = publication_message(batch)
        try:
            commit = self.publisher.publish_sentence_batch(
                self.data_root.processed_v2,
                batch_stems(batch),
                message,
            )
            self.publisher.verify_sentence_batch(
                self.data_root.processed_v2,
                batch_stems(batch),
            )
        except Exception:
            batch["state"] = "ready_to_publish"
            batch["error"] = "publisher_failure"
            self._write_ledger()
            raise
        batch["state"] = "published"
        batch["hf_commit"] = commit
        batch["published_at"] = timestamp()
        batch["error"] = None
        self._write_ledger()
        self._cleanup_remote_job(batch)

    def _assert_baseline(self) -> None:
        readme_hash, map_hash = baseline_hashes(self.data_root)
        assert self._ledger is not None
        if readme_hash != self._ledger["baseline_readme_sha256"]:
            raise ValueError("README baseline hash changed")
        if map_hash != self._ledger["baseline_map_sha256"]:
            raise ValueError("comparison-map baseline hash changed")

    def _cleanup_remote_job(self, batch: dict[str, Any]) -> None:
        remote_job_root = batch.get("remote_job_root")
        expected_prefix = f"{self.remote_run_root}/jobs/"
        if not isinstance(remote_job_root, str) or not remote_job_root.startswith(expected_prefix):
            raise ControllerRunError("Refusing cleanup outside the current Grid5000 run namespace")
        if batch.get("remote_cleaned"):
            return
        self.transport.remove_tree(remote_job_root)
        batch["remote_cleaned"] = True
        self._write_ledger()

    def _policy_check(self) -> None:
        self._run_frontend(("usagepolicycheck", "-t"))

    def _run_frontend(
        self, args: Sequence[str], *, allow_failure: bool = False
    ) -> subprocess.CompletedProcess[str]:
        result = self.transport.run_frontend(args)
        if result.returncode != 0 and not allow_failure:
            detail = (result.stderr or "").strip() or "frontend command failed"
            raise ControllerRunError(f"Grid5000 frontend command failed: {args[0]}: {detail}")
        return result

    def _finalize_run(self) -> None:
        assert self._ledger is not None
        if self._ledger.get("cleanup_state") == "complete":
            return
        self._policy_check()
        self._ledger["cleanup_state"] = "pending"
        self._write_ledger()
        self.transport.remove_tree(self.remote_run_root)
        self._ledger["cleanup_state"] = "complete"
        self._write_ledger()

    def _handle_interrupt(self) -> None:
        batch = self._current_batch
        if batch is None or batch.get("state") not in ACTIVE_STATES:
            self._write_ledger()
            return
        job_id = batch.get("oar_job_id")
        if isinstance(job_id, str) and job_id:
            with suppress(Exception):
                self._run_frontend(("oardel", job_id), allow_failure=True)
        batch["state"] = "cancelled"
        batch["error"] = "cancelled_by_interrupt"
        self._write_ledger()

__all__ = ["SentenceControllerLifecycleMixin"]
