"""Ledger lifecycle for the Grid5000 sentence controller."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_json
from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION
from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

from .sentence_controller_policy import (
    SEGMENTER_VERSION,
    ControllerRunError,
    baseline_hashes,
    is_safe_run_id,
    is_source_commit_migration_safe,
    load_json_mapping,
    new_run_id,
    read_json_mapping,
    timestamp,
    validate_ledger_baselines,
)
from .sentence_protocol import GRID5000_SENTENCE_CONTRACT_VERSION, plan_sentence_batches

_LEDGER_FILENAME = "grid5000_sentence_run.json"


class SentenceControllerLedgerMixin:
    """Create, validate, update, and persist the durable run ledger."""

    def initialize(self) -> dict[str, Any]:
        """Load and validate the ledger, or persist a new deterministic plan."""
        if self._ledger is not None:
            return self._ledger
        ledger = (
            self._load_existing_ledger() if self.ledger_path.is_file() else self._create_ledger()
        )
        self._ledger = ledger
        return ledger

    def _load_existing_ledger(self) -> dict[str, Any]:
        ledger = read_json_mapping(self.ledger_path)
        stored_run_id = ledger.get("run_id")
        if not isinstance(stored_run_id, str):
            raise ControllerRunError("Sentence ledger has no valid run_id")
        if self.run_id is not None and self.run_id != stored_run_id:
            raise ControllerRunError("Requested run_id does not match the existing ledger")
        self.run_id = stored_run_id
        self._refresh_resumable_source_commit(ledger)
        self._validate_immutable_ledger(ledger)
        return ledger

    def _create_ledger(self) -> dict[str, Any]:
        if self.run_id is None:
            self.run_id = new_run_id()
        if not is_safe_run_id(self.run_id):
            raise ControllerRunError(f"Unsafe Grid5000 run_id: {self.run_id!r}")
        ledger = self._new_ledger()
        self._write_ledger(ledger)
        return ledger

    def _refresh_resumable_source_commit(self, ledger: dict[str, Any]) -> None:
        stored_commit = ledger.get("source_commit")
        if stored_commit == self.source_commit or not is_source_commit_migration_safe(ledger):
            return
        if not isinstance(stored_commit, str):
            return
        updates = ledger.get("source_commit_updates", [])
        if not isinstance(updates, list):
            raise ControllerRunError("Sentence ledger source_commit_updates must be a list")
        ledger["source_commit"] = self.source_commit
        ledger["source_commit_updates"] = [
            *updates,
            {
                "from": stored_commit,
                "to": self.source_commit,
                "at": timestamp(),
                "reason": "resumable_unpublished_run",
            },
        ]
        self._write_ledger(ledger)

    def _new_ledger(self) -> dict[str, Any]:
        sentence_manifest = load_json_mapping(
            self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        )
        batches = plan_sentence_batches(
            self.data_root.processed_v2,
            sentence_manifest,
            max_stems=self.limits.max_stems,
            max_input_bytes=self.limits.max_input_bytes,
        )
        readme_hash, map_hash = baseline_hashes(self.data_root)
        return {
            "contract_version": GRID5000_SENTENCE_CONTRACT_VERSION,
            "run_id": self.run_id,
            "repo_id": self.repo_id,
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "segmenter_version": SEGMENTER_VERSION,
            "site": self.site,
            "queue": self.queue,
            "gpu_model": self.gpu_model,
            "baseline_readme_sha256": readme_hash,
            "baseline_map_sha256": map_hash,
            "limits": self.limits.as_payload(),
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "cleanup_state": "pending",
            "batches": [self._new_batch_record(batch) for batch in batches],
        }

    def _new_batch_record(self, batch: Any) -> dict[str, Any]:
        return {
            "index": batch.index,
            "stems": list(batch.stems),
            "input_bytes": batch.input_bytes,
            "state": "planned",
            "attempt": 0,
            "oar_job_id": None,
            "remote_job_root": None,
            "hf_commit": None,
            "error": None,
        }

    def _validate_immutable_ledger(self, ledger: Mapping[str, object]) -> None:
        for key, value in self._immutable_ledger_fields().items():
            if ledger.get(key) != value:
                raise ControllerRunError(f"Sentence ledger immutable field changed: {key}")
        validate_ledger_baselines(ledger)

    def _immutable_ledger_fields(self) -> dict[str, object]:
        return {
            "contract_version": GRID5000_SENTENCE_CONTRACT_VERSION,
            "repo_id": self.repo_id,
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "segmenter_version": SEGMENTER_VERSION,
            "site": self.site,
            "queue": self.queue,
            "gpu_model": self.gpu_model,
            "limits": self.limits.as_payload(),
        }

    def _write_ledger(self, ledger: dict[str, Any] | None = None) -> None:
        if ledger is not None:
            self._ledger = ledger
        if self._ledger is None:
            raise ControllerRunError("Cannot write an uninitialized sentence ledger")
        self._ledger["updated_at"] = timestamp()
        atomic_write_json(self.ledger_path, self._ledger)


__all__ = ["SentenceControllerLedgerMixin"]
