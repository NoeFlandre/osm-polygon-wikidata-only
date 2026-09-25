"""Receipt validation and local artifact import for sentence batches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_copy_file
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION
from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

from .sentence_controller_context import SentenceControllerContext
from .sentence_controller_policy import (
    BatchDict,
    ControllerRunError,
    batch_stems,
    load_json_mapping,
    normalized_receipt_value,
    read_json_mapping,
    receipt_artifacts,
    receipt_digest,
    source_projects,
    verified_incoming_artifact,
)
from .sentence_protocol import (
    import_checkpoint_tree,
    sentence_source_paths,
    validate_manifest_extension,
    validate_sentence_output,
)


class SentenceControllerImportMixin(SentenceControllerContext):
    """Verify remote receipts before installing data and checkpoints."""

    def _read_valid_receipt(
        self,
        batch: BatchDict,
        received: Path,
        received_data: DataRoot,
    ) -> dict[str, Any]:
        try:
            receipt = read_json_mapping(received / "receipt.json")
            self._validate_receipt(batch, receipt)
            return receipt
        except ControllerRunError as error:
            self._import_partial(batch, received_data)
            batch["state"] = "failed"
            batch["error"] = (
                "missing_receipt"
                if not (received / "receipt.json").is_file()
                else "invalid_receipt"
            )
            self._write_ledger()
            self._cleanup_remote_job(batch)
            raise ControllerRunError(
                f"Grid5000 batch {batch['index']} produced an invalid receipt; retryable"
            ) from error

    def _mark_batch_failed(
        self,
        batch: BatchDict,
        received_data: DataRoot,
        receipt: Mapping[str, object],
    ) -> None:
        self._import_partial(batch, received_data)
        batch["state"] = "failed"
        batch["error"] = str(receipt.get("error_type") or "remote_job_failed")
        self._write_ledger()

    def _validate_receipt(self, batch: Mapping[str, object], receipt: Mapping[str, object]) -> None:
        expected = {
            "job_id": str(batch["oar_job_id"]),
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "stems": batch_stems(batch),
        }
        for key, value in expected.items():
            actual = normalized_receipt_value(key, receipt.get(key))
            if actual != value:
                raise ControllerRunError(f"Grid5000 receipt mismatch: {key}")

    def _import_success(
        self,
        batch: Mapping[str, object],
        received_data: DataRoot,
        receipt: Mapping[str, object],
    ) -> None:
        artifacts = receipt_artifacts(receipt)
        manifest_relative = (Path("processed_v2") / SENTENCE_MANIFEST_RELATIVE_PATH).as_posix()
        manifest_path = verified_incoming_artifact(received_data.path, artifacts, manifest_relative)
        incoming_payload = read_json_mapping(manifest_path)
        local_manifest_path = self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        local_payload = load_json_mapping(local_manifest_path)
        if local_payload is None:
            local_payload = {**incoming_payload, "regions": []}
        validate_manifest_extension(
            local_payload,
            incoming_payload,
            selected_stems=batch_stems(batch),
        )
        files_to_copy: list[tuple[Path, Path]] = [(manifest_path, local_manifest_path)]
        for stem in batch_stems(batch):
            for project in source_projects(self.data_root.processed_v2, stem):
                relative = f"processed_v2/{project}/sentences/{stem}.parquet"
                incoming = verified_incoming_artifact(received_data.path, artifacts, relative)
                validate_sentence_output(
                    incoming,
                    expected_sha256=receipt_digest(artifacts, relative).sha256,
                )
                files_to_copy.append(
                    (
                        incoming,
                        self.data_root.processed_v2 / f"{project}/sentences/{stem}.parquet",
                    )
                )
        for source in files_to_copy:
            atomic_copy_file(*source)
        self._import_checkpoints(batch, received_data)

    def _import_partial(self, batch: Mapping[str, object], received_data: DataRoot) -> None:
        self._import_checkpoints(batch, received_data)

    def _import_checkpoints(self, batch: Mapping[str, object], received_data: DataRoot) -> None:
        for stem in batch_stems(batch):
            for project in source_projects(self.data_root.processed_v2, stem):
                self._import_checkpoint(stem, project, received_data)

    def _import_checkpoint(self, stem: str, project: str, received_data: DataRoot) -> None:
        incoming = received_data.v2_cache / "sentence-checkpoints" / stem / project
        if not incoming.is_dir():
            return
        source = sentence_source_paths(self.data_root.processed_v2, stem)
        source_path = next(path for path in source if project in path.parts)
        identity = {
            "contract_version": "v2-sentence-checkpoints-v1",
            "stem": stem,
            "project": project,
            "input_fingerprint": sha256_file(source_path),
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "batch_size": self.limits.batch_size,
        }
        import_checkpoint_tree(
            incoming,
            self.data_root.v2_cache / "sentence-checkpoints" / stem / project,
            expected_identity=identity,
        )


__all__ = ["SentenceControllerImportMixin"]
