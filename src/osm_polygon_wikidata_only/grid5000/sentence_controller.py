"""Resumable local controller for Grid5000 sentence-splitting jobs.

The imported private helpers intentionally remain available as compatibility
seams for callers and tests while their implementations live in focused
modules.
"""

# ruff: noqa: F401

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token, upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_copy_file, atomic_write_json
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.run_lock import exclusive_run_lock
from osm_polygon_wikidata_only.v2.config import (
    V2_ADDED_WIKIPEDIA_TAG_MAP_PATH,
    V2_REPO_ID,
)
from osm_polygon_wikidata_only.v2.publication import sentence_publication_ops
from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION
from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

from .sentence_controller_policy import (
    ACTIVE_STATES as _ACTIVE_STATES,
)
from .sentence_controller_policy import (
    EXOTIC_GRID5000_GPU_MODELS as _EXOTIC_GRID5000_GPU_MODELS,
)
from .sentence_controller_policy import (
    GRID5000_UV_VERSION as _GRID5000_UV_VERSION,
)
from .sentence_controller_policy import (
    REMOTE_NAMESPACE as _REMOTE_NAMESPACE,
)
from .sentence_controller_policy import (
    RETRYABLE_ARTIFACT_FAILURES as _RETRYABLE_ARTIFACT_FAILURES,
)
from .sentence_controller_policy import (
    SEGMENTER_VERSION as _SEGMENTER_VERSION,
)
from .sentence_controller_policy import (
    SUCCESS_STATES as _SUCCESS_STATES,
)
from .sentence_controller_policy import (
    TERMINAL_STATES as _TERMINAL_STATES,
)
from .sentence_controller_policy import (
    ControllerLimits,
    ControllerRunError,
)
from .sentence_controller_policy import (
    baseline_hashes as _baseline_hashes,
)
from .sentence_controller_policy import (
    batch_stems as _batch_stems,
)
from .sentence_controller_policy import (
    copy_required as _copy_required,
)
from .sentence_controller_policy import (
    git_source_commit as _git_source_commit_impl,
)
from .sentence_controller_policy import (
    infer_job_state as _infer_job_state,
)
from .sentence_controller_policy import (
    is_safe_run_id as _is_safe_run_id,
)
from .sentence_controller_policy import (
    is_source_commit_migration_safe as _is_source_commit_migration_safe,
)
from .sentence_controller_policy import (
    load_json_mapping as _load_json_mapping,
)
from .sentence_controller_policy import (
    new_run_id as _new_run_id,
)
from .sentence_controller_policy import (
    normalized_receipt_value as _normalized_receipt_value,
)
from .sentence_controller_policy import (
    parse_job_id as _parse_job_id,
)
from .sentence_controller_policy import (
    parse_job_status as _parse_job_status,
)
from .sentence_controller_policy import (
    publication_message as _publication_message,
)
from .sentence_controller_policy import (
    read_json_mapping as _read_json_mapping,
)
from .sentence_controller_policy import (
    receipt_artifact as _receipt_artifact,
)
from .sentence_controller_policy import (
    receipt_artifacts as _receipt_artifacts,
)
from .sentence_controller_policy import (
    receipt_digest as _receipt_digest,
)
from .sentence_controller_policy import (
    remote_batch_succeeded as _remote_batch_succeeded,
)
from .sentence_controller_policy import (
    remote_job_root as _remote_job_root,
)
from .sentence_controller_policy import (
    source_commit_batch_is_safe as _source_commit_batch_is_safe,
)
from .sentence_controller_policy import (
    source_projects as _source_projects,
)
from .sentence_controller_policy import (
    timestamp as _timestamp,
)
from .sentence_controller_policy import (
    validate_ledger_baselines as _validate_ledger_baselines,
)
from .sentence_controller_policy import (
    verified_incoming_artifact as _verified_incoming_artifact,
)
from .sentence_protocol import (
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_STEMS,
    DEFAULT_WALLTIME,
    GRID5000_SENTENCE_CONTRACT_VERSION,
    FileDigest,
    import_checkpoint_tree,
    plan_sentence_batches,
    sentence_source_paths,
    validate_manifest_extension,
    validate_sentence_output,
)
from .sentence_publication import (
    HfHubSentencePublisher as _HfHubSentencePublisher,
)
from .sentence_publication import (
    HubPublisher,
)
from .sentence_publication import (
    download_hf_file as _download_hf_file_impl,
)
from .sentence_publication import (
    expected_sentence_files as _expected_sentence_files,
)
from .sentence_publication import (
    missing_remote_files as _missing_remote_files,
)
from .sentence_publication import (
    validate_expected_sentence_files as _validate_expected_sentence_files,
)
from .sentence_publication import (
    verify_expected_sentence_files as _verify_expected_sentence_files,
)
from .sentence_publication import (
    verify_known_sentence_file as _verify_known_sentence_file,
)
from .sentence_publication import (
    verify_sentence_file as _verify_sentence_file,
)
from .sentence_transport import (
    Grid5000Transport,
)
from .sentence_transport import (
    SubprocessGrid5000Transport as _SubprocessGrid5000Transport,
)
from .sentence_transport import (
    validate_remote_home as _validate_remote_home,
)
from .sentence_transport import (
    validated_remote_home as _validated_remote_home,
)

_LEDGER_FILENAME = "grid5000_sentence_run.json"
DEFAULT_GRID5000_QUEUE = "besteffort"
DEFAULT_GRID5000_GPU_MODEL = "A40"
_QUEUE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_GPU_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


class Grid5000SentenceController:
    """Coordinate one serial, resumable stream of Grid5000 GPU jobs."""

    def __init__(
        self,
        data_root: DataRoot,
        *,
        site: str = "grenoble",
        queue: str = DEFAULT_GRID5000_QUEUE,
        gpu_model: str = DEFAULT_GRID5000_GPU_MODEL,
        repo_id: str = V2_REPO_ID,
        transport: Grid5000Transport,
        publisher: HubPublisher,
        max_stems: int = DEFAULT_MAX_STEMS,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        batch_size: int = 256,
        inference_batch_size: int = 16,
        walltime: str = DEFAULT_WALLTIME,
        run_id: str | None = None,
        source_commit: str | None = None,
        repo_root: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = 10.0,
    ) -> None:
        self.data_root = data_root
        self.site = site
        if not _QUEUE_PATTERN.fullmatch(queue):
            raise ControllerRunError(f"Unsafe Grid5000 queue: {queue!r}")
        self.queue = queue
        if not _GPU_MODEL_PATTERN.fullmatch(gpu_model):
            raise ControllerRunError(f"Unsafe Grid5000 GPU model: {gpu_model!r}")
        self.gpu_model = gpu_model
        self.repo_id = repo_id
        self.transport = transport
        self.publisher = publisher
        self.run_id = run_id
        self.repo_root = Path(repo_root or Path.cwd())
        self.source_commit = source_commit or _git_source_commit(self.repo_root)
        self.limits = ControllerLimits(
            max_stems=max_stems,
            max_input_bytes=max_input_bytes,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            walltime=walltime,
        )
        self._sleep = sleep
        self.poll_interval_s = poll_interval_s
        self._ledger: dict[str, Any] | None = None
        self._current_batch: dict[str, Any] | None = None

    @property
    def ledger_path(self) -> Path:
        """Return the one durable ledger used by this data root."""
        return self.data_root.cache / _LEDGER_FILENAME

    @property
    def remote_run_root(self) -> str:
        """Return the fixed, run-owned remote namespace."""
        if self.run_id is None:
            raise ControllerRunError("Controller run has not been initialized")
        return f"{_REMOTE_NAMESPACE}/{self.run_id}"

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
        ledger = _read_json_mapping(self.ledger_path)
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
            self.run_id = _new_run_id()
        if not _is_safe_run_id(self.run_id):
            raise ControllerRunError(f"Unsafe Grid5000 run_id: {self.run_id!r}")
        ledger = self._new_ledger()
        self._write_ledger(ledger)
        return ledger

    def _refresh_resumable_source_commit(self, ledger: dict[str, Any]) -> None:
        stored_commit = ledger.get("source_commit")
        if stored_commit == self.source_commit or not _is_source_commit_migration_safe(ledger):
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
                "at": _timestamp(),
                "reason": "resumable_unpublished_run",
            },
        ]
        self._write_ledger(ledger)

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

    def _process_batch(self, batch: dict[str, Any]) -> None:
        state = str(batch["state"])
        if state == "published":
            return
        if state == "ready_to_publish":
            self._publish_batch(batch)
            return
        if state in _ACTIVE_STATES:
            self._reconcile_batch(batch)
        elif state in {"planned", "failed"}:
            self._submit_batch(batch)
            self._reconcile_batch(batch)
        else:
            raise ControllerRunError(f"Batch {batch['index']} is in unsupported state {state!r}")
        self._publish_if_ready(batch)

    def _publish_if_ready(self, batch: dict[str, Any]) -> None:
        if batch["state"] == "ready_to_publish":
            self._publish_batch(batch)

    def _new_ledger(self) -> dict[str, Any]:
        sentence_manifest = _load_json_mapping(
            self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        )
        batches = plan_sentence_batches(
            self.data_root.processed_v2,
            sentence_manifest,
            max_stems=self.limits.max_stems,
            max_input_bytes=self.limits.max_input_bytes,
        )
        readme_hash, map_hash = _baseline_hashes(self.data_root)
        return {
            "contract_version": GRID5000_SENTENCE_CONTRACT_VERSION,
            "run_id": self.run_id,
            "repo_id": self.repo_id,
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "segmenter_version": _SEGMENTER_VERSION,
            "site": self.site,
            "queue": self.queue,
            "gpu_model": self.gpu_model,
            "baseline_readme_sha256": readme_hash,
            "baseline_map_sha256": map_hash,
            "limits": self.limits.as_payload(),
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
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
        _validate_ledger_baselines(ledger)

    def _immutable_ledger_fields(self) -> dict[str, object]:
        return {
            "contract_version": GRID5000_SENTENCE_CONTRACT_VERSION,
            "repo_id": self.repo_id,
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "segmenter_version": _SEGMENTER_VERSION,
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
        self._ledger["updated_at"] = _timestamp()
        atomic_write_json(self.ledger_path, self._ledger)

    def _submit_batch(self, batch: dict[str, Any]) -> None:
        batch["attempt"] = int(batch.get("attempt", 0)) + 1
        batch["state"] = "submitted"
        batch["oar_job_id"] = None
        batch["remote_job_root"] = _remote_job_root(
            self.remote_run_root, int(batch["index"]), int(batch["attempt"])
        )
        batch["remote_cleaned"] = False
        batch["error"] = None
        self._write_ledger()
        try:
            self._ensure_remote_namespace()
            with tempfile.TemporaryDirectory(
                prefix=f"grid5000-batch-{batch['index']}-", dir=self.data_root.cache
            ) as temporary:
                staging = Path(temporary)
                self._stage_batch(staging, batch)
                self.transport.upload_tree(staging, str(batch["remote_job_root"]))
            self._policy_check()
            job_type_args = (
                ("-t", "exotic") if self.gpu_model in _EXOTIC_GRID5000_GPU_MODELS else ()
            )
            submit_command = (
                "oarsub",
                "-q",
                self.queue,
                *job_type_args,
                "-p",
                f"gpu_model='{self.gpu_model}'",
                "-l",
                f"host=1/gpu=1,walltime={self.limits.walltime}",
                self._remote_job_command(batch),
            )
            submitted = self._run_frontend(submit_command)
            job_id = _parse_job_id(submitted.stdout or "")
            batch["oar_job_id"] = job_id
            batch["state"] = "running"
            self._write_ledger()
            self._policy_check()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            batch["state"] = "failed"
            batch["error"] = type(error).__name__
            self._write_ledger()
            raise ControllerRunError(f"Grid5000 batch submission failed: {error}") from error

    def _ensure_remote_namespace(self) -> None:
        self._run_frontend(
            (
                "mkdir",
                "-p",
                f"{self.remote_run_root}/model-cache",
                f"{self.remote_run_root}/uv-cache",
                f"{self.remote_run_root}/jobs",
            )
        )

    def _stage_batch(self, staging: Path, batch: Mapping[str, object]) -> None:
        code = staging / "code"
        data = staging / "result" / "data"
        code.mkdir(parents=True)
        _copy_required(self.repo_root / "pyproject.toml", code / "pyproject.toml")
        _copy_required(self.repo_root / "uv.lock", code / "uv.lock")
        _copy_required(self.repo_root / "README.md", code / "README.md")
        _copy_required(self.repo_root / "LICENSE", code / "LICENSE")
        for asset in ("dataset_hero.png", "dataset_hero_v2.png"):
            _copy_required(self.repo_root / "assets" / asset, code / "assets" / asset)
        shutil.copytree(self.repo_root / "src", code / "src")
        (code / "scripts").mkdir()
        _copy_required(
            self.repo_root / "scripts/grid5000_sentence_job.py",
            code / "scripts/grid5000_sentence_job.py",
        )
        data_v2 = data / "processed_v2"
        data_v2.mkdir(parents=True)
        _copy_required(
            self.data_root.processed_v2 / "manifests/processed_pbfs.json",
            data_v2 / "manifests/processed_pbfs.json",
        )
        sentence_manifest = self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        if sentence_manifest.is_file():
            _copy_required(sentence_manifest, data_v2 / SENTENCE_MANIFEST_RELATIVE_PATH)
        for stem in _batch_stems(batch):
            for source in sentence_source_paths(self.data_root.processed_v2, stem):
                relative = source.relative_to(self.data_root.processed_v2)
                _copy_required(source, data_v2 / relative)
            self._stage_checkpoint_trees(data, stem)

    def _stage_checkpoint_trees(self, staged_data: Path, stem: str) -> None:
        for project in _source_projects(self.data_root.processed_v2, stem):
            source = self.data_root.v2_cache / "sentence-checkpoints" / stem / project
            if source.is_dir():
                shutil.copytree(
                    source,
                    staged_data / "cache/v2/sentence-checkpoints" / stem / project,
                )

    def _remote_job_command(self, batch: Mapping[str, object]) -> str:
        remote_job_root = str(batch["remote_job_root"])
        remote_data = f"{remote_job_root}/result/data"
        receipt = f"{remote_job_root}/result/receipt.json"
        model_cache = f"{self.remote_run_root}/model-cache"
        uv_cache = f"{self.remote_run_root}/uv-cache"
        uv_bootstrap = f"{self.remote_run_root}/uv-bootstrap"
        uv_bin = f"{uv_bootstrap}/bin/uv"
        uv_python = f"{uv_bootstrap}/bin/python"
        venv_lib = f"{remote_job_root}/code/.venv/lib"
        stems = " ".join(shlex.quote(stem) for stem in _batch_stems(batch))
        return (
            f'cd "{remote_job_root}/code" && '
            f'if [ ! -x "{uv_bin}" ]; then '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN python3 -m venv "{uv_bootstrap}" && '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "{uv_python}" -m pip install '
            f'--disable-pip-version-check --no-input "uv=={_GRID5000_UV_VERSION}"; '
            f"fi && "
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="{uv_cache}" '
            f'"{uv_bin}" sync --frozen --extra sentence-splitting-gpu --no-dev && '
            f'nvidia_libs="$(find "{venv_lib}" -path "*/site-packages/nvidia/*/lib" '
            '-type d -print | paste -sd: -)" && '
            'if [ -z "$nvidia_libs" ]; then echo "CUDA libraries were not installed" >&2; exit 1; fi && '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="{uv_cache}" '
            'LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" '
            f'"{uv_bin}" run --no-sync python scripts/grid5000_sentence_job.py '
            f'--data-root "{remote_data}" --model-cache "{model_cache}" '
            f"--source-commit {shlex.quote(self.source_commit)} "
            '--job-id "$OAR_JOB_ID" '
            f"--batch-size {self.limits.batch_size} "
            f"--inference-batch-size {self.limits.inference_batch_size} "
            f'--receipt "{receipt}" --stems {stems}'
        )

    def _reconcile_batch(self, batch: dict[str, Any]) -> None:
        job_id = batch.get("oar_job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ControllerRunError(
                f"Batch {batch['index']} is {batch['state']} without a recorded OAR job ID; refusing duplicate submission"
            )
        while True:
            result = self._run_frontend(("oarstat", "-s", "-j", job_id), allow_failure=True)
            state, exit_code = _parse_job_status(result)
            if state not in _TERMINAL_STATES:
                self._sleep(self.poll_interval_s)
                continue
            self._retrieve_batch(batch, state=state, exit_code=exit_code)
            return

    def _retrieve_batch(
        self,
        batch: dict[str, Any],
        *,
        state: str,
        exit_code: int | None,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=f"grid5000-receive-{batch['index']}-", dir=self.data_root.cache
        ) as temporary:
            received = Path(temporary)
            self.transport.download_tree(f"{batch['remote_job_root']}/result", received)
            received_data = DataRoot(received / "data")
            receipt = self._read_valid_receipt(batch, received, received_data)
            if _remote_batch_succeeded(state, exit_code, receipt):
                self._import_success(batch, received_data, receipt)
                batch["state"] = "ready_to_publish"
                self._write_ledger()
                return
            self._mark_batch_failed(batch, received_data, receipt)
        self._cleanup_remote_job(batch)
        raise ControllerRunError(f"Grid5000 batch {batch['index']} failed and is retryable")

    def _read_valid_receipt(
        self,
        batch: dict[str, Any],
        received: Path,
        received_data: DataRoot,
    ) -> dict[str, Any]:
        try:
            receipt = _read_json_mapping(received / "receipt.json")
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
        batch: dict[str, Any],
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
            "stems": _batch_stems(batch),
        }
        for key, value in expected.items():
            actual = _normalized_receipt_value(key, receipt.get(key))
            if actual != value:
                raise ControllerRunError(f"Grid5000 receipt mismatch: {key}")

    def _import_success(
        self,
        batch: Mapping[str, object],
        received_data: DataRoot,
        receipt: Mapping[str, object],
    ) -> None:
        artifacts = _receipt_artifacts(receipt)
        manifest_relative = (Path("processed_v2") / SENTENCE_MANIFEST_RELATIVE_PATH).as_posix()
        manifest_path = _verified_incoming_artifact(
            received_data.path, artifacts, manifest_relative
        )
        incoming_payload = _read_json_mapping(manifest_path)
        local_manifest_path = self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        local_payload = _load_json_mapping(local_manifest_path)
        if local_payload is None:
            local_payload = {**incoming_payload, "regions": []}
        validate_manifest_extension(
            local_payload,
            incoming_payload,
            selected_stems=_batch_stems(batch),
        )
        files_to_copy: list[tuple[Path, Path]] = []
        files_to_copy.append((manifest_path, local_manifest_path))
        for stem in _batch_stems(batch):
            for project in _source_projects(self.data_root.processed_v2, stem):
                relative = f"processed_v2/{project}/sentences/{stem}.parquet"
                incoming = _verified_incoming_artifact(received_data.path, artifacts, relative)
                validate_sentence_output(
                    incoming,
                    expected_sha256=_receipt_digest(artifacts, relative).sha256,
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
        for stem in _batch_stems(batch):
            for project in _source_projects(self.data_root.processed_v2, stem):
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

    def _publish_batch(self, batch: dict[str, Any]) -> None:
        try:
            self._assert_baseline()
        except Exception as error:
            batch["error"] = type(error).__name__
            self._write_ledger()
            raise ControllerRunError(f"Protected publication baseline changed: {error}") from error
        message = _publication_message(batch)
        try:
            commit = self.publisher.publish_sentence_batch(
                self.data_root.processed_v2,
                _batch_stems(batch),
                message,
            )
            self.publisher.verify_sentence_batch(
                self.data_root.processed_v2,
                _batch_stems(batch),
            )
        except Exception:
            batch["state"] = "ready_to_publish"
            batch["error"] = "publisher_failure"
            self._write_ledger()
            raise
        batch["state"] = "published"
        batch["hf_commit"] = commit
        batch["published_at"] = _timestamp()
        batch["error"] = None
        self._write_ledger()
        self._cleanup_remote_job(batch)

    def _assert_baseline(self) -> None:
        readme_hash, map_hash = _baseline_hashes(self.data_root)
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
        if batch is None or batch.get("state") not in _ACTIVE_STATES:
            self._write_ledger()
            return
        job_id = batch.get("oar_job_id")
        if isinstance(job_id, str) and job_id:
            with suppress(Exception):
                self._run_frontend(("oardel", job_id), allow_failure=True)
        batch["state"] = "cancelled"
        batch["error"] = "cancelled_by_interrupt"
        self._write_ledger()


class SubprocessGrid5000Transport(_SubprocessGrid5000Transport):
    """Compatibility wrapper that preserves the façade's executable seam."""

    def __init__(self, site: str) -> None:
        super().__init__(site, executable_resolver=_required_executable)


class HfHubSentencePublisher(_HfHubSentencePublisher):
    """Compatibility wrapper that preserves the façade's patch seams."""

    def __init__(self, repo_id: str, *, token: str | None, cache_dir: Path) -> None:
        super().__init__(
            repo_id,
            token=token,
            cache_dir=cache_dir,
            # Resolve these names when the operation runs so the historical
            # controller-level monkeypatch seams remain usable after init.
            upload_function=lambda *args, **kwargs: upload_files(*args, **kwargs),
            download_function=lambda *args, **kwargs: _download_hf_file(*args, **kwargs),
        )


def run_grid5000_sentence_controller(
    data_root: DataRoot,
    *,
    site: str = "grenoble",
    queue: str = DEFAULT_GRID5000_QUEUE,
    gpu_model: str = DEFAULT_GRID5000_GPU_MODEL,
    repo_id: str = V2_REPO_ID,
    max_stems: int = DEFAULT_MAX_STEMS,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    batch_size: int = 256,
    inference_batch_size: int = 16,
    walltime: str = DEFAULT_WALLTIME,
    run_id: str | None = None,
    hf_token: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the controller under its non-blocking local lock."""
    transport = SubprocessGrid5000Transport(site)
    publisher = HfHubSentencePublisher(
        repo_id,
        token=resolve_hf_token(hf_token),
        cache_dir=data_root.cache / "hf-verify",
    )
    controller = Grid5000SentenceController(
        data_root,
        site=site,
        queue=queue,
        gpu_model=gpu_model,
        repo_id=repo_id,
        transport=transport,
        publisher=publisher,
        max_stems=max_stems,
        max_input_bytes=max_input_bytes,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        walltime=walltime,
        run_id=run_id,
        repo_root=repo_root,
    )
    with exclusive_run_lock(data_root.cache / "grid5000-sentence-splitting.lock"):
        return controller.run()


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ControllerRunError(f"Required executable is unavailable: {name}")
    return executable


def _git_source_commit(repo_root: Path) -> str:
    return _git_source_commit_impl(repo_root, executable_resolver=_required_executable)


def _download_hf_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None,
    local_dir: Path,
) -> Path:
    return _download_hf_file_impl(
        repo_id,
        filename,
        token=token,
        local_dir=local_dir,
    )


__all__ = [
    "DEFAULT_GRID5000_GPU_MODEL",
    "DEFAULT_GRID5000_QUEUE",
    "ControllerLimits",
    "ControllerRunError",
    "Grid5000SentenceController",
    "Grid5000Transport",
    "HfHubSentencePublisher",
    "HubPublisher",
    "SubprocessGrid5000Transport",
    "run_grid5000_sentence_controller",
]
