"""Resumable local controller for Grid5000 sentence-splitting jobs."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token, upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.run_lock import exclusive_run_lock
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import (
    V2_ADDED_WIKIPEDIA_TAG_MAP_PATH,
    V2_REPO_ID,
)
from osm_polygon_wikidata_only.v2.publication import sentence_publication_ops
from osm_polygon_wikidata_only.v2.sat import DEFAULT_SAT_MODEL_REVISION
from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

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

_LEDGER_FILENAME = "grid5000_sentence_run.json"
_RUN_DIRECTORY = "grid5000_sentence_runs"
_REMOTE_NAMESPACE = "$HOME/osm-polygon-wikidata-only-grid5000"
_SEGMENTER_VERSION = "2.2.1"
_ACTIVE_STATES = frozenset({"submitted", "running"})
_TERMINAL_STATES = frozenset({"terminated", "finishing", "failed", "error", "cancelled"})
_JOB_ID_PATTERN = re.compile(r"(?:job\s+id|job_id)\s*[:=]\s*(\d+)", re.IGNORECASE)
_STATE_PATTERN = re.compile(r"state\s*=\s*([A-Za-z_]+)", re.IGNORECASE)
_EXIT_CODE_PATTERN = re.compile(r"exit[_ ]code\s*=\s*(-?\d+)", re.IGNORECASE)


class Grid5000Transport(Protocol):
    """Frontend and file-transfer operations owned by the local controller."""

    def run_frontend(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one lightweight command on the configured site frontend."""

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        """Upload a local staging tree into a remote run-owned directory."""

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        """Download a remote result tree into a local temporary directory."""

    def remove_tree(self, remote_root: str) -> None:
        """Remove one exact run-owned remote directory."""


class HubPublisher(Protocol):
    """Local Hugging Face publication and verification boundary."""

    def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str:
        """Publish one atomic sentence batch and return its commit reference."""

    def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None:
        """Verify the uploaded sentence files and protected card assets."""


class ControllerRunError(RuntimeError):
    """Raised when a batch needs operator-visible resume or retry handling."""


@dataclass(frozen=True, slots=True)
class ControllerLimits:
    """Immutable limits recorded in every controller ledger."""

    max_stems: int = DEFAULT_MAX_STEMS
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    batch_size: int = 256
    inference_batch_size: int = 16
    walltime: str = DEFAULT_WALLTIME

    def as_payload(self) -> dict[str, object]:
        return {
            "max_stems": self.max_stems,
            "max_input_bytes": self.max_input_bytes,
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "walltime": self.walltime,
        }


class Grid5000SentenceController:
    """Coordinate one serial, resumable stream of Grid5000 GPU jobs."""

    def __init__(
        self,
        data_root: DataRoot,
        *,
        site: str = "grenoble",
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
        if self.ledger_path.is_file():
            ledger = _read_ledger(self.ledger_path)
            stored_run_id = ledger.get("run_id")
            if not isinstance(stored_run_id, str):
                raise ControllerRunError("Sentence ledger has no valid run_id")
            if self.run_id is not None and self.run_id != stored_run_id:
                raise ControllerRunError("Requested run_id does not match the existing ledger")
            self.run_id = stored_run_id
            self._validate_immutable_ledger(ledger)
        else:
            if self.run_id is None:
                self.run_id = _new_run_id()
            if not _is_safe_run_id(self.run_id):
                raise ControllerRunError(f"Unsafe Grid5000 run_id: {self.run_id!r}")
            ledger = self._new_ledger()
            self._write_ledger(ledger)
        self._ledger = ledger
        return ledger

    def run(self) -> dict[str, Any]:
        """Process batches serially until all finalized V2 stems are published."""
        ledger = self.initialize()
        try:
            for batch in ledger["batches"]:
                self._current_batch = batch
                state = str(batch["state"])
                if state == "published":
                    continue
                if state == "ready_to_publish":
                    self._publish_batch(batch)
                    continue
                if state in _ACTIVE_STATES:
                    self._reconcile_batch(batch)
                    if batch["state"] == "ready_to_publish":
                        self._publish_batch(batch)
                    continue
                if state in {"planned", "failed"}:
                    self._submit_batch(batch)
                    self._reconcile_batch(batch)
                    if batch["state"] == "ready_to_publish":
                        self._publish_batch(batch)
                    continue
                raise ControllerRunError(
                    f"Batch {batch['index']} is in unsupported state {state!r}"
                )
            if not all(batch["state"] == "published" for batch in ledger["batches"]):
                raise ControllerRunError("Sentence run is incomplete and remains resumable")
            self._finalize_run()
            return ledger
        except KeyboardInterrupt:
            self._handle_interrupt()
            raise

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
        expected = {
            "contract_version": GRID5000_SENTENCE_CONTRACT_VERSION,
            "repo_id": self.repo_id,
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "segmenter_version": _SEGMENTER_VERSION,
            "site": self.site,
            "limits": self.limits.as_payload(),
        }
        for key, value in expected.items():
            if ledger.get(key) != value:
                raise ControllerRunError(f"Sentence ledger immutable field changed: {key}")
        if not isinstance(ledger.get("baseline_readme_sha256"), str):
            raise ControllerRunError("Sentence ledger has no README baseline hash")
        if not isinstance(ledger.get("baseline_map_sha256"), str):
            raise ControllerRunError("Sentence ledger has no comparison-map baseline hash")
        if not isinstance(ledger.get("batches"), list):
            raise ControllerRunError("Sentence ledger batches must be a list")

    def _write_ledger(self, ledger: dict[str, Any] | None = None) -> None:
        if ledger is not None:
            self._ledger = ledger
        if self._ledger is None:
            raise ControllerRunError("Cannot write an uninitialized sentence ledger")
        self._ledger["updated_at"] = _timestamp()
        atomic_write_text(self.ledger_path, json_dumps(self._ledger) + "\n")

    def _submit_batch(self, batch: dict[str, Any]) -> None:
        batch["attempt"] = int(batch.get("attempt", 0)) + 1
        batch["state"] = "submitted"
        batch["oar_job_id"] = None
        batch["remote_job_root"] = _remote_job_root(
            self.remote_run_root, int(batch["index"]), int(batch["attempt"])
        )
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
            submit_command = (
                "oarsub",
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
        stems = " ".join(shlex.quote(stem) for stem in _batch_stems(batch))
        return (
            f'cd "{remote_job_root}/code" && '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="{uv_cache}" '
            f"uv sync --frozen --extra sentence-splitting-gpu --no-dev && "
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="{uv_cache}" '
            f"uv run --no-sync python scripts/grid5000_sentence_job.py "
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
            receipt = _read_json_mapping(received / "receipt.json")
            self._validate_receipt(batch, receipt)
            received_data = DataRoot(received / "data")
            succeeded = state == "terminated" and exit_code in {None, 0}
            if succeeded and receipt.get("status") == "succeeded":
                self._import_success(batch, received_data, receipt)
                batch["state"] = "ready_to_publish"
                self._write_ledger()
                return
            self._import_partial(batch, received_data)
            batch["state"] = "failed"
            batch["error"] = str(receipt.get("error_type") or "remote_job_failed")
            self._write_ledger()
        self._cleanup_remote_job(batch)
        raise ControllerRunError(f"Grid5000 batch {batch['index']} failed and is retryable")

    def _validate_receipt(self, batch: Mapping[str, object], receipt: Mapping[str, object]) -> None:
        expected = {
            "job_id": str(batch["oar_job_id"]),
            "source_commit": self.source_commit,
            "model_id": SAT_MODEL_ID,
            "model_revision": DEFAULT_SAT_MODEL_REVISION,
            "stems": _batch_stems(batch),
        }
        for key, value in expected.items():
            actual = receipt.get(key)
            if key == "stems" and isinstance(actual, list):
                actual = tuple(str(stem) for stem in actual)
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
            _copy_file_atomically(*source)
        self._import_checkpoints(batch, received_data)

    def _import_partial(self, batch: Mapping[str, object], received_data: DataRoot) -> None:
        self._import_checkpoints(batch, received_data)

    def _import_checkpoints(self, batch: Mapping[str, object], received_data: DataRoot) -> None:
        for stem in _batch_stems(batch):
            for project in _source_projects(self.data_root.processed_v2, stem):
                incoming = received_data.v2_cache / "sentence-checkpoints" / stem / project
                if not incoming.is_dir():
                    continue
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


class SubprocessGrid5000Transport:
    """SSH/rsync transport restricted to frontend and run-owned paths."""

    def __init__(self, site: str) -> None:
        self.site = site

    def run_frontend(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - args are controller-generated frontend commands
            [_required_executable("ssh"), self.site, *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def upload_tree(self, local_root: Path, remote_root: str) -> None:
        result = subprocess.run(  # noqa: S603 - remote_root is a validated run namespace
            [
                _required_executable("rsync"),
                "-a",
                f"{local_root}/",
                f"{self.site}:{remote_root}/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 upload failed: {(result.stderr or '').strip()}")

    def download_tree(self, remote_root: str, local_root: Path) -> None:
        local_root.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603 - remote_root is a validated run namespace
            [
                _required_executable("rsync"),
                "-a",
                f"{self.site}:{remote_root}/",
                f"{local_root}/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ControllerRunError(f"Grid5000 download failed: {(result.stderr or '').strip()}")

    def remove_tree(self, remote_root: str) -> None:
        if not remote_root.startswith(_REMOTE_NAMESPACE + "/"):
            raise ControllerRunError("Refusing to remove an outside Grid5000 namespace")
        result = self.run_frontend(("rm", "-rf", remote_root))
        if result.returncode != 0:
            raise ControllerRunError("Grid5000 cleanup failed")


class HfHubSentencePublisher:
    """Publish sentence artifacts locally and verify exact remote bytes."""

    def __init__(self, repo_id: str, *, token: str | None, cache_dir: Path) -> None:
        self.repo_id = repo_id
        self.token = token
        self.cache_dir = Path(cache_dir)

    def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str:
        operations = sentence_publication_ops(processed_v2, stems)
        return upload_files(
            self.repo_id,
            ops=operations,
            token=self.token,
            commit_message=message,
            num_threads=2,
        )

    def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None:
        operations = sentence_publication_ops(processed_v2, stems)
        expected = [(operation.local_path, operation.path_in_repo) for operation in operations]
        map_path = processed_v2 / V2_ADDED_WIKIPEDIA_TAG_MAP_PATH
        expected.append((map_path, V2_ADDED_WIKIPEDIA_TAG_MAP_PATH))
        if any(local is None for local, _ in expected):
            raise ControllerRunError("HF verification received an incomplete publication plan")
        inventory = RemoteInventory.fetch(self.repo_id, token=self.token)
        missing = [remote for _, remote in expected if not inventory.contains(remote)]
        if missing:
            raise ControllerRunError(f"HF sentence publication is missing files: {missing}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="hf-sentence-verify-", dir=self.cache_dir
        ) as temporary:
            for local, remote in expected:
                assert local is not None
                downloaded = _download_hf_file(
                    self.repo_id,
                    remote,
                    token=self.token,
                    local_dir=Path(temporary),
                )
                if sha256_file(downloaded) != sha256_file(local):
                    raise ControllerRunError(f"HF sentence publication hash mismatch: {remote}")


def run_grid5000_sentence_controller(
    data_root: DataRoot,
    *,
    site: str = "grenoble",
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


def _baseline_hashes(data_root: DataRoot) -> tuple[str, str]:
    readme = data_root.processed_v2 / "README.md"
    map_path = data_root.processed_v2 / V2_ADDED_WIKIPEDIA_TAG_MAP_PATH
    if not readme.is_file() or not map_path.is_file():
        raise ControllerRunError("Protected V2 README or comparison map is missing")
    return sha256_file(readme), sha256_file(map_path)


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json_mapping(path)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise ControllerRunError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(raw, dict):
        raise ControllerRunError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], raw)


def _read_ledger(path: Path) -> dict[str, Any]:
    return _read_json_mapping(path)


def _receipt_artifacts(receipt: Mapping[str, object]) -> dict[str, FileDigest]:
    raw = receipt.get("artifacts")
    if not isinstance(raw, list):
        raise ControllerRunError("Grid5000 receipt artifacts must be a list")
    artifacts: dict[str, FileDigest] = {}
    for raw_artifact in raw:
        if not isinstance(raw_artifact, Mapping):
            raise ControllerRunError("Grid5000 receipt artifact must be an object")
        relative = raw_artifact.get("relative_path")
        size = raw_artifact.get("size")
        digest = raw_artifact.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise ControllerRunError("Grid5000 receipt artifact has invalid fields")
        if relative in artifacts:
            raise ControllerRunError(f"Grid5000 receipt has duplicate artifact: {relative}")
        artifacts[relative] = FileDigest(
            relative_path=relative,
            size=size,
            sha256=digest,
        )
    return artifacts


def _receipt_digest(artifacts: Mapping[str, FileDigest], relative: str) -> FileDigest:
    try:
        return artifacts[relative]
    except KeyError as error:
        raise ControllerRunError(f"Grid5000 receipt is missing artifact: {relative}") from error


def _verified_incoming_artifact(
    root: Path,
    artifacts: Mapping[str, FileDigest],
    relative: str,
) -> Path:
    digest = _receipt_digest(artifacts, relative)
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ControllerRunError(f"Grid5000 artifact escapes result root: {relative}") from error
    if not path.is_file():
        raise ControllerRunError(f"Grid5000 artifact is missing from result: {relative}")
    if path.stat().st_size != digest.size or sha256_file(path) != digest.sha256:
        raise ControllerRunError(f"Grid5000 artifact hash mismatch: {relative}")
    return path


def _copy_file_atomically(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temporary)
    try:
        with source.open("rb") as source_stream, os.fdopen(fd, "wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _copy_required(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ControllerRunError(f"Required staging file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _batch_stems(batch: Mapping[str, object]) -> tuple[str, ...]:
    raw = batch.get("stems")
    if not isinstance(raw, list) or not all(isinstance(stem, str) for stem in raw):
        raise ControllerRunError("Sentence ledger batch stems are invalid")
    return tuple(str(stem) for stem in raw)


def _source_projects(processed_v2: Path, stem: str) -> tuple[str, ...]:
    return tuple(
        "wikivoyage" if "wikivoyage" in path.parts else "wikipedia"
        for path in sentence_source_paths(processed_v2, stem)
    )


def _publication_message(batch: Mapping[str, object]) -> str:
    stems = _batch_stems(batch)
    if len(stems) == 1:
        return f"Add Grid5000 sentence split {stems[0]}"
    return f"Add Grid5000 sentence splits {stems[0]} through {stems[-1]} ({len(stems)} regions)"


def _parse_job_id(output: str) -> str:
    match = _JOB_ID_PATTERN.search(output)
    if match is None:
        raise ControllerRunError("oarsub did not return a job ID")
    return match.group(1)


def _parse_job_status(result: subprocess.CompletedProcess[str]) -> tuple[str, int | None]:
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    match = _STATE_PATTERN.search(text)
    state = match.group(1).lower() if match else _infer_job_state(text)
    exit_match = _EXIT_CODE_PATTERN.search(text)
    exit_code = int(exit_match.group(1)) if exit_match else None
    return state, exit_code


def _infer_job_state(text: str) -> str:
    lowered = text.lower()
    for state in (*_TERMINAL_STATES, "waiting", "launching", "running"):
        if state in lowered:
            return state
    return "unknown"


def _remote_job_root(remote_run_root: str, index: int, attempt: int) -> str:
    return f"{remote_run_root}/jobs/batch-{index:08d}-attempt-{attempt:02d}"


def _git_source_commit(repo_root: Path) -> str:
    result = subprocess.run(  # noqa: S603 - fixed git revision query
        [_required_executable("git"), "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ControllerRunError("Could not determine the controller source commit")
    return result.stdout.strip()


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def _is_safe_run_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_-]+", value))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ControllerRunError(f"Required executable is unavailable: {name}")
    return executable


def _download_hf_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None,
    local_dir: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - package is a runtime dependency
        raise ControllerRunError("huggingface_hub is required for HF verification") from error
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir),
        )
    )


__all__ = [
    "ControllerLimits",
    "ControllerRunError",
    "Grid5000SentenceController",
    "Grid5000Transport",
    "HfHubSentencePublisher",
    "HubPublisher",
    "SubprocessGrid5000Transport",
    "run_grid5000_sentence_controller",
]
