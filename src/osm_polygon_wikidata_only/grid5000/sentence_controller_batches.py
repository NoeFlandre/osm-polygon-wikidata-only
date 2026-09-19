"""Submission, staging, reconciliation, and retrieval for sentence batches."""

from __future__ import annotations

import shlex
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

from .sentence_controller_context import SentenceControllerContext
from .sentence_controller_policy import (
    ACTIVE_STATES,
    EXOTIC_GRID5000_GPU_MODELS,
    GRID5000_UV_VERSION,
    ControllerRunError,
    batch_stems,
    copy_required,
    parse_job_id,
    parse_job_status,
    remote_batch_succeeded,
    remote_job_root,
    source_projects,
)
from .sentence_protocol import sentence_source_paths


class SentenceControllerBatchMixin(SentenceControllerContext):
    """Run one deterministic batch through remote execution and retrieval."""

    def _process_batch(self, batch: dict[str, Any]) -> None:
        state = str(batch["state"])
        if state == "published":
            return
        if state == "ready_to_publish":
            self._publish_batch(batch)
            return
        if state in ACTIVE_STATES:
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

    def _submit_batch(self, batch: dict[str, Any]) -> None:
        batch["attempt"] = int(batch.get("attempt", 0)) + 1
        batch["state"] = "submitted"
        batch["oar_job_id"] = None
        batch["remote_job_root"] = remote_job_root(
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
            job_type_args = ("-t", "exotic") if self.gpu_model in EXOTIC_GRID5000_GPU_MODELS else ()
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
            job_id = parse_job_id(submitted.stdout or "")
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
        copy_required(self.repo_root / "pyproject.toml", code / "pyproject.toml")
        copy_required(self.repo_root / "uv.lock", code / "uv.lock")
        copy_required(self.repo_root / "README.md", code / "README.md")
        copy_required(self.repo_root / "LICENSE", code / "LICENSE")
        for asset in ("dataset_hero.png", "dataset_hero_v2.png"):
            copy_required(self.repo_root / "assets" / asset, code / "assets" / asset)
        shutil.copytree(self.repo_root / "src", code / "src")
        (code / "scripts").mkdir()
        copy_required(
            self.repo_root / "scripts/grid5000_sentence_job.py",
            code / "scripts/grid5000_sentence_job.py",
        )
        data_v2 = data / "processed_v2"
        data_v2.mkdir(parents=True)
        copy_required(
            self.data_root.processed_v2 / "manifests/processed_pbfs.json",
            data_v2 / "manifests/processed_pbfs.json",
        )
        sentence_manifest = self.data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
        if sentence_manifest.is_file():
            copy_required(sentence_manifest, data_v2 / SENTENCE_MANIFEST_RELATIVE_PATH)
        for stem in batch_stems(batch):
            for source in sentence_source_paths(self.data_root.processed_v2, stem):
                relative = source.relative_to(self.data_root.processed_v2)
                copy_required(source, data_v2 / relative)
            self._stage_checkpoint_trees(data, stem)

    def _stage_checkpoint_trees(self, staged_data: Path, stem: str) -> None:
        for project in source_projects(self.data_root.processed_v2, stem):
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
        stems = " ".join(shlex.quote(stem) for stem in batch_stems(batch))
        return (
            f'cd "{remote_job_root}/code" && '
            f'if [ ! -x "{uv_bin}" ]; then '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN python3 -m venv "{uv_bootstrap}" && '
            f'env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "{uv_python}" -m pip install '
            f'--disable-pip-version-check --no-input "uv=={GRID5000_UV_VERSION}"; '
            "fi && "
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
            state, exit_code = parse_job_status(result)
            if state not in {"terminated", "finishing", "failed", "error", "cancelled"}:
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
            if remote_batch_succeeded(state, exit_code, receipt):
                self._import_success(batch, received_data, receipt)
                batch["state"] = "ready_to_publish"
                self._write_ledger()
                return
            self._mark_batch_failed(batch, received_data, receipt)
        self._cleanup_remote_job(batch)
        raise ControllerRunError(f"Grid5000 batch {batch['index']} failed and is retryable")


__all__ = ["SentenceControllerBatchMixin"]
