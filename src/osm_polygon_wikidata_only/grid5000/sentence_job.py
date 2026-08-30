"""GPU-only sentence-splitting job executed inside a Grid5000 reservation."""

from __future__ import annotations

import importlib
import os
import subprocess
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.v2.sat import (
    DEFAULT_SAT_MODEL_REVISION,
    SaT3lSegmenter,
)
from osm_polygon_wikidata_only.v2.sentence_logic import SAT_MODEL_ID
from osm_polygon_wikidata_only.v2.sentence_runner import (
    SentenceRunResult,
    run_v2_sentence_split,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

from .sentence_protocol import (
    FileDigest,
    sentence_source_paths,
    sha256_manifest,
)

_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"
_NVIDIA_SMI_ARGS = (
    "nvidia-smi",
    "--query-gpu=name,uuid,memory.total",
    "--format=csv,noheader",
)
_GPU_ORT_PROVIDERS = (_CUDA_PROVIDER, _CPU_PROVIDER)
_CHECKPOINT_DIRECTORY = "sentence-checkpoints"


class CommandRunner(Protocol):
    """Callable boundary for the node-local GPU preflight command."""

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run one command and return its completed process."""


@dataclass(frozen=True, slots=True)
class GpuIdentity:
    """Identity captured from the reserved node's NVIDIA runtime."""

    name: str
    uuid: str
    memory_total: str


@dataclass(frozen=True, slots=True)
class JobReceipt:
    """Auditable result emitted by one Grid5000 sentence job."""

    status: str
    job_id: str
    source_commit: str
    model_id: str
    model_revision: str
    segmenter_version: str
    ort_providers: tuple[str, ...]
    gpu: tuple[GpuIdentity, ...]
    stems: tuple[str, ...]
    batch_size: int
    inference_batch_size: int
    regions: tuple[dict[str, object], ...]
    artifacts: tuple[FileDigest, ...]
    started_at: str
    finished_at: str
    error_type: str | None = None
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible receipt mapping."""
        return asdict(self)


def run_sentence_job(
    data_root: DataRoot,
    *,
    stems: Sequence[str],
    model_cache: Path,
    source_commit: str,
    job_id: str,
    batch_size: int,
    inference_batch_size: int,
    receipt_path: Path,
    command_runner: CommandRunner | None = None,
) -> JobReceipt:
    """Run one selected sentence batch on a CUDA-capable reserved node."""
    selected_stems = _normalize_stems(stems)
    started_at = _timestamp()
    runner = command_runner or _run_command
    gpu: tuple[GpuIdentity, ...] = ()
    segmenter_version = "unknown"
    try:
        _validate_sources(data_root, selected_stems)
        gpu = _query_gpus(runner)
        _require_cuda_runtime()
        segmenter = SaT3lSegmenter(
            cache_dir=model_cache,
            revision=DEFAULT_SAT_MODEL_REVISION,
            inference_batch_size=inference_batch_size,
            ort_providers=_GPU_ORT_PROVIDERS,
            require_gpu=True,
        )
        segmenter_version = str(segmenter.version)
        result = run_v2_sentence_split(
            data_root,
            segmenter=segmenter,
            batch_size=batch_size,
            stems=selected_stems,
        )
        artifacts = _collect_artifacts(data_root, selected_stems, result)
        receipt = JobReceipt(
            status="succeeded",
            job_id=job_id,
            source_commit=source_commit,
            model_id=segmenter.model_id,
            model_revision=str(segmenter.revision),
            segmenter_version=segmenter_version,
            ort_providers=tuple(segmenter.ort_providers),
            gpu=gpu,
            stems=selected_stems,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            regions=tuple(cast(dict[str, object], asdict(region)) for region in result.regions),
            artifacts=artifacts,
            started_at=started_at,
            finished_at=_timestamp(),
        )
        _write_receipt(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt = JobReceipt(
            status="failed",
            job_id=job_id,
            source_commit=source_commit,
            model_id=SAT_MODEL_ID,
            model_revision=DEFAULT_SAT_MODEL_REVISION,
            segmenter_version=segmenter_version,
            ort_providers=_GPU_ORT_PROVIDERS,
            gpu=gpu,
            stems=selected_stems,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            regions=(),
            artifacts=(),
            started_at=started_at,
            finished_at=_timestamp(),
            error_type=type(error).__name__,
            error_message="redacted; inspect the reserved-job log",
        )
        _write_receipt_safely(receipt_path, receipt)
        raise


def _normalize_stems(stems: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(stems)
    if not selected:
        raise ValueError("At least one sentence stem is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Sentence stems must be unique")
    return tuple(sorted(selected))


def _validate_sources(data_root: DataRoot, stems: Sequence[str]) -> None:
    manifest = load_v2_manifest(data_root.processed_v2)
    missing = sorted(set(stems) - set(manifest))
    if missing:
        raise ValueError(f"Sentence stems are not finalized in V2: {missing}")
    for stem in stems:
        sentence_source_paths(data_root.processed_v2, stem)


def _query_gpus(command_runner: CommandRunner) -> tuple[GpuIdentity, ...]:
    result = command_runner(_NVIDIA_SMI_ARGS)
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi GPU preflight failed")
    identities = _parse_gpu_output(result.stdout)
    if not identities:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return identities


def _parse_gpu_output(stdout: str | None) -> tuple[GpuIdentity, ...]:
    return tuple(_parse_gpu_line(line) for line in (stdout or "").splitlines() if line.strip())


def _parse_gpu_line(line: str) -> GpuIdentity:
    fields = tuple(field.strip() for field in line.split(",", 2))
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("nvidia-smi returned an invalid GPU identity")
    return GpuIdentity(name=fields[0], uuid=fields[1], memory_total=fields[2])


def _require_cuda_runtime() -> None:
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        available = set(onnxruntime.get_available_providers())
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Grid5000 sentence splitting requires CUDAExecutionProvider") from error
    if _CUDA_PROVIDER not in available:
        raise RuntimeError("Grid5000 sentence splitting requires CUDAExecutionProvider")


def _collect_artifacts(
    data_root: DataRoot,
    stems: Sequence[str],
    result: SentenceRunResult,
) -> tuple[FileDigest, ...]:
    paths: list[Path] = [result.manifest_path]
    for stem in stems:
        for project in ("wikipedia", "wikivoyage"):
            output_path = data_root.processed_v2 / project / "sentences" / f"{stem}.parquet"
            if output_path.is_file():
                paths.append(output_path)
            checkpoint_root = data_root.v2_cache / _CHECKPOINT_DIRECTORY / stem / project
            for checkpoint_path in sorted(checkpoint_root.glob("metadata.json")):
                paths.append(checkpoint_path)
            paths.extend(sorted(checkpoint_root.glob("batch-*.parquet")))
    return sha256_manifest(paths, root=data_root.path)


def _write_receipt(path: Path, receipt: JobReceipt) -> None:
    atomic_write_text(path, json_dumps(receipt.to_payload()) + "\n")


def _write_receipt_safely(path: Path, receipt: JobReceipt) -> None:
    with suppress(BaseException):
        _write_receipt(path, receipt)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the job calls this with the fixed nvidia-smi query
        list(args),
        check=False,
        capture_output=True,
        text=True,
        env=_job_environment(),
    )


def _job_environment() -> dict[str, str]:
    """Drop any Hub credential from the compute-node subprocess environment."""
    return {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}


__all__ = [
    "CommandRunner",
    "GpuIdentity",
    "JobReceipt",
    "run_sentence_job",
]
