"""Ledger, receipt, and command policy for the Grid5000 controller."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import loads as json_loads

from .sentence_protocol import (
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_STEMS,
    DEFAULT_WALLTIME,
    FileDigest,
    sentence_source_paths,
)

REMOTE_NAMESPACE = "$HOME/osm-polygon-wikidata-only-grid5000"
SEGMENTER_VERSION = "2.2.1"
GRID5000_UV_VERSION = "0.11.16"
EXOTIC_GRID5000_GPU_MODELS = frozenset({"A100-PCIE-40GB", "A100-SXM4-40GB", "H100 NVL"})
ACTIVE_STATES = frozenset({"submitted", "running"})
TERMINAL_STATES = frozenset({"terminated", "finishing", "failed", "error", "cancelled"})
SUCCESS_STATES = frozenset({"terminated", "finishing"})
RETRYABLE_ARTIFACT_FAILURES = frozenset({"missing_receipt", "invalid_receipt"})

_JOB_ID_PATTERN = re.compile(r"(?:job\s+id|job_id)\s*[:=]\s*(\d+)", re.IGNORECASE)
_STATE_PATTERN = re.compile(r"state\s*=\s*([A-Za-z_]+)", re.IGNORECASE)
_EXIT_CODE_PATTERN = re.compile(r"exit[_ ]code\s*=\s*(-?\d+)", re.IGNORECASE)


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


def baseline_hashes(data_root: DataRoot) -> tuple[str, str]:
    """Return hashes of the protected V2 publication artifacts."""
    readme = data_root.processed_v2 / "README.md"
    map_path = data_root.processed_v2 / "assets/v2_added_wikipedia_tag_documents.png"
    if not readme.is_file() or not map_path.is_file():
        raise ControllerRunError("Protected V2 README or comparison map is missing")
    return sha256_file(readme), sha256_file(map_path)


def load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json_mapping(path)


def read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise ControllerRunError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(raw, dict):
        raise ControllerRunError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], raw)


def validate_ledger_baselines(ledger: Mapping[str, object]) -> None:
    if not isinstance(ledger.get("baseline_readme_sha256"), str):
        raise ControllerRunError("Sentence ledger has no README baseline hash")
    if not isinstance(ledger.get("baseline_map_sha256"), str):
        raise ControllerRunError("Sentence ledger has no comparison-map baseline hash")
    if not isinstance(ledger.get("batches"), list):
        raise ControllerRunError("Sentence ledger batches must be a list")


def remote_batch_succeeded(
    state: str,
    exit_code: int | None,
    receipt: Mapping[str, object],
) -> bool:
    return (
        state in SUCCESS_STATES and exit_code in {None, 0} and receipt.get("status") == "succeeded"
    )


def receipt_artifacts(receipt: Mapping[str, object]) -> dict[str, FileDigest]:
    raw = receipt.get("artifacts")
    if not isinstance(raw, list):
        raise ControllerRunError("Grid5000 receipt artifacts must be a list")
    artifacts: dict[str, FileDigest] = {}
    for raw_artifact in raw:
        artifact = receipt_artifact(raw_artifact)
        if artifact.relative_path in artifacts:
            raise ControllerRunError(
                f"Grid5000 receipt has duplicate artifact: {artifact.relative_path}"
            )
        artifacts[artifact.relative_path] = artifact
    return artifacts


def receipt_artifact(raw_artifact: object) -> FileDigest:
    if not isinstance(raw_artifact, Mapping):
        raise ControllerRunError("Grid5000 receipt artifact must be an object")
    typed_artifact = cast(Mapping[str, object], raw_artifact)
    relative = typed_artifact.get("relative_path")
    size = typed_artifact.get("size")
    digest = typed_artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(size, int) or not isinstance(digest, str):
        raise ControllerRunError("Grid5000 receipt artifact has invalid fields")
    return FileDigest(relative_path=relative, size=size, sha256=digest)


def normalized_receipt_value(key: str, actual: object) -> object:
    if key == "stems" and isinstance(actual, list):
        return tuple(str(stem) for stem in actual)
    return actual


def receipt_digest(artifacts: Mapping[str, FileDigest], relative: str) -> FileDigest:
    try:
        return artifacts[relative]
    except KeyError as error:
        raise ControllerRunError(f"Grid5000 receipt is missing artifact: {relative}") from error


def verified_incoming_artifact(
    root: Path,
    artifacts: Mapping[str, FileDigest],
    relative: str,
) -> Path:
    digest = receipt_digest(artifacts, relative)
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


def copy_required(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ControllerRunError(f"Required staging file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def batch_stems(batch: Mapping[str, object]) -> tuple[str, ...]:
    raw = batch.get("stems")
    if not isinstance(raw, list) or not all(isinstance(stem, str) for stem in raw):
        raise ControllerRunError("Sentence ledger batch stems are invalid")
    return tuple(str(stem) for stem in raw)


def is_source_commit_migration_safe(ledger: Mapping[str, object]) -> bool:
    raw_batches = ledger.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        return False
    return all(source_commit_batch_is_safe(raw_batch) for raw_batch in raw_batches)


def source_commit_batch_is_safe(batch: object) -> bool:
    if not isinstance(batch, Mapping):
        return False
    typed_batch = cast(Mapping[str, object], batch)
    state = typed_batch.get("state")
    if state not in {"planned", "failed"}:
        return False
    if not source_commit_job_is_safe(typed_batch, state):
        return False
    return typed_batch.get("hf_commit") in (None, "")


def source_commit_job_is_safe(batch: Mapping[str, object], state: object) -> bool:
    if state == "planned":
        return batch.get("oar_job_id") in (None, "")
    if state == "failed":
        return (
            batch.get("oar_job_id") in (None, "")
            or batch.get("error") in RETRYABLE_ARTIFACT_FAILURES
        )
    return True


def source_projects(processed_v2: Path, stem: str) -> tuple[str, ...]:
    return tuple(
        "wikivoyage" if "wikivoyage" in path.parts else "wikipedia"
        for path in sentence_source_paths(processed_v2, stem)
    )


def publication_message(batch: Mapping[str, object]) -> str:
    stems = batch_stems(batch)
    if len(stems) == 1:
        return f"Add Grid5000 sentence split {stems[0]}"
    return f"Add Grid5000 sentence splits {stems[0]} through {stems[-1]} ({len(stems)} regions)"


def parse_job_id(output: str) -> str:
    match = _JOB_ID_PATTERN.search(output)
    if match is None:
        raise ControllerRunError("oarsub did not return a job ID")
    return match.group(1)


def parse_job_status(result: subprocess.CompletedProcess[str]) -> tuple[str, int | None]:
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    match = _STATE_PATTERN.search(text)
    state = match.group(1).lower() if match else infer_job_state(text)
    exit_match = _EXIT_CODE_PATTERN.search(text)
    exit_code = int(exit_match.group(1)) if exit_match else None
    return state, exit_code


def infer_job_state(text: str) -> str:
    lowered = text.lower()
    for state in (*TERMINAL_STATES, "waiting", "launching", "running"):
        if state in lowered:
            return state
    return "unknown"


def remote_job_root(remote_run_root: str, index: int, attempt: int) -> str:
    return f"{remote_run_root}/jobs/batch-{index:08d}-attempt-{attempt:02d}"


def git_source_commit(
    repo_root: Path,
    *,
    executable_resolver: Callable[[str], str] | None = None,
) -> str:
    resolve_executable = executable_resolver or required_executable
    result = subprocess.run(  # noqa: S603 - fixed git revision query
        [resolve_executable("git"), "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ControllerRunError("Could not determine the controller source commit")
    return result.stdout.strip()


def new_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def is_safe_run_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_-]+", value))


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ControllerRunError(f"Required executable is unavailable: {name}")
    return executable


__all__ = [
    "ACTIVE_STATES",
    "EXOTIC_GRID5000_GPU_MODELS",
    "GRID5000_UV_VERSION",
    "REMOTE_NAMESPACE",
    "RETRYABLE_ARTIFACT_FAILURES",
    "SEGMENTER_VERSION",
    "SUCCESS_STATES",
    "TERMINAL_STATES",
    "ControllerLimits",
    "ControllerRunError",
    "baseline_hashes",
    "batch_stems",
    "copy_required",
    "git_source_commit",
    "infer_job_state",
    "is_safe_run_id",
    "is_source_commit_migration_safe",
    "load_json_mapping",
    "new_run_id",
    "normalized_receipt_value",
    "parse_job_id",
    "parse_job_status",
    "publication_message",
    "read_json_mapping",
    "receipt_artifact",
    "receipt_artifacts",
    "receipt_digest",
    "remote_batch_succeeded",
    "remote_job_root",
    "required_executable",
    "source_commit_batch_is_safe",
    "source_projects",
    "timestamp",
    "validate_ledger_baselines",
    "verified_incoming_artifact",
]
