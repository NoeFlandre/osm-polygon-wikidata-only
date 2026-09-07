"""Resumable, single-job controller for the geographic-name NER pilot."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from osm_polygon_wikidata_only.grid5000.sentence_controller import (
    Grid5000Transport,
    SubprocessGrid5000Transport,
)
from osm_polygon_wikidata_only.io.atomic import atomic_copy_file, atomic_write_json
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.run_lock import exclusive_run_lock

REMOTE_NAMESPACE = "$HOME/osm-polygon-wikidata-only-grid5000/geographic-ner"
DEFAULT_SITE = "rennes"
DEFAULT_QUEUE = "besteffort"
DEFAULT_GPU_MODEL = "A40"
DEFAULT_PERIOD = "day"
JOB_SECONDS = 1020
JOB_BATCH_SIZE = 128
JOB_INFERENCE_BATCH_SIZE = 16
JOB_WALLTIME = "0:20"

_LEDGER_NAME = "ledger.json"
_LOCK_NAME = "run.lock"
_SAFE_SITE = re.compile(r"[A-Za-z][A-Za-z0-9.-]{0,63}\Z")
_SAFE_QUEUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SAFE_GPU = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SAFE_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_SAFE_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_SAFE_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")
_JOB_ID = re.compile(r"(?:OAR[_ ]JOB[_ ]ID|job[_ ]id)\s*[:=]\s*(\d+)", re.IGNORECASE)
_STATE = re.compile(r"state\s*=\s*([A-Za-z_]+)", re.IGNORECASE)
_EXIT_CODE = re.compile(r"exit[_ ]code\s*=\s*(-?\d+)", re.IGNORECASE)
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_LOCK_HASH = re.compile(r"--hash=sha256:[0-9a-fA-F]{64}(?=\s|$)")
_TERMINAL_STATES = frozenset({"terminated", "finishing", "error", "failed", "cancelled"})
_REQUIRED_RUNTIME_FILES = (
    Path("requirements/geographic-ner-gpu.txt"),
    Path("src/osm_polygon_wikidata_only/__init__.py"),
    Path("src/osm_polygon_wikidata_only/ner/__init__.py"),
    Path("src/osm_polygon_wikidata_only/ner/job.py"),
    Path("src/osm_polygon_wikidata_only/ner/pipeline.py"),
    Path("src/osm_polygon_wikidata_only/ner/otter.py"),
    Path("src/osm_polygon_wikidata_only/io/__init__.py"),
    Path("src/osm_polygon_wikidata_only/io/atomic.py"),
    Path("src/osm_polygon_wikidata_only/io/hashing.py"),
    Path("src/osm_polygon_wikidata_only/io/run_lock.py"),
    Path("src/osm_polygon_wikidata_only/utils/__init__.py"),
    Path("src/osm_polygon_wikidata_only/utils/json.py"),
)


class ControllerRunError(RuntimeError):
    """Raised when an operator-visible recovery step is required."""


class NerPublisher(Protocol):
    def __call__(self, output_dir: Path, *, repo_id: str, run_id: str) -> str:
        """Publish one already validated NER output tree."""


def _default_publisher(output_dir: Path, *, repo_id: str, run_id: str) -> str:
    from osm_polygon_wikidata_only.ner.publication import publish_ner_shard

    return publish_ner_shard(output_dir, repo_id=repo_id, run_id=run_id, token=None)


class Grid5000NerController:
    """Coordinate one immutable staging tree and one serial OAR job at a time."""

    def __init__(
        self,
        staging_dir: Path,
        run_dir: Path,
        *,
        run_id: str,
        site: str,
        queue: str,
        gpu_model: str,
        period: str,
        repo_id: str,
        transport: Grid5000Transport,
        sleep: Callable[[float], None],
        publish: bool,
        publisher: NerPublisher,
    ) -> None:
        _validate_config(run_id, site, queue, gpu_model, period, repo_id)
        self.staging_dir = Path(staging_dir)
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.site = site
        self.queue = queue
        self.gpu_model = gpu_model
        self.period = period
        self.repo_id = repo_id
        self.transport = transport
        self.sleep = sleep
        self.publish = publish
        self.publisher = publisher
        self.ledger_path = self.run_dir / _LEDGER_NAME
        self.output_dir = self.run_dir / "output"
        self.ledger: dict[str, Any] | None = None

    @property
    def remote_root(self) -> str:
        return f"{REMOTE_NAMESPACE}/{self.run_id}"

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_run_lock(self.run_dir / _LOCK_NAME):
            self.ledger = self._load_or_create_ledger()
            return self._run_locked()

    def _run_locked(self) -> dict[str, Any]:
        ledger = self._require_ledger()
        state = str(ledger["state"])
        if state in {"published", "submitting", "failed", "ready_to_publish", "completed"}:
            return self._handle_stored_state(state)
        self._start_or_resume_job(state)
        self._reconcile()
        return self._require_ledger()

    def _handle_stored_state(self, state: str) -> dict[str, Any]:
        ledger = self._require_ledger()
        if state in {"submitting", "failed"}:
            self._raise_stored_recovery(state)
        if state == "ready_to_publish":
            self._publish_if_requested()
        if state == "completed":
            self._publish_completed_if_requested()
        return ledger

    def _raise_stored_recovery(self, state: str) -> None:
        if state == "submitting":
            raise ControllerRunError(
                "OAR submission result is uncertain; operator recovery required before retry"
            )
        raise ControllerRunError("Grid5000 job failed; output is retained and remains resumable")

    def _publish_if_requested(self) -> None:
        if self.publish:
            self._publish()

    def _publish_completed_if_requested(self) -> None:
        if self.publish:
            self._mark_ready_to_publish()
            self._publish()

    def _start_or_resume_job(self, state: str) -> None:
        ledger = self._require_ledger()
        if state == "planned":
            self._submit(upload=not bool(ledger.get("uploaded")))
            return
        if state == "paused":
            self._submit(upload=False)
            return
        if state != "running":
            raise ControllerRunError(f"Unsupported geographic NER ledger state: {state!r}")

    def _load_or_create_ledger(self) -> dict[str, Any]:
        if self.ledger_path.is_file():
            ledger = _read_mapping(self.ledger_path)
            self._validate_ledger(ledger)
            return ledger
        manifest, digest = _stage_manifest(self.staging_dir)
        ledger = {
            "version": 1,
            "run_id": self.run_id,
            "repo_id": self.repo_id,
            "site": self.site,
            "queue": self.queue,
            "gpu_model": self.gpu_model,
            "period": self.period,
            "batch_size": JOB_BATCH_SIZE,
            "remote_root": self.remote_root,
            "stage_manifest": manifest,
            "stage_sha256": digest,
            "state": "planned",
            "job_id": None,
            "attempt": 0,
            "uploaded": False,
            "remote_home": None,
            "error": None,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
        }
        atomic_write_json(self.ledger_path, ledger)
        return ledger

    def _validate_ledger(self, ledger: Mapping[str, object]) -> None:
        self._validate_ledger_identity(ledger)
        self._validate_ledger_shape(ledger)
        stored_manifest = ledger["stage_manifest"]
        stage_digest = ledger["stage_sha256"]
        current_manifest, current_digest = _stage_manifest(self.staging_dir)
        if current_manifest != stored_manifest or current_digest != stage_digest:
            raise ControllerRunError(
                "Geographic NER staging manifest is immutable after submission"
            )

    def _validate_ledger_identity(self, ledger: Mapping[str, object]) -> None:
        expected = {
            "version": 1,
            "run_id": self.run_id,
            "repo_id": self.repo_id,
            "site": self.site,
            "queue": self.queue,
            "gpu_model": self.gpu_model,
            "period": self.period,
            "batch_size": JOB_BATCH_SIZE,
            "remote_root": self.remote_root,
        }
        for key, value in expected.items():
            if ledger.get(key) != value:
                raise ControllerRunError(f"Immutable geographic NER ledger field changed: {key}")

    def _validate_ledger_shape(self, ledger: Mapping[str, object]) -> None:
        if not isinstance(ledger.get("state"), str):
            raise ControllerRunError("Geographic NER ledger has no valid state")
        self._validate_ledger_manifest(ledger)
        self._validate_ledger_digest(ledger)

    def _validate_ledger_manifest(self, ledger: Mapping[str, object]) -> None:
        manifest = ledger.get("stage_manifest")
        if not isinstance(manifest, list) or any(
            not isinstance(entry, Mapping) for entry in manifest
        ):
            raise ControllerRunError("Geographic NER ledger has no staging digest")

    def _validate_ledger_digest(self, ledger: Mapping[str, object]) -> None:
        digest = ledger.get("stage_sha256")
        if not isinstance(digest, str) or _SAFE_DIGEST.fullmatch(digest) is None:
            raise ControllerRunError("Geographic NER ledger has no staging digest")

    def _submit(self, *, upload: bool) -> None:
        if upload:
            self._stage_for_submission()
        self._policy_check()
        remote_home = self._remote_home()
        self._prepare_submission()
        self._submit_job(self._submission_command(remote_home))
        self._policy_check()

    def _stage_for_submission(self) -> None:
        ledger = self._require_ledger()
        self._ensure_remote_namespace()
        self._upload_stage()
        ledger["uploaded"] = True
        self._write_ledger()

    def _remote_home(self) -> str:
        remote_home = self._require_ledger().get("remote_home")
        if isinstance(remote_home, str):
            return remote_home
        remote_home = self._resolve_remote_home()
        self._require_ledger()["remote_home"] = remote_home
        return remote_home

    def _prepare_submission(self) -> None:
        ledger = self._require_ledger()
        ledger["attempt"] = int(ledger.get("attempt", 0)) + 1
        ledger["state"] = "submitting"
        ledger["job_id"] = None
        ledger["error"] = None
        self._write_ledger()

    def _submit_job(self, command: tuple[str, ...]) -> None:
        try:
            result = self.transport.run_frontend(command)
        except BaseException as error:
            self._raise_uncertain_submission(error)
        if result.returncode != 0:
            self._raise_uncertain_submission(None)
        job_id = _parse_job_id(_result_text(result))
        if job_id is None:
            self._raise_uncertain_submission(None)
        self._record_submitted_job(job_id)

    def _record_submitted_job(self, job_id: str) -> None:
        ledger = self._require_ledger()
        ledger["job_id"] = job_id
        ledger["state"] = "running"
        ledger["error"] = None
        self._write_ledger()

    def _raise_uncertain_submission(self, error: BaseException | None) -> NoReturn:
        self._write_submission_uncertain(error)
        raise ControllerRunError(
            "OAR submission result is uncertain; operator recovery required before retry"
        ) from error

    def _write_submission_uncertain(self, error: BaseException | None) -> None:
        ledger = self._require_ledger()
        ledger["state"] = "submitting"
        ledger["job_id"] = None
        ledger["error"] = "submission_uncertain"
        if error is not None:
            ledger["submission_exception"] = type(error).__name__
        self._write_ledger()

    def _submission_command(self, remote_home: str) -> tuple[str, ...]:
        root = f"{remote_home}{self.remote_root[len('$HOME') :]}"
        return (
            "oarsub",
            "-q",
            self.queue,
            "-p",
            f"gpu_model='{self.gpu_model}'",
            "-l",
            f"host=1/gpu=1,walltime={JOB_WALLTIME}",
            "-t",
            self.period,
            f"bash {shlex.quote(root + '/run.sh')}",
        )

    def _ensure_remote_namespace(self) -> None:
        self._run_frontend(
            (
                "mkdir",
                "-p",
                self.remote_root,
                f"{self.remote_root}/output",
                f"{self.remote_root}/shared-cache",
                f"{self.remote_root}/uv-cache",
                f"{self.remote_root}/environment-cache",
            )
        )

    def _upload_stage(self) -> None:
        manifest, digest = _stage_manifest(self.staging_dir)
        if digest != self._require_ledger().get("stage_sha256"):
            raise ControllerRunError("Geographic NER staging changed before upload")
        with tempfile.TemporaryDirectory(prefix=".ner-stage-", dir=self.run_dir) as temporary:
            staged = Path(temporary)
            _copy_stage_inputs(self.staging_dir, staged)
            _, current_digest = _stage_manifest(self.staging_dir)
            if current_digest != digest:
                raise ControllerRunError("Geographic NER staging changed during upload snapshot")
            _verify_stage_copy(staged, manifest)
            self.transport.upload_tree(staged, self.remote_root)

    def _resolve_remote_home(self) -> str:
        result = self._run_frontend(("printf", "%s", "$HOME"), allow_failure=True)
        if result.returncode != 0:
            raise ControllerRunError("Could not resolve the Grid5000 remote home")
        home = (result.stdout or "").strip()
        if not re.fullmatch(r"/[A-Za-z0-9._/-]+", home) or ".." in home.split("/"):
            raise ControllerRunError("Grid5000 remote home is invalid")
        return home

    def _reconcile(self) -> None:
        job_id = self._known_job_id()
        state, exit_code = self._wait_for_job(job_id)
        self._policy_check()
        self._retrieve(state, exit_code)

    def _known_job_id(self) -> str:
        job_id = self._require_ledger().get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ControllerRunError(
                "Known geographic NER run has no OAR job ID; refusing duplicate submission"
            )
        return job_id

    def _wait_for_job(self, job_id: str) -> tuple[str, int | None]:
        while True:
            result = self._run_frontend(("oarstat", "-f", "-j", job_id), allow_failure=True)
            state, exit_code = _parse_job_status(result)
            if state == "unknown":
                raise ControllerRunError(
                    f"Could not inspect OAR job {job_id}; run remains resumable"
                )
            if state not in _TERMINAL_STATES:
                self.sleep(10.0)
                continue
            return state, exit_code

    def _retrieve(self, state: str, exit_code: int | None) -> None:
        try:
            self._download_output()
        except ControllerRunError:
            self._mark_failed("invalid_output")
            raise
        receipt = self._validated_received_output()
        self._finish_retrieved(state, exit_code, receipt)

    def _download_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ner-receive-", dir=self.run_dir) as temporary:
            received = Path(temporary)
            self.transport.download_tree(f"{self.remote_root}/output", received)
            remote_output = received / "output"
            if not (remote_output / "receipt.json").is_file():
                remote_output = received
            _validate_downloaded_output_tree(remote_output)
            _copy_tree(remote_output, self.output_dir)

    def _validated_received_output(self) -> dict[str, Any]:
        try:
            return _validate_output(self.output_dir, self.staging_dir)
        except ControllerRunError:
            self._mark_failed("invalid_output")
            raise

    def _finish_retrieved(
        self, state: str, exit_code: int | None, receipt: Mapping[str, object]
    ) -> None:
        if not _job_succeeded(state, exit_code):
            self._mark_failed("remote_job_failed")
            raise ControllerRunError(
                "Grid5000 geographic NER job failed; output was retrieved and remains resumable"
            )
        if receipt["status"] == "paused":
            self._set_state("paused")
            return
        self._mark_ready_to_publish()
        if self.publish:
            self._publish()
        else:
            self._set_state("completed")

    def _mark_failed(self, error: str) -> None:
        ledger = self._require_ledger()
        ledger["state"] = "failed"
        ledger["error"] = error
        self._write_ledger()

    def _set_state(self, state: str) -> None:
        ledger = self._require_ledger()
        if ledger.get("state") == "completed" and state == "paused":
            raise ControllerRunError("Completed geographic NER run cannot regress to paused")
        ledger["state"] = state
        ledger["error"] = None
        self._write_ledger()

    def _mark_ready_to_publish(self) -> None:
        ledger = self._require_ledger()
        ledger["state"] = "ready_to_publish"
        ledger["error"] = None
        self._write_ledger()

    def _publish(self) -> None:
        ledger = self._require_ledger()
        _validate_output(self.output_dir, self.staging_dir)
        try:
            commit = self.publisher(self.output_dir, repo_id=self.repo_id, run_id=self.run_id)
        except ControllerRunError:
            raise
        except Exception as error:
            ledger["state"] = "ready_to_publish"
            ledger["error"] = "publisher_failure"
            self._write_ledger()
            raise ControllerRunError(
                "Geographic NER publication failed; run remains ready_to_publish"
            ) from error
        ledger["state"] = "published"
        ledger["commit"] = commit
        ledger["published_at"] = _timestamp()
        ledger["error"] = None
        self._write_ledger()

    def _policy_check(self) -> None:
        result = self._run_frontend(
            ("usagepolicycheck", "-t", "--sites", self.site), allow_failure=True
        )
        text = _result_text(result)
        if result.returncode != 0 or _policy_failure(text):
            raise ControllerRunError(
                f"Grid5000 usage policy check failed for site {self.site}: {text.strip()}"
            )

    def _run_frontend(self, args: Sequence[str], *, allow_failure: bool = False) -> Any:
        result = self.transport.run_frontend(tuple(args))
        if result.returncode != 0 and not allow_failure:
            raise ControllerRunError(f"Grid5000 frontend command failed: {args[0]}")
        return result

    def _write_ledger(self) -> None:
        ledger = self._require_ledger()
        ledger["updated_at"] = _timestamp()
        atomic_write_json(self.ledger_path, ledger)

    def _require_ledger(self) -> dict[str, Any]:
        if self.ledger is None:
            raise ControllerRunError("Geographic NER ledger is not initialized")
        return self.ledger


def run_grid5000_ner_controller(
    staging_dir: Path,
    run_dir: Path,
    run_id: str,
    repo_id: str,
    *,
    site: str = DEFAULT_SITE,
    queue: str = DEFAULT_QUEUE,
    gpu_model: str = DEFAULT_GPU_MODEL,
    period: str = DEFAULT_PERIOD,
    transport: Grid5000Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    publish: bool = False,
    publisher: NerPublisher | None = None,
) -> dict[str, Any]:
    """Run or resume one serial geographic NER job without live side effects in tests."""
    _validate_config(run_id, site, queue, gpu_model, period, repo_id)
    controller = Grid5000NerController(
        staging_dir,
        run_dir,
        run_id=run_id,
        site=site,
        queue=queue,
        gpu_model=gpu_model,
        period=period,
        repo_id=repo_id,
        transport=transport or SubprocessGrid5000Transport(site),
        sleep=sleep,
        publish=publish,
        publisher=publisher or _default_publisher,
    )
    return controller.run()


def _validate_config(
    run_id: str,
    site: str,
    queue: str,
    gpu_model: str,
    period: str,
    repo_id: str,
) -> None:
    _require_safe("run_id", run_id, _SAFE_RUN_ID)
    _require_safe("site", site, _SAFE_SITE)
    _require_safe("queue", queue, _SAFE_QUEUE)
    _require_safe("gpu_model", gpu_model, _SAFE_GPU)
    if period not in {"day", "night"}:
        raise ValueError(f"Unsafe Grid5000 period: {period!r}")
    _require_safe("repo_id", repo_id, _SAFE_REPO)


def _require_safe(name: str, value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"Unsafe Grid5000 {name}: {value!r}")


def _stage_manifest(staging_dir: Path) -> tuple[list[dict[str, object]], str]:
    if not staging_dir.is_dir():
        raise ControllerRunError(f"Staging directory is missing: {staging_dir}")
    files = _staging_files(staging_dir)
    manifest = [_stage_file_record(staging_dir, path) for path in files]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(encoded).hexdigest()


def _staging_files(staging_dir: Path) -> list[Path]:
    required = _required_staging_files(staging_dir)
    files = [*required, *_staged_code_files(staging_dir / "code")]
    optional_script = staging_dir / "run.sh"
    if optional_script.exists():
        files.append(optional_script)
    return files


def _required_staging_files(staging_dir: Path) -> list[Path]:
    required = [staging_dir / "input.parquet", staging_dir / "contract.json"]
    code = staging_dir / "code"
    if code.is_symlink() or not code.is_dir() or any(not path.is_file() for path in required):
        raise ControllerRunError("Staging requires input.parquet, contract.json, and code/")
    _validate_runtime_tree(staging_dir)
    return required


def _validate_runtime_tree(staging_dir: Path) -> None:
    missing = _missing_runtime_files(staging_dir)
    if missing:
        raise ControllerRunError(
            "Staging code is missing required geographic NER runtime files: "
            + ", ".join(path.as_posix() for path in missing)
        )

    _validate_runtime_lock(staging_dir / "code" / _REQUIRED_RUNTIME_FILES[0])


def _missing_runtime_files(staging_dir: Path) -> tuple[Path, ...]:
    return tuple(
        relative
        for relative in _REQUIRED_RUNTIME_FILES
        if _runtime_file_missing(staging_dir / "code" / relative)
    )


def _runtime_file_missing(path: Path) -> bool:
    return path.is_symlink() or not path.is_file()


def _validate_runtime_lock(lock: Path) -> None:
    _validate_lock_blocks(_runtime_lock_blocks(lock))


def _runtime_lock_blocks(lock: Path) -> tuple[str, ...]:
    try:
        text = lock.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ControllerRunError("Staged GPU requirements lock is unreadable") from error
    return tuple(
        block for block in re.split(r"(?m)^(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)", text) if "==" in block
    )


def _validate_lock_blocks(blocks: Sequence[str]) -> None:
    if not blocks or any(_LOCK_HASH.search(block) is None for block in blocks):
        raise ControllerRunError("Staged GPU requirements lock must hash every pinned package")


def _staged_code_files(code: Path) -> list[Path]:
    return sorted(path for path in code.rglob("*") if path.is_file())


def _stage_file_record(staging_dir: Path, path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ControllerRunError(f"Symlinks are not allowed in staging: {path}")
    return {
        "path": path.relative_to(staging_dir).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _copy_stage_inputs(source: Path, target: Path) -> None:
    for relative in (Path("input.parquet"), Path("contract.json")):
        _copy_checked(source / relative, target / relative)
    _copy_checked_tree(source / "code", target / "code")
    run_script = target / "run.sh"
    run_script.write_text(_run_script(), encoding="utf-8")
    run_script.chmod(0o755)


def _run_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_ROOT="$(cd "$(dirname "$0")" && pwd -P)"\n'
        'cd "$RUN_ROOT"\n'
        'UV_BOOTSTRAP="$RUN_ROOT/uv-bootstrap"\n'
        'UV_CACHE_DIR="$RUN_ROOT/uv-cache"\n'
        'ENV_CACHE="$RUN_ROOT/environment-cache"\n'
        'UV_BIN="$UV_BOOTSTRAP/bin/uv"\n'
        'if [ ! -x "$UV_BIN" ]; then\n'
        '  mkdir -p "$UV_BOOTSTRAP" "$UV_CACHE_DIR"\n'
        "  if command -v python3 >/dev/null 2>&1; then\n"
        '    python3 -m venv "$UV_BOOTSTRAP"\n'
        "  elif command -v uv >/dev/null 2>&1; then\n"
        '    uv venv --python 3.12 "$UV_BOOTSTRAP"\n'
        "  else\n"
        '    echo "Python 3 or uv is required for the GPU bootstrap" >&2\n'
        "    exit 1\n"
        "  fi\n"
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR" '
        '    "$UV_BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check '
        '    --no-input "uv==0.11.16"\n'
        "fi\n"
        'LOCK_SHA="$(sha256sum code/requirements/geographic-ner-gpu.txt | cut -d" " -f1)"\n'
        'ENV_DIR="$ENV_CACHE/$LOCK_SHA"\n'
        'if [ ! -x "$ENV_DIR/bin/python" ]; then\n'
        '  mkdir -p "$ENV_CACHE"\n'
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR" '
        '    "$UV_BIN" venv --python 3.12 "$ENV_DIR"\n'
        "fi\n"
        'if [ ! -f "$ENV_DIR/.ready" ]; then\n'
        '  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN UV_CACHE_DIR="$UV_CACHE_DIR" '
        '    "$UV_BIN" pip install --python "$ENV_DIR/bin/python" --require-hashes '
        "    -r code/requirements/geographic-ner-gpu.txt\n"
        '  touch "$ENV_DIR/.ready"\n'
        "fi\n"
        'export PATH="$ENV_DIR/bin:$PATH"\n'
        'export PYTHONPATH="$RUN_ROOT/code/src:$RUN_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"\n'
        "exec python -m osm_polygon_wikidata_only.ner.job "
        "--source input.parquet --output-dir output --contract contract.json "
        f"--model-cache shared-cache --seconds {JOB_SECONDS} --batch-size "
        f"{JOB_BATCH_SIZE} --inference-batch-size {JOB_INFERENCE_BATCH_SIZE}\n"
    )


def _copy_checked(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ControllerRunError(f"Required staged file is missing or unsafe: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_checked_tree(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ControllerRunError(f"Required staged code directory is missing or unsafe: {source}")
    for item in sorted(source.rglob("*")):
        _copy_checked_tree_item(source, target, item)


def _copy_checked_tree_item(source: Path, target: Path, item: Path) -> None:
    relative = item.relative_to(source)
    destination = target / relative
    if item.is_symlink():
        raise ControllerRunError(f"Symlinks are not allowed in staged code: {item}")
    if item.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
    elif item.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def _verify_stage_copy(staged: Path, manifest: Sequence[Mapping[str, object]]) -> None:
    for entry in manifest:
        _verify_stage_entry(staged, entry)


def _verify_stage_entry(staged: Path, entry: Mapping[str, object]) -> None:
    relative = entry.get("path")
    if relative == "run.sh":
        return
    relative, digest = _stage_entry_values(entry)
    path = staged / relative
    if not _stage_copy_matches(path, digest):
        raise ControllerRunError(f"Staging snapshot changed: {relative}")


def _stage_entry_values(entry: Mapping[str, object]) -> tuple[str, str]:
    relative = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ControllerRunError("Staging manifest is invalid")
    return relative, digest


def _stage_copy_matches(path: Path, digest: str) -> bool:
    return path.is_file() and sha256_file(path) == digest


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise ControllerRunError("Downloaded Grid5000 output directory is missing")
    for item in sorted(source.rglob("*")):
        _copy_output_item(source, target, item)


def _validate_downloaded_output_tree(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ControllerRunError("Downloaded Grid5000 output directory is missing")
    receipt = _read_receipt(output_dir / "receipt.json")
    allowed = _downloaded_output_paths(receipt)
    for item in sorted(output_dir.rglob("*")):
        _validate_downloaded_output_item(output_dir, item, allowed)


def _downloaded_output_paths(receipt: Mapping[str, object]) -> set[str]:
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ControllerRunError("Geographic NER receipt artifacts must be a list")
    paths = {"receipt.json", "predictions.sqlite3", "run.lock"}
    for artifact in raw_artifacts:
        if not isinstance(artifact, Mapping):
            raise ControllerRunError("Geographic NER artifact is not an object")
        paths.add(_artifact_relative(artifact.get("path")))
    return paths


def _validate_downloaded_output_item(output_dir: Path, item: Path, allowed: set[str]) -> None:
    relative = item.relative_to(output_dir).as_posix()
    if item.is_symlink():
        raise ControllerRunError(f"Symlinks are not allowed in downloaded output: {relative}")
    if item.is_file() and relative not in allowed:
        raise ControllerRunError(f"Unexpected geographic NER output file: {relative}")


def _copy_output_item(source: Path, target: Path, item: Path) -> None:
    relative = item.relative_to(source)
    destination = target / relative
    if item.is_symlink():
        raise ControllerRunError(f"Symlinks are not allowed in downloaded output: {relative}")
    if item.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
    elif item.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy_file(item, destination)


def _validate_output(output_dir: Path, staging_dir: Path) -> dict[str, Any]:
    _validate_downloaded_output_tree(output_dir)
    receipt = _read_receipt(output_dir / "receipt.json")
    _validate_receipt_header(receipt)
    _validate_receipt_identity(receipt, staging_dir)
    _validate_receipt_artifacts(output_dir, receipt)
    return receipt


def _read_receipt(receipt_path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise ControllerRunError(f"Invalid geographic NER receipt: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ControllerRunError("Geographic NER receipt must be an object")
    return receipt


def _validate_receipt_header(receipt: Mapping[str, object]) -> None:
    status = receipt.get("status")
    if status not in {"completed", "paused"}:
        raise ControllerRunError("Geographic NER receipt is incomplete")
    if receipt.get("batch_size") != JOB_BATCH_SIZE or type(receipt.get("batch_size")) is not int:
        raise ControllerRunError("Geographic NER receipt batch size is not the pilot batch size")


def _validate_receipt_identity(receipt: Mapping[str, object], staging_dir: Path) -> None:
    if receipt.get("source_sha256") != sha256_file(staging_dir / "input.parquet"):
        raise ControllerRunError("Geographic NER source hash mismatch")
    source_rows = _source_row_count(staging_dir / "input.parquet")
    if receipt.get("source_rows") != source_rows:
        raise ControllerRunError("Geographic NER source row count does not match staged input")
    contract = _read_contract_payload(staging_dir / "contract.json")
    if receipt.get("contract") != contract:
        raise ControllerRunError("Geographic NER receipt contract configuration mismatch")
    if receipt.get("contract_id") not in _contract_ids(staging_dir / "contract.json"):
        raise ControllerRunError("Geographic NER contract hash mismatch")


def _validate_receipt_artifacts(output_dir: Path, receipt: Mapping[str, object]) -> None:
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ControllerRunError("Geographic NER receipt artifacts must be a list")
    if receipt.get("status") == "completed" and not raw_artifacts:
        raise ControllerRunError("Geographic NER receipt has no completed artifacts")
    _validate_artifacts(output_dir, raw_artifacts, receipt)


def _contract_ids(path: Path) -> set[str]:
    raw = _read_contract_payload(path)
    candidates = {sha256_file(path)}
    values = cast(dict[str, Any], dict(raw))
    values["languages"] = tuple(cast(list[str], values["languages"]))
    try:
        from osm_polygon_wikidata_only.ner.pipeline import Contract

        candidates.add(Contract(**values).identity)
    except (ImportError, TypeError, ValueError) as error:
        raise ControllerRunError("Invalid staged geographic NER contract") from error
    return candidates


def _read_contract_payload(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise ControllerRunError(f"Invalid staged geographic NER contract: {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("languages"), list):
        raise ControllerRunError("Invalid staged geographic NER contract")
    return raw


def _source_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path).metadata
    except (ImportError, OSError, ValueError) as error:
        raise ControllerRunError("Could not read staged geographic NER input") from error
    if metadata is None:
        raise ControllerRunError("Staged geographic NER input has no Parquet metadata")
    return metadata.num_rows


def _validate_artifacts(
    output_dir: Path, raw_artifacts: Sequence[object], receipt: Mapping[str, object]
) -> None:
    seen: set[str] = set()
    artifacts: list[tuple[str, int]] = []
    for raw in raw_artifacts:
        artifacts.append(_validate_artifact(output_dir, raw, seen))
    _validate_artifact_order([relative for relative, _ in artifacts])
    _validate_receipt_counts(receipt, [rows for _, rows in artifacts])


def _validate_artifact(output_dir: Path, raw: object, seen: set[str]) -> tuple[str, int]:
    relative, digest, metadata = _artifact_metadata(raw)
    if relative in seen:
        raise ControllerRunError(f"Duplicate geographic NER artifact: {relative}")
    seen.add(relative)
    path = _artifact_path(output_dir, relative)
    if sha256_file(path) != digest:
        raise ControllerRunError(f"Geographic NER artifact hash mismatch: {relative}")
    _validate_artifact_size(path, relative, metadata)
    return relative, _artifact_rows(path, relative, metadata)


def _artifact_metadata(raw: object) -> tuple[str, str, Mapping[str, object]]:
    if not isinstance(raw, Mapping):
        raise ControllerRunError("Geographic NER artifact is not an object")
    metadata = cast(Mapping[str, object], raw)
    relative = metadata.get("path")
    digest = metadata.get("sha256")
    return _artifact_relative(relative), _artifact_digest(digest), metadata


def _artifact_relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or _ARTIFACT_NAME.fullmatch(value) is None
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ControllerRunError("Geographic NER artifact metadata is invalid")
    return value


def _artifact_digest(value: object) -> str:
    if not isinstance(value, str) or _SAFE_DIGEST.fullmatch(value) is None:
        raise ControllerRunError("Geographic NER artifact metadata is invalid")
    return value.lower()


def _artifact_path(output_dir: Path, relative: str) -> Path:
    path = output_dir / relative
    try:
        path.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise ControllerRunError(f"Artifact escapes output directory: {relative}") from error
    if path.is_symlink() or not path.is_file():
        raise ControllerRunError(f"Geographic NER artifact is missing: {relative}")
    return path


def _validate_artifact_size(path: Path, relative: str, metadata: Mapping[str, object]) -> None:
    if "size" not in metadata:
        return
    size = metadata.get("size")
    if type(size) is not int or size != path.stat().st_size:
        raise ControllerRunError(f"Geographic NER artifact size mismatch: {relative}")


def _artifact_rows(path: Path, relative: str, metadata: Mapping[str, object]) -> int:
    if "rows" not in metadata:
        raise ControllerRunError(f"Geographic NER artifact row count is required: {relative}")
    rows = metadata.get("rows")
    if type(rows) is not int or rows < 0:
        raise ControllerRunError(f"Geographic NER artifact row count is invalid: {relative}")
    _validate_parquet_footer(path, relative, rows)
    return rows


def _validate_artifact_order(paths: Sequence[str]) -> None:
    if not paths:
        return
    expected = [f"batch-{index:06d}.parquet" for index in range(len(paths))]
    if list(paths) != expected:
        raise ControllerRunError("Geographic NER artifact order is invalid")


def _validate_receipt_counts(receipt: Mapping[str, object], rows: Sequence[int]) -> None:
    processed = _required_count(receipt.get("processed_rows"), "processed")
    source = _required_count(receipt.get("source_rows"), "source")
    _validate_processed_rows(processed, rows)
    _validate_source_rows(receipt, processed, source)


def _required_count(value: object, name: str) -> int:
    if value is None:
        raise ControllerRunError(f"Geographic NER {name} row count is required")
    if type(value) is not int or value < 0:
        raise ControllerRunError(f"Geographic NER {name} row count is invalid")
    return value


def _validate_processed_rows(processed: int | None, rows: Sequence[int | None]) -> None:
    if processed is None:
        return
    if not _all_rows_present(rows):
        return
    if sum(cast(int, row) for row in rows) != processed:
        raise ControllerRunError("Geographic NER artifact rows do not match the receipt")


def _all_rows_present(rows: Sequence[int | None]) -> bool:
    return all(row is not None for row in rows)


def _validate_source_rows(
    receipt: Mapping[str, object], processed: int | None, source: int | None
) -> None:
    if source is None:
        return
    if processed is None:
        return
    if processed > source:
        raise ControllerRunError("Geographic NER processed rows exceed source rows")
    _require_completed_rows(receipt, processed, source)


def _require_completed_rows(receipt: Mapping[str, object], processed: int, source: int) -> None:
    if receipt.get("status") == "completed" and processed != source:
        raise ControllerRunError("Completed geographic NER receipt has incomplete rows")


def _validate_parquet_footer(path: Path, relative: str, expected_rows: int) -> None:
    header, footer = _parquet_markers(path, relative)
    if (header, footer) != (b"PAR1", b"PAR1"):
        _reject_invalid_batch(relative)
        return
    actual_rows = _read_parquet_rows(path, relative)
    if actual_rows != expected_rows:
        raise ControllerRunError(f"Geographic NER Parquet row count mismatch: {relative}")


def _parquet_markers(path: Path, relative: str) -> tuple[bytes, bytes]:
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
            stream.seek(-4, 2)
            footer = stream.read(4)
    except OSError as error:
        raise ControllerRunError(f"Could not read artifact footer: {relative}") from error
    return header, footer


def _reject_invalid_batch(relative: str) -> None:
    if re.fullmatch(r"batch-\d{6}\.parquet", relative) is not None:
        raise ControllerRunError(f"Geographic NER artifact is not Parquet: {relative}")


def _read_parquet_rows(path: Path, relative: str) -> int:
    try:
        import pyarrow.parquet as pq

    except (ImportError, OSError, ValueError) as error:
        raise ControllerRunError(f"Could not validate Parquet artifact: {relative}") from error
    try:
        return pq.read_metadata(path).num_rows
    except (OSError, ValueError) as error:
        raise ControllerRunError(f"Could not validate Parquet artifact: {relative}") from error


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as error:
        raise ControllerRunError(f"Invalid geographic NER ledger: {path}") from error
    if not isinstance(raw, dict):
        raise ControllerRunError("Geographic NER ledger must be an object")
    return raw


def _parse_job_id(text: str) -> str | None:
    match = _JOB_ID.search(text)
    return None if match is None else match.group(1)


def _parse_job_status(result: Any) -> tuple[str, int | None]:
    text = _result_text(result)
    match = _STATE.search(text)
    state = match.group(1).lower() if match else _infer_state(text)
    exit_match = _EXIT_CODE.search(text)
    exit_code = None if exit_match is None else int(exit_match.group(1))
    return state, exit_code


def _infer_state(text: str) -> str:
    lowered = text.lower()
    for state in (*_TERMINAL_STATES, "waiting", "launching", "running"):
        if state in lowered:
            return state
    return "unknown"


def _job_succeeded(state: str, exit_code: int | None) -> bool:
    return state in {"terminated", "finishing"} and exit_code in {None, 0}


def _result_text(result: Any) -> str:
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def _policy_failure(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\berror\b", lowered)
        or "violation" in lowered
        or "access denied" in lowered
        or "permission denied" in lowered
        or "quota" in lowered
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DEFAULT_GPU_MODEL",
    "DEFAULT_PERIOD",
    "DEFAULT_QUEUE",
    "DEFAULT_SITE",
    "ControllerRunError",
    "Grid5000NerController",
    "Grid5000Transport",
    "SubprocessGrid5000Transport",
    "run_grid5000_ner_controller",
]
