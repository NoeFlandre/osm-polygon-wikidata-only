"""Resumable local controller for Grid5000 sentence-splitting jobs.

The controller composes focused mixins; helper implementations live in the
``sentence_controller_*`` and ``sentence_*`` modules that define them.
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token, upload_files
from osm_polygon_wikidata_only.io.run_lock import exclusive_run_lock
from osm_polygon_wikidata_only.v2.config import (
    V2_REPO_ID,
)

from .sentence_controller_batches import SentenceControllerBatchMixin
from .sentence_controller_import import SentenceControllerImportMixin
from .sentence_controller_ledger import SentenceControllerLedgerMixin
from .sentence_controller_lifecycle import SentenceControllerLifecycleMixin
from .sentence_controller_policy import (
    REMOTE_NAMESPACE as _REMOTE_NAMESPACE,
)
from .sentence_controller_policy import (
    ControllerLimits,
    ControllerRunError,
)
from .sentence_controller_policy import (
    git_source_commit as _git_source_commit_impl,
)
from .sentence_protocol import (
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_STEMS,
    DEFAULT_WALLTIME,
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
from .sentence_transport import (
    Grid5000Transport,
)
from .sentence_transport import (
    SubprocessGrid5000Transport as _SubprocessGrid5000Transport,
)

_LEDGER_FILENAME = "grid5000_sentence_run.json"
DEFAULT_GRID5000_QUEUE = "besteffort"
DEFAULT_GRID5000_GPU_MODEL = "A40"
_QUEUE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_GPU_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


class Grid5000SentenceController(
    SentenceControllerLifecycleMixin,
    SentenceControllerBatchMixin,
    SentenceControllerImportMixin,
    SentenceControllerLedgerMixin,
):
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
