"""Bounded background publication queue."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads

LOGGER = logging.getLogger(__name__)

UploadOps = list[PublicationOp]
UploadOperation = Callable[[UploadOps, str], None]


@dataclass(frozen=True)
class _UploadJob:
    ops: UploadOps
    message: str
    state_path: Path | None


class BackgroundUploadQueue:
    """Upload completed PBF artifacts while local processing continues."""

    def __init__(
        self,
        *,
        upload: UploadOperation,
        max_pending: int = 2,
        state_dir: Path | None = None,
        attempts: int = 3,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._upload = upload
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self._attempts = attempts
        self._jobs: queue.Queue[_UploadJob | None] = queue.Queue(max_pending)
        self._state_dir = state_dir
        if state_dir is not None:
            state_dir.mkdir(parents=True, exist_ok=True)
        self._failures: list[str] = []
        self._thread = threading.Thread(target=self._worker, name="hf-upload", daemon=False)
        self._closed = False
        self._thread.start()

    def submit(self, ops: UploadOps, message: str) -> None:
        if self._closed:
            raise RuntimeError("upload queue is closed")
        state_path = None
        if self._state_dir is not None:
            state_path = self._state_dir / f"{uuid4().hex}.json"
            atomic_write_text(
                state_path,
                json_dumps(
                    {
                        "message": message,
                        "ops": [
                            {
                                "action": op.action,
                                "path_in_repo": op.path_in_repo,
                                "local_path": str(op.local_path) if op.local_path else None,
                            }
                            for op in ops
                        ],
                    }
                )
                + "\n",
            )
        self._jobs.put(_UploadJob(list(ops), message, state_path))
        LOGGER.info("Queued background upload: %s", message)

    def resume_pending(self) -> int:
        if self._state_dir is None:
            return 0
        paths = sorted(self._state_dir.glob("*.json"))
        for path in paths:
            raw = json_loads(path.read_text(encoding="utf-8"))
            ops = [
                PublicationOp(
                    action=entry["action"],
                    path_in_repo=entry["path_in_repo"],
                    local_path=Path(entry["local_path"]) if entry["local_path"] else None,
                )
                for entry in raw["ops"]
            ]
            self._jobs.put(_UploadJob(ops, str(raw["message"]), path))
        return len(paths)

    def close_and_wait(self) -> list[str]:
        if not self._closed:
            self._closed = True
            self._jobs.put(None)
        self._thread.join()
        return list(self._failures)

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                try:
                    last_error: Exception | None = None
                    for _ in range(self._attempts):
                        try:
                            self._upload(job.ops, job.message)
                            last_error = None
                            break
                        # ``except Exception`` retained: ``huggingface_hub``
                        # legitimately exposes a broad set of unstable
                        # exception types; every retry attempt must see
                        # them uniformly.
                        except Exception as error:
                            last_error = error
                    if last_error is not None:
                        raise last_error
                    if job.state_path is not None:
                        job.state_path.unlink(missing_ok=True)
                    LOGGER.info("Background upload complete: %s", job.message)
                # ``except Exception`` retained: the outer branch
                # isolates the worker thread from any failure (upload,
                # state write, retry accounting) and records it into the
                # ``failures`` list rather than crashing the daemon
                # thread. The queue records failures; it does NOT
                # translate every exception into ``UploadError``.
                except Exception as error:
                    detail = f"{job.message}: {error}"
                    LOGGER.error("Background upload failed: %s", detail)
                    self._failures.append(detail)
            finally:
                self._jobs.task_done()


__all__ = ["BackgroundUploadQueue", "UploadOperation", "UploadOps"]
