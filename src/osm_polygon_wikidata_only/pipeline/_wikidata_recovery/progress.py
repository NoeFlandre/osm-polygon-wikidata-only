"""Thread-safe progress state and heartbeat for Wikidata recovery."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Literal


class RecoveryProgress:
    def __init__(
        self,
        stem: str,
        batch_total: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stem = stem
        self._batch_total = batch_total
        self._clock = clock
        self._started_at = clock()
        self._lock = threading.Lock()
        self._batch = 0
        self._stage = "starting"
        self._completed = 0
        self._total = 0
        self._documents = 0
        self._sections = 0
        self._facts = 0

    def start_batch(self, batch: int, qids: tuple[str, ...]) -> None:
        with self._lock:
            self._batch = batch
            self._stage = "starting"
            self._completed = 0
            self._total = len(qids)
            self._documents = self._sections = self._facts = 0

    def set_stage(self, stage: str, *, total: int) -> None:
        with self._lock:
            self._stage = stage
            self._completed = 0
            self._total = total

    def advance(
        self,
        count: int = 1,
        *,
        documents: int = 0,
        sections: int = 0,
        facts: int = 0,
    ) -> None:
        with self._lock:
            self._completed = min(self._total, self._completed + count)
            self._documents += documents
            self._sections += sections
            self._facts += facts

    def checkpoint_saved(self, *, documents: int, sections: int, facts: int) -> None:
        with self._lock:
            self._stage = "checkpoint saved"
            self._completed = self._total
            self._documents = documents
            self._sections = sections
            self._facts = facts

    def message(self) -> str:
        with self._lock:
            elapsed = max(0, round(self._clock() - self._started_at))
            eta = "unknown"
            if self._batch > 1 and elapsed > 0:
                remaining = max(0, self._batch_total - self._batch)
                eta = f"{round(elapsed / (self._batch - 1) * remaining)}s"
            return (
                f"Wikidata recovery progress {self._stem}: batch {self._batch}/{self._batch_total}; "
                f"{self._stage} {self._completed}/{self._total}; documents {self._documents}; "
                f"sections {self._sections}; facts {self._facts}; {elapsed}s elapsed; ETA {eta}"
            )


class RecoveryHeartbeat:
    """Emit the latest recovery snapshot periodically without affecting recovery."""

    def __init__(
        self,
        progress: RecoveryProgress,
        log: Callable[[str], None],
        *,
        interval_s: float = 60.0,
    ) -> None:
        self._progress = progress
        self._log = log
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._log(self._progress.message())
            except Exception:
                return

    def __enter__(self) -> RecoveryHeartbeat:
        self._thread = threading.Thread(
            target=self._run, name="wikidata-recovery-progress", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False


__all__: list[str] = []
