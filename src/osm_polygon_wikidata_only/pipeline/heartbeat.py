"""Periodic, low-noise logging for long enrichment stages."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Literal, Protocol, cast

from osm_polygon_wikidata_only.enrichment.progress import EnrichmentProgressSnapshot

LOGGER = logging.getLogger(__name__)
ENRICHMENT_HEARTBEAT_INTERVAL_S = 120.0


class StopSignal(Protocol):
    def wait(self, timeout: float) -> bool: ...

    def set(self) -> None: ...


class ThreadHandle(Protocol):
    def start(self) -> None: ...

    def join(self) -> None: ...


class ThreadFactory(Protocol):
    def __call__(
        self,
        *,
        target: Callable[[], None],
        name: str,
        daemon: bool,
    ) -> ThreadHandle: ...


def _thread_factory(*, target: Callable[[], None], name: str, daemon: bool) -> ThreadHandle:
    return cast(ThreadHandle, threading.Thread(target=target, name=name, daemon=daemon))


def _debug(message: str) -> None:
    LOGGER.debug("%s", message)


class EnrichmentHeartbeat:
    """Log an enrichment snapshot periodically while a context is active."""

    def __init__(
        self,
        *,
        region: str,
        snapshot: Callable[[], EnrichmentProgressSnapshot],
        log: Callable[[str], None],
        interval_s: float = ENRICHMENT_HEARTBEAT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        stop_event: StopSignal | None = None,
        thread_factory: ThreadFactory = _thread_factory,
        debug: Callable[[str], None] = _debug,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        self._region = region
        self._snapshot = snapshot
        self._log = log
        self._interval_s = interval_s
        self._clock = clock
        self._stop = stop_event or threading.Event()
        self._thread_factory = thread_factory
        self._debug = debug
        self._started_at = clock()
        self._thread: ThreadHandle | None = None

    def run(self) -> None:
        """Wait for each interval and emit the latest factual snapshot.

        Observational heartbeat failures (a transient snapshot
        callable error or a torn-down downstream state) are
        contained: the failure is logged at debug, the stop signal
        is set, and the daemon thread exits cleanly. The heartbeat
        is a liveness signal, not an ETA, and an observational
        failure must not disrupt the surrounding pipeline.
        """
        while not self._stop.wait(self._interval_s):
            try:
                snapshot = self._snapshot()
                elapsed_minutes = int(max(0.0, self._clock() - self._started_at) // 60)
                self._log(
                    f"Enrichment progress for {self._region}: {elapsed_minutes}m elapsed; "
                    f"phase={snapshot.phase.capitalize()}; "
                    f"Wikidata {snapshot.qids_completed}/{snapshot.qids_total} QIDs; "
                    f"Wikipedia {snapshot.sites_completed}/{snapshot.sites_total} sites, "
                    f"{snapshot.articles_attempted} articles attempted"
                )
            except Exception as error:
                self._debug(f"Enrichment heartbeat stopped: {error}")
                self._stop.set()
                return

    def __enter__(self) -> EnrichmentHeartbeat:
        self._thread = self._thread_factory(
            target=self.run,
            name="enrichment-progress",
            daemon=True,
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


__all__ = ["ENRICHMENT_HEARTBEAT_INTERVAL_S", "EnrichmentHeartbeat"]
