"""Process-wide scheduling for polite Wikimedia API traffic."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class AdaptiveRequestScheduler:
    """Bound global concurrency, pacing, and cooldown across Wikimedia hosts."""

    def __init__(
        self,
        *,
        max_in_flight: int = 3,
        requests_per_minute: float = 180,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= max_in_flight <= 3:
            raise ValueError("max_in_flight must be between 1 and 3")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._semaphore = threading.BoundedSemaphore(max_in_flight)
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0

    def defer(self, delay_s: float) -> None:
        """Apply one cooldown to every future request."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + max(0.0, delay_s))

    def run(self, operation: Callable[[], T]) -> T:
        """Run an operation after acquiring global concurrency and rate capacity."""
        with self._semaphore:
            with self._lock:
                now = self._clock()
                ready_at = max(now, self._next_request_at, self._cooldown_until)
                self._next_request_at = ready_at + self._interval
            wait = ready_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            return operation()


_DEFAULT_SCHEDULER = AdaptiveRequestScheduler()


def default_scheduler() -> AdaptiveRequestScheduler:
    return _DEFAULT_SCHEDULER


__all__ = ["AdaptiveRequestScheduler", "default_scheduler"]
