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
        max_requests_per_minute: float | None = None,
        minimum_requests_per_minute: float = 60,
        successes_per_increase: int = 100,
        host_throttle_window_s: float = 10.0,
        host_throttle_threshold: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= max_in_flight <= 16:
            raise ValueError("max_in_flight must be between 1 and 16")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        maximum = (
            requests_per_minute if max_requests_per_minute is None else max_requests_per_minute
        )
        if maximum < requests_per_minute:
            raise ValueError("max_requests_per_minute must not be below the initial rate")
        if minimum_requests_per_minute <= 0 or minimum_requests_per_minute > requests_per_minute:
            raise ValueError(
                "minimum_requests_per_minute must be positive and no greater than the initial rate"
            )
        if successes_per_increase <= 0:
            raise ValueError("successes_per_increase must be positive")
        if host_throttle_window_s <= 0:
            raise ValueError("host_throttle_window_s must be positive")
        if host_throttle_threshold < 1:
            raise ValueError("host_throttle_threshold must be at least 1")
        self._semaphore = threading.BoundedSemaphore(max_in_flight)
        self._current_requests_per_minute = requests_per_minute
        self._max_requests_per_minute = maximum
        self._minimum_requests_per_minute = minimum_requests_per_minute
        self._successes_per_increase = successes_per_increase
        self._successful_requests = 0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0
        self._host_throttle_window_s = host_throttle_window_s
        self._host_throttle_threshold = host_throttle_threshold
        self._host_throttle_events: dict[str, float] = {}

    def defer(self, delay_s: float) -> None:
        """Apply one cooldown to every future request."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + max(0.0, delay_s))

    @property
    def current_requests_per_minute(self) -> float:
        """Return the active process-wide request rate."""
        with self._lock:
            return self._current_requests_per_minute

    def report_success(self) -> None:
        """Gradually increase request pace after a successful request window."""
        with self._lock:
            if self._current_requests_per_minute >= self._max_requests_per_minute:
                self._successful_requests = 0
                return
            self._successful_requests += 1
            if self._successful_requests < self._successes_per_increase:
                return
            self._successful_requests = 0
            self._current_requests_per_minute = min(
                self._max_requests_per_minute,
                self._current_requests_per_minute * 1.25,
            )

    def report_throttled(self, delay_s: float) -> None:
        """Apply a global cooldown and reduce the active request rate."""
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until,
                self._clock() + max(0.0, delay_s),
            )
            self._current_requests_per_minute = max(
                self._minimum_requests_per_minute,
                self._current_requests_per_minute / 2,
            )
            self._successful_requests = 0

    def report_host_throttled(self, host: str, delay_s: float) -> None:
        """Record a per-host throttle; escalate globally only when systemic."""
        with self._lock:
            now = self._clock()
            cutoff = now - self._host_throttle_window_s
            self._host_throttle_events = {
                h: t for h, t in self._host_throttle_events.items() if t > cutoff
            }
            self._host_throttle_events[host] = now
            if len(self._host_throttle_events) < self._host_throttle_threshold:
                return
        self.report_throttled(delay_s)

    def run(self, operation: Callable[[], T]) -> T:
        """Run an operation after acquiring global concurrency and rate capacity."""
        with self._semaphore:
            with self._lock:
                now = self._clock()
                ready_at = max(now, self._next_request_at, self._cooldown_until)
                interval = 60.0 / self._current_requests_per_minute
                self._next_request_at = ready_at + interval
            wait = ready_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            return operation()


_DEFAULT_SCHEDULER = AdaptiveRequestScheduler()


def default_scheduler() -> AdaptiveRequestScheduler:
    return _DEFAULT_SCHEDULER


__all__ = ["AdaptiveRequestScheduler", "default_scheduler"]
