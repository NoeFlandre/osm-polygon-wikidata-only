"""Process-wide scheduling for polite Wikimedia API traffic.

The scheduler is *hierarchical*: one global budget (concurrency cap +
request-rate ceiling) is shared fairly across every Wikimedia host,
while each host keeps independent pacing, cooldown, and throttle
history. A ``429``/``503`` from a single host cools down only that host;
the global rate is reduced only when throttling becomes *systemic*
(several distinct hosts throttled within a bounded window) or when an
explicit global backoff is requested.

Important: per-host pacing (:meth:`pace_host`) happens *before* the
global concurrency permit is acquired, so a host stuck in a long
cooldown can never monopolise the (small) pool of global permits and
starve unrelated healthy hosts.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")

# Length of the rolling window used for "requests/429s in the last
# minute" telemetry. Centralised so the snapshot and pruning agree.
ROLLING_WINDOW_S = 60.0

# Proportional systemic-throttle defaults.  Callers that want the
# active-host-aware decision should pass these (or import them).
# See :meth:`AdaptiveRequestScheduler._systemic_threshold`.
SYSTEMIC_ACTIVE_HOST_WINDOW_S = 60.0
SYSTEMIC_MINIMUM_HOSTS = 5
SYSTEMIC_HOST_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class RequestSchedulerSnapshot:
    """A thread-safe point-in-time view of Wikimedia request-budget usage.

    ``throttle_events`` is a *rolling* count of host throttle responses
    in the last minute, not a cumulative total. ``maximum_requests_per_minute``
    is the configured client-side ceiling, not a guaranteed server allowance.
    """

    requests_last_minute: int
    current_requests_per_minute: float
    maximum_requests_per_minute: float
    utilization_percent: float
    in_flight: int
    max_in_flight: int
    throttle_events: int
    throttled_hosts_last_minute: int
    cooling_down_hosts: int
    cooldown_remaining_s: float


@dataclass
class _HostState:
    """Independent pacing/cooldown/history for one Wikimedia host."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    cooldown_until: float = 0.0
    next_request_at: float = 0.0
    recent_throttles: deque[float] = field(default_factory=lambda: deque[float]())


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
        active_host_window_s: float | None = None,
        minimum_systemic_hosts: int | None = None,
        systemic_host_fraction: float | None = None,
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
        # Proportional systemic threshold: when any of the new parameters
        # are provided, use proportional mode; otherwise fall back to the
        # fixed ``host_throttle_threshold``.
        proportional = (
            active_host_window_s is not None
            or minimum_systemic_hosts is not None
            or systemic_host_fraction is not None
        )
        self._proportional_mode = proportional
        self._active_host_window_s = (
            active_host_window_s if active_host_window_s is not None else 60.0
        )
        self._minimum_systemic_hosts = (
            minimum_systemic_hosts if minimum_systemic_hosts is not None else 5
        )
        self._systemic_host_fraction = (
            systemic_host_fraction if systemic_host_fraction is not None else 0.10
        )
        if self._active_host_window_s <= 0:
            raise ValueError("active_host_window_s must be positive")
        if self._minimum_systemic_hosts < 1:
            raise ValueError("minimum_systemic_hosts must be at least 1")
        if not 0 < self._systemic_host_fraction <= 1.0:
            raise ValueError("systemic_host_fraction must be between 0 (exclusive) and 1")
        self._max_in_flight = max_in_flight
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
        # host -> last throttle time, used for systemic detection.
        self._systemic_host_events: dict[str, float] = {}
        # Rolling timestamps of every host throttle response (telemetry).
        self._global_throttle_times: deque[float] = deque()
        # Per-host independent state.
        self._host_states: dict[str, _HostState] = {}
        self._hosts_lock = threading.Lock()
        self._request_started_at: deque[float] = deque()
        self._in_flight = 0
        # Active host tracking for proportional systemic detection.
        # Maps host -> last activity timestamp. Protected by ``_lock``.
        self._active_host_timestamps: dict[str, float] = {}
        # Monotonic timestamp of the last systemic global reduction. A
        # second escalation within ``host_throttle_window_s`` is suppressed
        # so a flurry of throttles from many hosts does not repeatedly
        # halve the global rate within seconds. Initialised to -inf so
        # the first systemic event is always allowed to fire.
        self._last_systemic_reduction_at: float = float("-inf")

    def defer(self, delay_s: float) -> None:
        """Apply one cooldown to every future request (explicit global backoff)."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self._clock() + max(0.0, delay_s))

    @property
    def max_in_flight(self) -> int:
        """Return the configured process-wide concurrency bound."""
        return self._max_in_flight

    @property
    def current_requests_per_minute(self) -> float:
        """Return the active process-wide request rate."""
        with self._lock:
            return self._current_requests_per_minute

    def pace_host(self, host: str, *, min_interval_s: float = 0.0) -> None:
        """Wait for ``host``'s cooldown and enforce its minimum interval.

        Called *before* acquiring the global concurrency permit so that a
        host in a long cooldown cannot hold a scarce global permit and
        block unrelated hosts. Honours per-host ``Retry-After`` cooldowns
        set by :meth:`report_host_throttled`.

        After waking from the initial sleep, the cooldown is re-checked
        so a 429 that arrived while the request was waiting cannot be
        silently ignored.
        """
        # Record this host as active for proportional threshold.
        with self._lock:
            self._active_host_timestamps[host] = self._clock()
        state = self._host_state(host)
        while True:
            with state.lock:
                now = self._clock()
                ready_at = max(now, state.cooldown_until, state.next_request_at)
                state.next_request_at = ready_at + max(0.0, min_interval_s)
            wait = ready_at - self._clock()
            if wait <= 0:
                return
            self._sleep(wait)
            # Re-check: a 429 (Retry-After) may have been registered
            # while we slept and pushed the cooldown past our wake time.
            with state.lock:
                if state.cooldown_until > self._clock():
                    continue
            return

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
        """Apply an explicit *global* cooldown and halve the active rate.

        Reserved for signals known to be process-wide. Per-host ``429``/``503``
        responses should use :meth:`report_host_throttled` instead, which scopes
        the cooldown to the host and only escalates globally when throttling is
        systemic.
        """
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
        """Record a per-host throttle and escalate globally only when systemic.

        Always cools down ``host`` for ``delay_s`` (honouring ``Retry-After``)
        and records the response in the rolling telemetry. The global rate is
        reduced at most once, only when ``host_throttle_threshold`` *distinct*
        hosts have been throttled within ``host_throttle_window_s`` seconds.

        The systemic decision and the ``_last_systemic_reduction_at`` update
        happen inside a single critical section so that, even with many
        threads reporting distinct hosts at the same instant, only the first
        thread to acquire the lock wins the global reduction.
        """
        now = self._clock()
        delay = max(0.0, delay_s)
        # Per-host cooldown + rolling throttle history.
        state = self._host_state(host)
        with state.lock:
            state.cooldown_until = max(state.cooldown_until, now + delay)
            self._record_recent(state.recent_throttles, now)
        # Rolling global throttle telemetry + atomic systemic decision.
        systemic_apply = False
        with self._lock:
            self._record_recent(self._global_throttle_times, now)
            cutoff = now - self._host_throttle_window_s
            self._systemic_host_events = {
                h: t for h, t in self._systemic_host_events.items() if t > cutoff
            }
            self._systemic_host_events[host] = now
            systemic = (
                len(self._systemic_host_events) >= self._systemic_threshold()
                and now - self._last_systemic_reduction_at > self._host_throttle_window_s
            )
            if systemic:
                # Decision and suppression-timestamp update are one atomic
                # operation: the next contender that acquires this lock will
                # see the fresh ``_last_systemic_reduction_at`` and fail the
                # guard, guaranteeing at most one reduction per window.
                self._last_systemic_reduction_at = now
                systemic_apply = True
        if systemic_apply:
            self._apply_global_throttle(delay_s, count_event=False)

    def _apply_global_throttle(self, delay_s: float, *, count_event: bool) -> None:
        with self._lock:
            if count_event:
                self._record_recent(self._global_throttle_times, self._clock())
            self._cooldown_until = max(
                self._cooldown_until,
                self._clock() + max(0.0, delay_s),
            )
            self._current_requests_per_minute = max(
                self._minimum_requests_per_minute,
                self._current_requests_per_minute / 2,
            )
            self._successful_requests = 0

    def _systemic_threshold(self) -> int:
        """Compute the dynamic systemic threshold based on active host population.

        In proportional mode the threshold scales with the active host count:
        ``min(active, max(minimum_systemic_hosts, ceil(active * fraction)))``.

        For small populations (active ≤ minimum_systemic_hosts) this clamps
        to the total active count, requiring *all* active hosts to be
        throttled before declaring systemic throttling.

        In legacy mode (no proportional parameters provided), the fixed
        ``host_throttle_threshold`` is returned unchanged.

        Must be called while ``self._lock`` is held.
        """
        if not self._proportional_mode:
            return self._host_throttle_threshold
        active = self._active_host_count()
        if active == 0:
            return self._host_throttle_threshold
        return min(
            active,
            max(
                self._minimum_systemic_hosts,
                math.ceil(active * self._systemic_host_fraction),
            ),
        )

    def _active_host_count(self) -> int:
        """Count hosts active within the rolling window.

        Must be called while ``self._lock`` is held.
        """
        now = self._clock()
        cutoff = now - self._active_host_window_s
        self._active_host_timestamps = {
            h: t for h, t in self._active_host_timestamps.items() if t > cutoff
        }
        return len(self._active_host_timestamps)

    def _host_state(self, host: str) -> _HostState:
        with self._hosts_lock:
            state = self._host_states.get(host)
            if state is None:
                state = _HostState()
                self._host_states[host] = state
            return state

    @staticmethod
    def _record_recent(deq: deque[float], now: float) -> None:
        deq.append(now)
        AdaptiveRequestScheduler._prune_recent(deq, now)

    @staticmethod
    def _prune_recent(deq: deque[float], now: float) -> None:
        cutoff = now - ROLLING_WINDOW_S
        while deq and deq[0] < cutoff:
            deq.popleft()

    def snapshot(self) -> RequestSchedulerSnapshot:
        """Return measured traffic and adaptive-budget state for operator logs."""
        with self._lock:
            now = self._clock()
            cutoff = now - ROLLING_WINDOW_S
            while self._request_started_at and self._request_started_at[0] < cutoff:
                self._request_started_at.popleft()
            recent = len(self._request_started_at)
            self._prune_recent(self._global_throttle_times, now)
            throttle_events = len(self._global_throttle_times)
            global_cooldown = max(0.0, self._cooldown_until - now)
            utilization = recent / self._max_requests_per_minute * 100.0
        throttled_hosts = 0
        cooling_down = 0
        with self._hosts_lock:
            host_states = list(self._host_states.values())
        for state in host_states:
            with state.lock:
                self._prune_recent(state.recent_throttles, now)
                if state.recent_throttles:
                    throttled_hosts += 1
                if state.cooldown_until > now:
                    cooling_down += 1
        return RequestSchedulerSnapshot(
            requests_last_minute=recent,
            current_requests_per_minute=self.current_requests_per_minute,
            maximum_requests_per_minute=self._max_requests_per_minute,
            utilization_percent=utilization,
            in_flight=self._in_flight,
            max_in_flight=self._max_in_flight,
            throttle_events=throttle_events,
            throttled_hosts_last_minute=throttled_hosts,
            cooling_down_hosts=cooling_down,
            cooldown_remaining_s=global_cooldown,
        )

    def run(self, operation: Callable[[], T]) -> T:
        """Run an operation after acquiring global concurrency and rate capacity.

        Only the *global* budget is enforced here; per-host pacing and cooldowns
        are handled separately by :meth:`pace_host` so they never hold a global
        permit.
        """
        with self._semaphore:
            with self._lock:
                now = self._clock()
                ready_at = max(now, self._next_request_at, self._cooldown_until)
                interval = 60.0 / self._current_requests_per_minute
                self._next_request_at = ready_at + interval
            wait = ready_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            with self._lock:
                self._request_started_at.append(self._clock())
                self._in_flight += 1
            try:
                return operation()
            finally:
                with self._lock:
                    self._in_flight -= 1


_DEFAULT_SCHEDULER = AdaptiveRequestScheduler(
    active_host_window_s=SYSTEMIC_ACTIVE_HOST_WINDOW_S,
    minimum_systemic_hosts=SYSTEMIC_MINIMUM_HOSTS,
    systemic_host_fraction=SYSTEMIC_HOST_FRACTION,
)


def default_scheduler() -> AdaptiveRequestScheduler:
    return _DEFAULT_SCHEDULER


__all__ = [
    "SYSTEMIC_ACTIVE_HOST_WINDOW_S",
    "SYSTEMIC_HOST_FRACTION",
    "SYSTEMIC_MINIMUM_HOSTS",
    "AdaptiveRequestScheduler",
    "RequestSchedulerSnapshot",
    "default_scheduler",
]
