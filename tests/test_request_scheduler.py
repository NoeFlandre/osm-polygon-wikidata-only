"""Hierarchical scheduling: global budget + independent per-host control.

These tests pin the behaviour required to keep healthy Wikimedia hosts
productive when an individual host returns a ``429``/``503``. They use
injected deterministic clocks and sleeps so no real time passes.
"""

from __future__ import annotations

import threading

import pytest

from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler


def _fake_clock() -> tuple[list[float], list[float], object, object]:
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    return now, sleeps, clock, sleep


def test_pace_host_enforces_per_host_minimum_interval() -> None:
    _now, sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000, clock=clock, sleep=sleep)

    scheduler.pace_host("en.wikipedia.org", min_interval_s=0.1)
    scheduler.pace_host("en.wikipedia.org", min_interval_s=0.1)

    assert sleeps == [pytest.approx(0.1)]


def test_pace_host_does_not_delay_an_unrelated_host() -> None:
    _now, sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000, clock=clock, sleep=sleep)

    scheduler.pace_host("en.wikipedia.org", min_interval_s=0.1)
    scheduler.pace_host("fr.wikipedia.org", min_interval_s=0.1)

    # Different host: no inherited spacing from en.wikipedia.org.
    assert sleeps == []


def test_report_host_throttled_delays_only_that_host() -> None:
    """A 429 from one host must cool down only that host."""
    _now, sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        clock=clock,
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 30.0)

    # fr.wikipedia.org is cooled down ...
    scheduler.pace_host("fr.wikipedia.org")
    assert sleeps == [30.0]
    # ... but a healthy host is unaffected.
    before = list(sleeps)
    scheduler.pace_host("de.wikipedia.org")
    assert sleeps == before
    # And the global rate was not reduced.
    assert scheduler.current_requests_per_minute == 200


def test_repeated_single_host_throttles_do_not_halve_global_rate() -> None:
    _now, _sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        host_throttle_threshold=3,
        clock=clock,
        sleep=sleep,
    )

    for _ in range(5):
        scheduler.report_host_throttled("fr.wikipedia.org", 1.0)

    assert scheduler.current_requests_per_minute == 200


def test_pace_host_cooldown_uses_retry_after_delay() -> None:
    _now, sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000, clock=clock, sleep=sleep)

    scheduler.report_host_throttled("es.wikipedia.org", 17.0)
    scheduler.pace_host("es.wikipedia.org")

    assert sleeps == [17.0]


def test_host_cooldown_does_not_hold_the_global_permit() -> None:
    """Per-host pacing happens before the global semaphore is acquired."""
    _now, sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=1,
        requests_per_minute=100_000,
        host_throttle_threshold=10,
        clock=clock,
        sleep=sleep,
    )

    # Cool one host for a long time. Because this is a single host it
    # must NOT trigger a global backoff.
    scheduler.report_host_throttled("a.wikipedia.org", 1000.0)

    # An unrelated host must be able to acquire the single global
    # permit immediately, without waiting for a.wikipedia.org's cooldown.
    scheduler.pace_host("b.wikipedia.org")
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == []


def test_snapshot_reports_max_in_flight_and_rolling_throttle_metrics() -> None:
    _now, _sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=8,
        requests_per_minute=1200,
        max_requests_per_minute=1200,
        minimum_requests_per_minute=200,
        host_throttle_window_s=10.0,
        host_throttle_threshold=10,
        clock=clock,
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 5.0)
    scheduler.report_host_throttled("fr.wikipedia.org", 5.0)
    scheduler.report_host_throttled("de.wikipedia.org", 5.0)

    snapshot = scheduler.snapshot()

    assert snapshot.max_in_flight == 8
    # Three throttle responses, all within the rolling window.
    assert snapshot.throttle_events == 3
    assert snapshot.throttled_hosts_last_minute == 2
    # Both hosts are still cooling down.
    assert snapshot.cooling_down_hosts == 2


def test_rolling_throttle_metrics_expire_after_window() -> None:
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=1200,
        max_requests_per_minute=1200,
        minimum_requests_per_minute=200,
        host_throttle_window_s=10.0,
        host_throttle_threshold=10,
        clock=clock,
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 5.0)
    now[0] += 61.0  # past the 60s rolling window

    snapshot = scheduler.snapshot()

    assert snapshot.throttle_events == 0
    assert snapshot.throttled_hosts_last_minute == 0
    # Cooldown (5s) has also expired.
    assert snapshot.cooling_down_hosts == 0


def test_operation_exception_releases_global_permit() -> None:
    scheduler = AdaptiveRequestScheduler(max_in_flight=1, requests_per_minute=100_000)

    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        scheduler.run(boom)

    completed = threading.Event()

    def work() -> str:
        completed.set()
        return "ok"

    assert scheduler.run(work) == "ok"
    assert completed.is_set()


def test_simultaneous_distinct_host_throttles_trigger_at_most_one_global_reduction() -> None:
    """Race: many threads reporting distinct hosts at the same instant.

    Without atomicity between the systemic decision and the
    ``_last_systemic_reduction_at`` update, several threads can each
    observe the threshold crossed and each halve the global rate.
    The fix collapses the decision and the timestamp update into one
    critical section so only the first thread wins. The assertion is
    repeated across many iterations so a fluky timing on the racy
    version is caught.
    """
    for _ in range(25):
        scheduler = AdaptiveRequestScheduler(
            requests_per_minute=1200,
            max_requests_per_minute=1200,
            minimum_requests_per_minute=200,
            host_throttle_threshold=3,
            host_throttle_window_s=10.0,
        )
        barrier = threading.Barrier(8)
        hosts = tuple(f"h{i}.wikipedia.org" for i in range(8))

        def report(host: str) -> None:
            barrier.wait(timeout=5)
            scheduler.report_host_throttled(host, 1.0)

        threads = [threading.Thread(target=report, args=(h,)) for h in hosts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        # Exactly one halving: 1200 -> 600, not multiple halvings down to 200.
        assert scheduler.current_requests_per_minute == 600.0


def test_pace_host_rechecks_cooldown_after_waking() -> None:
    """A 429 introduced while pace_host is sleeping must extend the wait.

    Without the re-check, a request that already passed its initial
    cooldown check could wake up after a fresh ``Retry-After`` and
    proceed without honoring it.
    """
    now = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=100_000, clock=clock, sleep=lambda s: sleeps.append(s)
    )
    # Start a 5s cooldown so the first pace_host sleeps.
    scheduler.report_host_throttled("a.wikipedia.org", 5.0)

    introduced = [False]

    def sleep_with_extension(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds
        # While pace_host is "sleeping" the original 5s cooldown, extend
        # it to 30s total so the request must wait the full extension.
        if not introduced[0]:
            introduced[0] = True
            scheduler.report_host_throttled("a.wikipedia.org", 30.0)

    # Override the scheduler's sleep for this test only.
    object.__setattr__(scheduler, "_sleep", sleep_with_extension)

    scheduler.pace_host("a.wikipedia.org")

    # The first sleep honored the initial 5s; the re-check then caught
    # the extension (5s elapsed + 30s delay = cooldown ends at t=35) and
    # slept an additional 30s to reach t=35.
    assert sleeps[0] == 5.0
    assert sum(sleeps) == 35.0


def test_global_recovery_is_gradual_and_bounded() -> None:
    _now, _sleeps, clock, sleep = _fake_clock()
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        successes_per_increase=1,
        host_throttle_threshold=3,
        clock=clock,
        sleep=sleep,
    )

    # Force a systemic global backoff.
    scheduler.report_host_throttled("a.wikipedia.org", 1.0)
    scheduler.report_host_throttled("b.wikipedia.org", 1.0)
    scheduler.report_host_throttled("c.wikipedia.org", 1.0)
    assert scheduler.current_requests_per_minute == 100

    scheduler.report_success()
    assert scheduler.current_requests_per_minute == 125
    scheduler.report_success()
    assert scheduler.current_requests_per_minute == pytest.approx(156.25)
    # Never exceeds the configured ceiling.
    for _ in range(50):
        scheduler.report_success()
    assert scheduler.current_requests_per_minute == 400


def test_concurrent_snapshots_are_thread_safe() -> None:
    scheduler = AdaptiveRequestScheduler(max_in_flight=4, requests_per_minute=100_000)
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            scheduler.run(lambda: None)
            scheduler.report_host_throttled("fr.wikipedia.org", 0.001)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for _ in range(200):
        scheduler.snapshot()
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
