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


# ---------------------------------------------------------------------------
# Proportional systemic backoff tests (TDD Red)
# ---------------------------------------------------------------------------


def _make_proportional_scheduler(
    now: list[float],
    clock: object,
    sleep: object,
    *,
    requests_per_minute: float = 1200.0,
    active_host_window_s: float = 60.0,
    minimum_systemic_hosts: int = 5,
    systemic_host_fraction: float = 0.10,
    host_throttle_window_s: float = 10.0,
) -> AdaptiveRequestScheduler:
    """Create a scheduler with the new proportional systemic parameters."""
    return AdaptiveRequestScheduler(
        max_in_flight=8,
        requests_per_minute=requests_per_minute,
        max_requests_per_minute=requests_per_minute,
        minimum_requests_per_minute=200.0,
        host_throttle_window_s=host_throttle_window_s,
        active_host_window_s=active_host_window_s,
        minimum_systemic_hosts=minimum_systemic_hosts,
        systemic_host_fraction=systemic_host_fraction,
        clock=clock,
        sleep=sleep,
    )


def _activate_hosts(
    scheduler: AdaptiveRequestScheduler,
    hosts: tuple[str, ...],
    *,
    min_interval_s: float = 0.0,
) -> None:
    """Pace all hosts so they register as active."""
    for host in hosts:
        scheduler.pace_host(host, min_interval_s=min_interval_s)


def test_proportional_195_hosts_3_throttled_no_global_reduction() -> None:
    """With 195 active hosts, throttling 3 must NOT reduce the global rate."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    for i in range(3):
        scheduler.report_host_throttled(hosts[i], 2.0)

    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_195_hosts_7_throttled_no_global_reduction() -> None:
    """With 195 active hosts, throttling 7 must NOT reduce the global rate."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    for i in range(7):
        scheduler.report_host_throttled(hosts[i], 2.0)

    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_every_throttled_host_gets_retry_after_cooldown() -> None:
    """Each throttled host must receive its own Retry-After cooldown."""
    now, sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    scheduler.report_host_throttled(hosts[0], 15.0)
    scheduler.report_host_throttled(hosts[1], 30.0)

    # Host[0] sleeps its full 15s cooldown.
    scheduler.pace_host(hosts[0])
    assert sleeps[-1] == pytest.approx(15.0)
    # Host[1] had a 30s cooldown set at t=0. After host[0]'s sleep
    # advanced the clock to t=15, the remaining cooldown is 15s.
    scheduler.pace_host(hosts[1])
    assert sleeps[-1] == pytest.approx(15.0)

    # Rate is unchanged (only 2 of 195).
    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_healthy_hosts_productive_while_bad_cool_down() -> None:
    """Healthy hosts must keep working while throttled hosts are cooling down."""
    now, sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    # Throttle 5 hosts with large cooldowns.
    for i in range(5):
        scheduler.report_host_throttled(hosts[i], 60.0)

    # Healthy hosts must not be delayed by throttled host cooldowns.
    before_len = len(sleeps)
    scheduler.pace_host(hosts[100])
    assert len(sleeps) == before_len  # no sleep for a healthy host

    # Rate unchanged (5 < ceil(195 * 0.10) = 20).
    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_threshold_triggers_exactly_one_global_reduction() -> None:
    """Reaching the proportional threshold triggers exactly one global reduction.

    For 195 active hosts with 10% fraction and min 5:
    threshold = min(195, max(5, ceil(195 * 0.10))) = min(195, max(5, 20)) = 20
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    # Throttle exactly 20 distinct hosts (= threshold).
    for i in range(20):
        scheduler.report_host_throttled(hosts[i], 1.0)

    assert scheduler.current_requests_per_minute == 600.0  # exactly one halving


def test_proportional_one_below_threshold_no_reduction() -> None:
    """One fewer than the threshold must NOT trigger global reduction.

    threshold = 20 for 195 hosts; throttling 19 should not reduce.
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    for i in range(19):
        scheduler.report_host_throttled(hosts[i], 1.0)

    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_small_population_all_throttled() -> None:
    """Small population: 3 hosts, all 3 throttled → one global reduction.

    threshold = min(3, max(5, ceil(3 * 0.10))) = min(3, 5) = 3
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = ("a.wikipedia.org", "b.wikipedia.org", "c.wikipedia.org")
    _activate_hosts(scheduler, hosts)

    for host in hosts:
        scheduler.report_host_throttled(host, 1.0)

    assert scheduler.current_requests_per_minute == 600.0


def test_proportional_small_population_partial_no_reduction() -> None:
    """Small population: 3 hosts, 2 throttled → no global reduction.

    threshold = min(3, max(5, ceil(3 * 0.10))) = min(3, 5) = 3
    Only 2 < 3 so no reduction.
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = ("a.wikipedia.org", "b.wikipedia.org", "c.wikipedia.org")
    _activate_hosts(scheduler, hosts)

    scheduler.report_host_throttled(hosts[0], 1.0)
    scheduler.report_host_throttled(hosts[1], 1.0)

    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_inactive_hosts_expire_from_denominator() -> None:
    """Hosts inactive beyond the active-host window must expire from the denominator.

    Start with 195 active hosts (threshold=20). After the window elapses
    with only 5 hosts refreshed, threshold = min(5, max(5, 1)) = 5.
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep, active_host_window_s=60.0)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    # Advance past the active window.
    now[0] += 61.0

    # Only 5 hosts are refreshed.
    live_hosts = hosts[:5]
    _activate_hosts(scheduler, live_hosts)

    # threshold = min(5, max(5, ceil(5 * 0.10))) = min(5, 5) = 5
    # Throttle all 5 → global reduction.
    for host in live_hosts:
        scheduler.report_host_throttled(host, 1.0)

    assert scheduler.current_requests_per_minute == 600.0


def test_proportional_host_reactivation_enters_denominator() -> None:
    """A host becoming active again re-enters the denominator.

    With 5 active hosts (threshold=5), adding a 6th raises threshold:
    min(6, max(5, ceil(6 * 0.10))) = min(6, 5) = 5
    Still 5. But with 50 active → min(50, max(5, 5)) = 5 ... so
    let's test with 50 hosts: threshold=max(5, ceil(50*0.10))=5.
    Actually for clean test: 60 hosts → ceil(60*0.10)=6 → threshold=6.
    """
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep, active_host_window_s=60.0)
    # Start with 10 active hosts → threshold = max(5, ceil(10*0.10)) = max(5,1) = 5
    hosts_initial = tuple(f"h{i}.wikipedia.org" for i in range(10))
    _activate_hosts(scheduler, hosts_initial)

    # Advance past window so initial hosts expire.
    now[0] += 61.0

    # Re-activate 60 hosts (some new, some old) → threshold = max(5, ceil(60*0.10)) = 6
    hosts_new = tuple(f"h{i}.wikipedia.org" for i in range(60))
    _activate_hosts(scheduler, hosts_new)

    # Throttle exactly 5 → below threshold (6), no reduction.
    for i in range(5):
        scheduler.report_host_throttled(hosts_new[i], 1.0)
    assert scheduler.current_requests_per_minute == 1200.0

    # Throttle one more (6th) → meets threshold, one reduction.
    scheduler.report_host_throttled(hosts_new[5], 1.0)
    assert scheduler.current_requests_per_minute == 600.0


def test_proportional_duplicate_throttles_count_once() -> None:
    """Duplicate throttles from one host count once toward systemic detection."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    # Throttle the same host 20 times — should count as 1 distinct host.
    for _ in range(20):
        scheduler.report_host_throttled(hosts[0], 1.0)

    assert scheduler.current_requests_per_minute == 1200.0


def test_proportional_concurrent_reports_at_threshold_at_most_one_reduction() -> None:
    """Concurrent reports at the threshold cause at most one global reduction."""
    for _ in range(25):
        now, _sleeps, clock, sleep = _fake_clock()
        scheduler = _make_proportional_scheduler(now, clock, sleep)
        hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
        _activate_hosts(scheduler, hosts)

        # Use real threading to race 25 distinct host throttles (threshold=20).
        barrier = threading.Barrier(25)

        def report(host: str) -> None:
            barrier.wait(timeout=5)
            scheduler.report_host_throttled(host, 1.0)

        threads = [threading.Thread(target=report, args=(hosts[i],)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Exactly one halving: 1200 → 600, not 300 or lower.
        assert scheduler.current_requests_per_minute == 600.0


def test_proportional_suppression_window_unchanged() -> None:
    """Suppression window prevents a second reduction within the same window."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep, host_throttle_window_s=10.0)
    hosts = tuple(f"h{i}.wikipedia.org" for i in range(195))
    _activate_hosts(scheduler, hosts)

    # First wave: 20 throttled → one reduction (1200 → 600).
    for i in range(20):
        scheduler.report_host_throttled(hosts[i], 1.0)
    assert scheduler.current_requests_per_minute == 600.0

    # Second wave (within the same 10s window): additional throttles
    # must NOT cause a second reduction.
    for i in range(20, 40):
        scheduler.report_host_throttled(hosts[i], 1.0)
    assert scheduler.current_requests_per_minute == 600.0

    # After the suppression window passes, a new wave CAN reduce.
    now[0] += 11.0
    _activate_hosts(scheduler, hosts)  # refresh active set
    for i in range(40, 60):
        scheduler.report_host_throttled(hosts[i], 1.0)
    assert scheduler.current_requests_per_minute == 300.0


def test_proportional_explicit_report_throttled_still_reduces() -> None:
    """The process-wide report_throttled still unconditionally reduces."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)

    scheduler.report_throttled(5.0)
    assert scheduler.current_requests_per_minute == 600.0


def test_proportional_existing_ceiling_and_inflight_enforced() -> None:
    """Existing ceiling and in-flight limits remain enforced."""
    now, _sleeps, clock, sleep = _fake_clock()
    scheduler = _make_proportional_scheduler(now, clock, sleep)

    assert scheduler.max_in_flight == 8
    # Run operations and verify the rate doesn't exceed the configured max.
    for _ in range(10):
        scheduler.run(lambda: None)

    snapshot = scheduler.snapshot()
    assert snapshot.maximum_requests_per_minute == 1200.0
    assert snapshot.max_in_flight == 8


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
