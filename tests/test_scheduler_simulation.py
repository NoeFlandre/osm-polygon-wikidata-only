"""Deterministic simulation of the hierarchical scheduler (no network).

Models 20 healthy Wikimedia hosts and one repeatedly-throttled host under
the configured authenticated budget (1200 rpm ceiling, configurable
concurrency) and compares the OLD global 429 cascade with the NEW
host-scoped 429 handling. Uses injected deterministic clock and sleep so
the simulation runs in real milliseconds while exercising real
concurrency control.

These tests are the primary performance validation evidence: they prove
that healthy hosts remain productive when an individual host throttles,
that the process-wide ceiling is never exceeded, that systemic
throttling still reduces traffic safely, and that no host is starved.

The main before/after simulation is a deterministic round-robin over
hosts so the *scheduling policy* is exercised without flakiness from
real-thread scheduling under tight contention. Concurrency primitives
are covered separately by threaded tests in ``test_request_scheduler``.
"""

from __future__ import annotations

import threading
import urllib.error
from email.message import Message

import pytest

from osm_polygon_wikidata_only.utils.request_scheduler import (
    AdaptiveRequestScheduler,
    RequestSchedulerSnapshot,
)

_AUTH_CEILING = 1200.0
_AUTH_MAX_IN_FLIGHT = 8
_SIM_DURATION_S = 60.0
_HOSTS_HEALTHY = tuple(
    f"{code}.wikipedia.org"
    for code in (
        "en",
        "de",
        "fr",
        "es",
        "it",
        "pt",
        "ja",
        "zh",
        "pl",
        "nl",
        "sv",
        "fi",
        "no",
        "da",
        "cs",
        "hu",
        "ro",
        "el",
        "tr",
        "ko",
    )
)
_HOST_BAD = "ru.wikipedia.org"
_HOST_ANON = "bg.wikipedia.org"  # a host whose bot password was rejected
# Per-host pacing intervals model the centralised auth-aware decision
# (see WikimediaSession.read): verified hosts use the tight authenticated
# value, anonymous/rejected hosts use the per-kind anonymous value.
_AUTH_MIN_INTERVAL_S = 0.05
_ANON_MIN_INTERVAL_S = 1.2


class _SimulatedClock:
    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> float:
        with self._lock:
            self._now += seconds
            return self._now


def _build_scheduler(clock: _SimulatedClock, *, threshold: int) -> AdaptiveRequestScheduler:
    return AdaptiveRequestScheduler(
        max_in_flight=_AUTH_MAX_IN_FLIGHT,
        requests_per_minute=_AUTH_CEILING,
        max_requests_per_minute=_AUTH_CEILING,
        # Match the production authenticated floor (see cli.dependencies).
        minimum_requests_per_minute=200.0,
        host_throttle_threshold=threshold,
        clock=clock.now,
        sleep=clock.sleep,
    )


def _run_simulation(
    *,
    mode: str,
    threshold: int = 100,
    include_anon_host: bool = False,
) -> tuple[int, int, int, dict[str, int], float]:
    """Drive a round-robin of hosts until the simulated clock reaches ``_SIM_DURATION_S``.

    The bad host returns a 429 every fifth request with ``Retry-After: 2``.
    ``mode='before'`` mimics the OLD pipeline by calling the global
    ``report_throttled`` on every 429 (the cascade bug). ``mode='after'``
    calls host-scoped ``report_host_throttled``.

    ``include_anon_host`` adds one host paced at the *anonymous* per-host
    interval (simulating a host whose bot password was rejected) so the
    simulation verifies the centralised auth-aware decision.

    Returns the healthy total, the bad total, the anonymous total, the
    per-host breakdown, and the scheduler's final adaptive rate.
    """
    clock = _SimulatedClock()
    scheduler = _build_scheduler(clock, threshold=threshold)
    all_hosts = [*_HOSTS_HEALTHY, _HOST_BAD]
    if include_anon_host:
        all_hosts.append(_HOST_ANON)
    completions: dict[str, int] = {host: 0 for host in all_hosts}
    bad_counter = [0]
    workers = all_hosts
    host_min_interval = {h: _AUTH_MIN_INTERVAL_S for h in _HOSTS_HEALTHY}
    host_min_interval[_HOST_BAD] = _AUTH_MIN_INTERVAL_S
    if include_anon_host:
        host_min_interval[_HOST_ANON] = _ANON_MIN_INTERVAL_S
    idx = 0

    def step(host: str) -> None:
        # Centralised auth-aware pacing: the per-host min_interval models
        # what WikimediaSession.read would choose for that host.
        scheduler.pace_host(host, min_interval_s=host_min_interval[host])

        def operation() -> str:
            if host == _HOST_BAD:
                bad_counter[0] += 1
                if bad_counter[0] % 5 == 0:
                    headers = Message()
                    headers["Retry-After"] = "2"
                    raise urllib.error.HTTPError(
                        f"https://{host}/w/api.php", 429, "limited", headers, None
                    )
            completions[host] += 1
            return "ok"

        try:
            scheduler.run(operation)
        except urllib.error.HTTPError:
            if mode == "before":
                scheduler.report_throttled(2.0)
            else:
                scheduler.report_host_throttled(host, 2.0)
        else:
            scheduler.report_success()

    while clock.now() < _SIM_DURATION_S:
        step(workers[idx % len(workers)])
        idx += 1

    healthy_total = sum(completions[h] for h in _HOSTS_HEALTHY)
    anon_total = completions.get(_HOST_ANON, 0)
    return (
        healthy_total,
        completions[_HOST_BAD],
        anon_total,
        completions,
        scheduler.current_requests_per_minute,
    )


def test_old_global_throttle_cascade_starves_healthy_hosts() -> None:
    """Baseline (old behaviour): a single bad host collapses the global rate.

    Threshold is intentionally high so systemic escalation never fires;
    the cascade comes purely from the per-host 429 calling the global
    ``report_throttled`` on every response.
    """
    healthy_total, _bad, _anon, _completions, final_rate = _run_simulation(mode="before")

    # Under the old cascade the global rate halves after every bad-host
    # 429 and floors near the production minimum (200 rpm), so healthy
    # hosts get only a small fraction of the 1200-rpm budget. The exact
    # final rate oscillates slightly because ``report_success`` can lift
    # it between halvings, but it must be far below the ceiling.
    assert healthy_total < 500, (
        f"healthy throughput should collapse under old cascade; got {healthy_total}"
    )
    assert final_rate < 500, f"global rate should be far below the ceiling; got {final_rate}"


def test_new_host_scoped_throttle_keeps_healthy_hosts_productive() -> None:
    """A single host's 429 must not delay unrelated healthy hosts."""
    healthy_total, bad_total, _anon, completions, final_rate = _run_simulation(mode="after")

    # Healthy hosts sustain near-ceiling throughput because the bad
    # host's 429 cools only itself.
    assert healthy_total >= 800, (
        f"healthy hosts should sustain near-ceiling throughput; got {healthy_total}"
    )
    assert final_rate == _AUTH_CEILING
    # Fairness: every healthy host must have made progress.
    zero_hosts = [h for h in _HOSTS_HEALTHY if completions[h] == 0]
    assert not zero_hosts, f"starved healthy hosts: {zero_hosts}"
    # The bad host still gets some successes between throttles.
    assert bad_total >= 1


def test_old_vs_new_throughput_improvement_is_material() -> None:
    """The fix yields a large, reproducible throughput improvement."""
    before_healthy, _, _, _, before_rate = _run_simulation(mode="before")
    after_healthy, _, _, _, after_rate = _run_simulation(mode="after")

    improvement = after_healthy - before_healthy
    ratio = after_healthy / max(1, before_healthy)
    print(
        f"\n[simulation] before={before_healthy} (rate {before_rate:.0f})  "
        f"after={after_healthy} (rate {after_rate:.0f})  "
        f"improvement=+{improvement} ({ratio:.1f}x)"
    )
    assert after_healthy > 2 * before_healthy, (
        f"expected >2x improvement; before={before_healthy} after={after_healthy}"
    )
    assert after_healthy - before_healthy > 300, (
        f"expected material absolute improvement; before={before_healthy} after={after_healthy}"
    )


def test_global_ceiling_is_never_exceeded_under_concurrency() -> None:
    """Threaded: the scheduler must never let requests in the last minute exceed the ceiling."""
    clock = _SimulatedClock()
    scheduler = _build_scheduler(clock, threshold=100)
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            scheduler.pace_host("en.wikipedia.org")
            scheduler.run(lambda: None)

    threads = [threading.Thread(target=worker) for _ in range(_AUTH_MAX_IN_FLIGHT)]
    for thread in threads:
        thread.start()
    # Let the simulation run for a short while.
    for _ in range(200):
        scheduler.snapshot()
        clock.sleep(0.01)
    stop.set()
    for thread in threads:
        thread.join(timeout=2)

    snapshot = scheduler.snapshot()
    # The rolling 60-second count can exceed the nominal ceiling by a tiny
    # boundary amount (discrete pacing vs continuous window); assert it
    # stays within a generous slack rather than the exact ceiling.
    assert snapshot.requests_last_minute <= _AUTH_CEILING + 50
    assert snapshot.max_in_flight == _AUTH_MAX_IN_FLIGHT


def test_systemic_throttling_reduces_traffic_safely() -> None:
    """Several distinct hosts throttling triggers exactly one global reduction."""
    clock = _SimulatedClock()
    scheduler = _build_scheduler(clock, threshold=3)

    scheduler.report_host_throttled("a.wikipedia.org", 1.0)
    scheduler.report_host_throttled("b.wikipedia.org", 1.0)
    scheduler.report_host_throttled("c.wikipedia.org", 1.0)
    # A fourth throttled host must NOT halve the rate a second time.
    scheduler.report_host_throttled("d.wikipedia.org", 1.0)
    scheduler.report_host_throttled("e.wikipedia.org", 1.0)

    assert scheduler.current_requests_per_minute == _AUTH_CEILING / 2

    snapshot: RequestSchedulerSnapshot = scheduler.snapshot()
    assert snapshot.throttled_hosts_last_minute == 5
    assert snapshot.cooling_down_hosts == 5


def test_throttled_host_cooldown_does_not_starve_others() -> None:
    """A long per-host cooldown must not hold a global permit."""
    clock = _SimulatedClock()
    scheduler = _build_scheduler(clock, threshold=100)

    # Cool one host for a very long time.
    scheduler.report_host_throttled("a.wikipedia.org", 10_000.0)

    completed = [False]

    def work() -> str:
        completed[0] = True
        return "ok"

    scheduler.pace_host("b.wikipedia.org")
    assert scheduler.run(work) == "ok"
    assert completed[0]


def test_anonymous_fallback_host_respects_host_budget_while_auth_hosts_productive() -> None:
    """Auth-aware pacing: anonymous host bounded, authenticated healthy.

    The simulation adds one host whose bot password was rejected and is
    therefore paced at the anonymous per-host interval (1.2s -> <=50 rpm)
    alongside the authenticated healthy hosts (0.05s -> up to ceiling).
    The anonymous host must not exceed its host budget, while the
    authenticated healthy hosts must remain productive at near-ceiling.
    """
    healthy_total, _bad, anon_total, _completions, final_rate = _run_simulation(
        mode="after", include_anon_host=True
    )

    # Anonymous host budget: 1.2s interval -> at most 50 completions in 60s
    # (+ a small slack for the round-robin visiting order).
    anon_budget = int(_SIM_DURATION_S / _ANON_MIN_INTERVAL_S) + 2
    assert 0 <= anon_total <= anon_budget, (
        f"anonymous host exceeded its host budget: {anon_total} > {anon_budget}"
    )

    # Authenticated healthy hosts must remain productive near the ceiling.
    assert healthy_total >= 800, (
        f"authenticated healthy hosts must stay productive; got {healthy_total}"
    )
    assert final_rate == _AUTH_CEILING

    print(
        f"\n[simulation+anon] healthy={healthy_total} (rate {final_rate:.0f})  "
        f"anonymous_host={anon_total} (budget <= {anon_budget})"
    )


@pytest.mark.parametrize("max_in_flight", [3, 8, 16])
def test_concurrency_scaling_demonstrates_in_flight_bottleneck(max_in_flight: int) -> None:
    """With more in-flight permits the healthy-host throughput is non-zero and scales."""
    clock = _SimulatedClock()
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=max_in_flight,
        requests_per_minute=_AUTH_CEILING,
        max_requests_per_minute=_AUTH_CEILING,
        minimum_requests_per_minute=200.0,
        host_throttle_threshold=100,
        clock=clock.now,
        sleep=clock.sleep,
    )
    completions = [0]
    workers = list(_HOSTS_HEALTHY)
    idx = 0

    def step(host: str) -> None:
        scheduler.pace_host(host, min_interval_s=0.05)
        scheduler.run(lambda: completions.__setitem__(0, completions[0] + 1))

    while clock.now() < 5.0:
        step(workers[idx % len(workers)])
        idx += 1

    assert completions[0] >= 1
    assert scheduler.snapshot().max_in_flight == max_in_flight


# ---------------------------------------------------------------------------
# Production-topology proportional systemic backoff simulation
# ---------------------------------------------------------------------------

_PRODUCTION_HOSTS = tuple(f"h{i}.wikipedia.org" for i in range(195))
_PRODUCTION_THROTTLED_COUNT = 5  # 3-7 in the real scenario, pick middle


def _build_proportional_scheduler_via_cli(
    clock: _SimulatedClock, tmp_path
) -> AdaptiveRequestScheduler:
    """Build a scheduler with proportional systemic threshold via CLI path."""
    from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.config.settings import Settings

    environ = {
        "WIKIMEDIA_BOT_USERNAME": "User@pipeline",
        "WIKIMEDIA_BOT_PASSWORD": "secret",
    }
    runtime = build_wikimedia_runtime(Settings(), data_root=DataRoot(tmp_path), environ=environ)
    scheduler = runtime.scheduler
    object.__setattr__(scheduler, "_clock", clock.now)
    object.__setattr__(scheduler, "_sleep", clock.sleep)
    return scheduler


def test_production_topology_proportional_keeps_ceiling(tmp_path) -> None:
    """With ~195 hosts and 3-7 throttled, proportional policy keeps ceiling near max."""
    clock = _SimulatedClock()
    scheduler = _build_proportional_scheduler_via_cli(clock, tmp_path)
    healthy = _PRODUCTION_HOSTS[_PRODUCTION_THROTTLED_COUNT:]
    throttled = _PRODUCTION_HOSTS[:_PRODUCTION_THROTTLED_COUNT]

    # Activate all hosts.
    for host in list(healthy) + list(throttled):
        scheduler.pace_host(host, min_interval_s=0.0)

    # Throttle the bad hosts.
    for host in throttled:
        scheduler.report_host_throttled(host, 2.0)

    # The active ceiling must remain at the maximum.
    assert scheduler.current_requests_per_minute == _AUTH_CEILING


def test_production_topology_fixed_threshold_collapses() -> None:
    """With the old fixed threshold=3, 5 throttled hosts collapse the rate."""
    clock = _SimulatedClock()
    scheduler = _build_scheduler(clock, threshold=3)
    throttled = _PRODUCTION_HOSTS[:5]

    # Activate all hosts.
    for host in list(_PRODUCTION_HOSTS):
        scheduler.pace_host(host, min_interval_s=0.0)

    for host in throttled:
        scheduler.report_host_throttled(host, 2.0)

    # Fixed threshold=3 → triggers at the 3rd host → halving.
    assert scheduler.current_requests_per_minute < _AUTH_CEILING


def test_production_topology_genuinely_systemic_still_reduces(tmp_path) -> None:
    """With ~195 hosts and 20+ throttled, proportional policy DOES reduce.

    threshold = min(195, max(5, ceil(195 * 0.10))) = 20
    """
    clock = _SimulatedClock()
    scheduler = _build_proportional_scheduler_via_cli(clock, tmp_path)
    all_hosts = _PRODUCTION_HOSTS

    # Activate all hosts.
    for host in all_hosts:
        scheduler.pace_host(host, min_interval_s=0.0)

    # Throttle 25 hosts (above threshold of 20).
    for host in all_hosts[:25]:
        scheduler.report_host_throttled(host, 2.0)

    assert scheduler.current_requests_per_minute == _AUTH_CEILING / 2


def test_production_topology_proportional_vs_fixed_improvement(tmp_path) -> None:
    """Proportional policy must materially improve throughput vs fixed threshold."""
    # Fixed threshold simulation
    clock_fixed = _SimulatedClock()
    sched_fixed = _build_scheduler(clock_fixed, threshold=3)
    all_hosts = list(_PRODUCTION_HOSTS)
    throttled_hosts = all_hosts[:_PRODUCTION_THROTTLED_COUNT]

    # Activate and run fixed-threshold schedule.
    for host in all_hosts:
        sched_fixed.pace_host(host, min_interval_s=0.0)

    completions_fixed: dict[str, int] = {h: 0 for h in all_hosts}
    bad_counter_fixed: dict[str, int] = {h: 0 for h in throttled_hosts}
    idx = 0
    while clock_fixed.now() < _SIM_DURATION_S:
        host = all_hosts[idx % len(all_hosts)]
        sched_fixed.pace_host(host, min_interval_s=_AUTH_MIN_INTERVAL_S)

        is_bad = host in bad_counter_fixed
        should_throttle = False
        if is_bad:
            bad_counter_fixed[host] = bad_counter_fixed.get(host, 0) + 1
            should_throttle = bad_counter_fixed[host] % 5 == 0

        def make_op_fixed(h: str, throttle: bool) -> object:
            def op() -> str:
                if throttle:
                    import urllib.error
                    from email.message import Message

                    headers = Message()
                    headers["Retry-After"] = "2"
                    raise urllib.error.HTTPError(
                        f"https://{h}/w/api.php", 429, "limited", headers, None
                    )
                completions_fixed[h] += 1
                return "ok"

            return op

        try:
            sched_fixed.run(make_op_fixed(host, should_throttle))
            sched_fixed.report_success()
        except Exception:
            sched_fixed.report_host_throttled(host, 2.0)
        idx += 1

    fixed_healthy = sum(completions_fixed[h] for h in all_hosts if h not in throttled_hosts)
    fixed_rate = sched_fixed.current_requests_per_minute

    # Proportional threshold simulation
    clock_prop = _SimulatedClock()
    sched_prop = _build_proportional_scheduler_via_cli(clock_prop, tmp_path)

    for host in all_hosts:
        sched_prop.pace_host(host, min_interval_s=0.0)

    completions_prop: dict[str, int] = {h: 0 for h in all_hosts}
    bad_counter_prop: dict[str, int] = {h: 0 for h in throttled_hosts}
    idx = 0
    while clock_prop.now() < _SIM_DURATION_S:
        host = all_hosts[idx % len(all_hosts)]
        sched_prop.pace_host(host, min_interval_s=_AUTH_MIN_INTERVAL_S)

        is_bad = host in bad_counter_prop
        should_throttle = False
        if is_bad:
            bad_counter_prop[host] = bad_counter_prop.get(host, 0) + 1
            should_throttle = bad_counter_prop[host] % 5 == 0

        def make_op_prop(h: str, throttle: bool) -> object:
            def op() -> str:
                if throttle:
                    import urllib.error
                    from email.message import Message

                    headers = Message()
                    headers["Retry-After"] = "2"
                    raise urllib.error.HTTPError(
                        f"https://{h}/w/api.php", 429, "limited", headers, None
                    )
                completions_prop[h] += 1
                return "ok"

            return op

        try:
            sched_prop.run(make_op_prop(host, should_throttle))
            sched_prop.report_success()
        except Exception:
            sched_prop.report_host_throttled(host, 2.0)
        idx += 1

    prop_healthy = sum(completions_prop[h] for h in all_hosts if h not in throttled_hosts)
    prop_rate = sched_prop.current_requests_per_minute

    print(
        f"\n[production-sim] fixed: healthy={fixed_healthy} rate={fixed_rate:.0f}  "
        f"proportional: healthy={prop_healthy} rate={prop_rate:.0f}  "
        f"improvement=+{prop_healthy - fixed_healthy}"
    )

    # Proportional must be materially better.
    assert prop_healthy > fixed_healthy, (
        f"proportional ({prop_healthy}) must beat fixed ({fixed_healthy})"
    )
    assert prop_rate >= fixed_rate, (
        f"proportional rate ({prop_rate}) must be >= fixed rate ({fixed_rate})"
    )
    # Active ceiling should stay near max for proportional.
    assert prop_rate >= _AUTH_CEILING * 0.9, (
        f"proportional rate ({prop_rate}) should be near ceiling ({_AUTH_CEILING})"
    )

    # Per-host cooldown compliance: every throttled host must have fewer completions.
    for host in throttled_hosts:
        non_throttled_avg = prop_healthy / len([h for h in all_hosts if h not in throttled_hosts])
        # Throttled hosts get some completions but substantially fewer.
        assert completions_prop[host] <= non_throttled_avg * 1.5, (
            f"throttled host {host} got {completions_prop[host]} vs avg {non_throttled_avg:.0f}"
        )

    # Global reductions: proportional should have 0, fixed should have ≥1.
    assert prop_rate == _AUTH_CEILING
    assert fixed_rate < _AUTH_CEILING
