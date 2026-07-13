"""Tests for the small utilities."""

from __future__ import annotations

import json
import threading

import pytest

from osm_polygon_wikidata_only.utils.json import dumps, dumps_compact_list, loads
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler
from osm_polygon_wikidata_only.utils.retry import with_retries
from osm_polygon_wikidata_only.utils.time import parse_iso_to_z, utc_now_iso


def test_dumps_is_deterministic() -> None:
    a = dumps({"b": 2, "a": 1})
    b = dumps({"a": 1, "b": 2})
    assert a == b
    assert a == json.dumps(
        {"a": 1, "b": 2}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def test_dumps_preserves_unicode() -> None:
    assert "é" in dumps({"name": "Café"})


def test_dumps_uses_compact_separators() -> None:
    assert dumps({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_loads_round_trips() -> None:
    assert loads('{"a":1}') == {"a": 1}


def test_dumps_compact_list_sorts_and_dedups() -> None:
    out = dumps_compact_list(["b", "a", "a", ""])
    assert loads(out) == ["a", "b"]


def test_utc_now_iso_has_z_suffix() -> None:
    ts = utc_now_iso()
    assert ts.endswith("Z")
    # 20 chars: YYYY-MM-DDTHH:MM:SSZ
    assert len(ts) == 20


def test_parse_iso_to_z_normalizes_z_suffix() -> None:
    assert parse_iso_to_z("2026-01-02T03:04:05Z") == "2026-01-02T03:04:05Z"


def test_parse_iso_to_z_normalizes_offset() -> None:
    assert parse_iso_to_z("2026-01-02T03:04:05+00:00") == "2026-01-02T03:04:05Z"


def test_parse_iso_to_z_returns_input_on_garbage() -> None:
    assert parse_iso_to_z("not a date") == "not a date"


def test_scheduler_global_cooldown_delays_next_request() -> None:
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        max_in_flight=3,
        requests_per_minute=60,
        clock=lambda: now[0],
        sleep=sleep,
    )
    scheduler.defer(12.0)
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == [12.0]


def test_scheduler_never_exceeds_global_concurrency() -> None:
    scheduler = AdaptiveRequestScheduler(max_in_flight=3, requests_per_minute=100_000)
    three_entered = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0
    lock = threading.Lock()

    def work() -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 3:
                three_entered.set()
        release.wait(timeout=2)
        with lock:
            active -= 1

    threads = [threading.Thread(target=lambda: scheduler.run(work)) for _ in range(6)]
    for thread in threads:
        thread.start()
    assert three_entered.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
    assert peak == 3


def test_scheduler_reports_configured_concurrency_without_private_introspection() -> None:
    scheduler = AdaptiveRequestScheduler(max_in_flight=7, requests_per_minute=100_000)

    assert scheduler.max_in_flight == 7


def test_scheduler_snapshot_reports_recent_budget_usage_and_throttles() -> None:
    now = [0.0]
    scheduler = AdaptiveRequestScheduler(
        max_in_flight=3,
        requests_per_minute=1200,
        max_requests_per_minute=1200,
        minimum_requests_per_minute=200,
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    for _ in range(10):
        scheduler.run(lambda: None)
    now[0] = 60.0
    scheduler.report_host_throttled("pl.wikipedia.org", 4.0)

    snapshot = scheduler.snapshot()

    assert snapshot.requests_last_minute == 10
    assert snapshot.maximum_requests_per_minute == 1200
    assert snapshot.utilization_percent == pytest.approx(10 / 1200 * 100)
    assert snapshot.in_flight == 0
    assert snapshot.throttle_events == 1


def test_scheduler_snapshot_reports_operation_as_in_flight() -> None:
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)

    observed = scheduler.run(lambda: scheduler.snapshot().in_flight)

    assert observed == 1


def test_scheduler_raises_rate_after_success_window_without_exceeding_ceiling() -> None:
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=100,
        max_requests_per_minute=130,
        successes_per_increase=2,
    )

    assert scheduler.current_requests_per_minute == 100
    scheduler.report_success()
    assert scheduler.current_requests_per_minute == 100
    scheduler.report_success()
    assert scheduler.current_requests_per_minute == 125
    scheduler.report_success()
    scheduler.report_success()
    assert scheduler.current_requests_per_minute == 130


def test_scheduler_without_higher_ceiling_remains_fixed() -> None:
    scheduler = AdaptiveRequestScheduler(requests_per_minute=180, successes_per_increase=1)

    scheduler.report_success()

    assert scheduler.current_requests_per_minute == 180


def test_scheduler_throttling_applies_cooldown_and_halves_active_rate() -> None:
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        successes_per_increase=1,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_throttled(12)

    assert scheduler.current_requests_per_minute == 100
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == [12]


def test_scheduler_successes_restore_rate_after_throttling() -> None:
    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        successes_per_increase=1,
    )
    scheduler.report_throttled(0)

    scheduler.report_success()

    assert scheduler.current_requests_per_minute == 125


def test_report_host_throttled_single_host_does_not_escalate() -> None:
    """A 429 from one host must not trigger the global cooldown."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 60.0)

    assert scheduler.current_requests_per_minute == 200
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == []


def test_report_host_throttled_escalates_when_threshold_hosts_fail() -> None:
    """Three distinct hosts within the window must trigger global backoff."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        successes_per_increase=1,
        host_throttle_threshold=3,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 60.0)
    scheduler.report_host_throttled("de.wikipedia.org", 60.0)
    assert scheduler.current_requests_per_minute == 200

    scheduler.report_host_throttled("es.wikipedia.org", 60.0)
    assert scheduler.current_requests_per_minute == 100
    assert scheduler.run(lambda: "ok") == "ok"
    assert sleeps == [60.0]


def test_report_host_throttled_prunes_events_outside_window() -> None:
    """Old events expire, so a slow trickle never escalates."""
    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    scheduler = AdaptiveRequestScheduler(
        requests_per_minute=200,
        max_requests_per_minute=400,
        minimum_requests_per_minute=60,
        host_throttle_window_s=10.0,
        host_throttle_threshold=3,
        clock=lambda: now[0],
        sleep=sleep,
    )

    scheduler.report_host_throttled("fr.wikipedia.org", 5.0)
    now[0] += 11.0
    scheduler.report_host_throttled("de.wikipedia.org", 5.0)
    now[0] += 11.0
    scheduler.report_host_throttled("es.wikipedia.org", 5.0)

    assert scheduler.current_requests_per_minute == 200
    assert sleeps == []


def test_retry_returns_after_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("osm_polygon_wikidata_only.utils.retry.time.sleep", lambda _: None)
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary")
        return "ok"

    assert with_retries(operation, attempts=3, base_delay=0, retry_on=(OSError,)) == "ok"
    assert attempts == 3


def test_retry_raises_last_error_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("osm_polygon_wikidata_only.utils.retry.time.sleep", lambda _: None)

    with pytest.raises(OSError, match="offline"):
        with_retries(
            lambda: (_ for _ in ()).throw(OSError("offline")),
            attempts=2,
            base_delay=0,
            retry_on=(OSError,),
        )
