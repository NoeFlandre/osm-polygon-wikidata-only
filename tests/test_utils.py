"""Tests for the small utilities."""

from __future__ import annotations

import json
import threading
import urllib.error
from email.message import Message

import pytest

from osm_polygon_wikidata_only.utils import rate_limit
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


def test_defer_host_moves_next_request_after_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: 10.0)
    rate_limit.defer_host("en.wikipedia.org", 30.0)
    assert rate_limit.next_wait_seconds("en.wikipedia.org") == 30.0


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


def test_retry_after_seconds_parses_numeric_header() -> None:
    headers = Message()
    headers["Retry-After"] = "12.5"
    error = urllib.error.HTTPError("https://example.test", 429, "limited", headers, None)
    assert rate_limit.retry_after_seconds(error) == 12.5


def test_retry_after_seconds_uses_default_for_invalid_header() -> None:
    headers = Message()
    headers["Retry-After"] = "not-a-date"
    error = urllib.error.HTTPError("https://example.test", 429, "limited", headers, None)
    assert rate_limit.retry_after_seconds(error, default_s=17) == 17
