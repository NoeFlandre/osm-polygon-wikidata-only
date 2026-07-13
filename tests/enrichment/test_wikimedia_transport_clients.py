"""Per-client invariants that survive the Wikimedia transport centralization.

These tests pin specific behaviour of the three callers that
previously inlined the read+gzip+JSON+throttle mechanics:

* a real cache hit must short-circuit every transport-side effect
  (no HTTPS validation, no request construction, no session read, no
  retry loop);
* Wikipedia and Wikidata throttles must NOT emit the augmentation-
  style warning; only ``AdaptiveRequestScheduler.report_host_throttled``
  runs;
* the augmentation client must emit exactly one WARNING per
  throttled attempt, with the existing message shape;
* non-object JSON error messages must match the *exact* strings each
  caller used before centralization;
* the throttle callback (and the scheduler notification it forwards
  to) must fire exactly once per throttled attempt.
"""

from __future__ import annotations

import logging
import urllib.error
from email.message import Message
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes -- session, scheduler, cache
# ---------------------------------------------------------------------------


class _StubSession:
    """Records ``WikimediaSession.read`` calls and surfaces canned responses."""

    def __init__(self, responses: list[tuple[bytes, str] | BaseException]) -> None:
        self.reads: list[tuple[Any, float, float]] = []
        self._responses = list(responses)

    def read(
        self,
        request: Any,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]:
        self.reads.append((request, min_interval_anonymous_s, min_interval_authenticated_s))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _RecordingScheduler:
    """Captures ``report_host_throttled`` invocations."""

    def __init__(self) -> None:
        self.throttle_calls: list[tuple[str, float]] = []

    def report_host_throttled(self, host: str, delay: float) -> None:
        self.throttle_calls.append((host, delay))


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.test", code, "error", headers, None)


def _make_settings(**overrides: Any) -> Any:
    base = {
        "user_agent": "ua",
        "request_max_retries": 1,
        "request_base_delay_s": 0.0,
        "request_timeout_s": 60.0,
        "wikipedia_min_interval_s": 1.0,
        "wikidata_min_interval_s": 1.0,
        "augmentation_min_interval_s": 1.0,
        "wikimedia_authenticated_min_interval_s": 0.5,
        "rate_limit_retry_after_default_s": 60.0,
    }
    base.update(overrides)
    return type("Settings", (), base)()


# ---------------------------------------------------------------------------
# Augmentation client -- cache short-circuit
# ---------------------------------------------------------------------------


def test_augmentation_cache_hit_skips_every_transport_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit returns the cached value without HTTPS validation, request
    construction, retry, session read, or throttle invocation.

    This guards behaviour that the previous Phase 3 implementation
    silently regressed: the URL was still parsed even when the hit
    short-circuited the read.
    """

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession([])
    cache = _HitCache(parsed={"hit": True})

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    parsed = client.get_json("http://not-https.example.test/api", key="k")

    assert parsed is cache.cached_parsed
    assert session.reads == []
    assert scheduler.throttle_calls == []


def test_augmentation_cache_miss_runs_full_pipeline() -> None:

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(
        responses=[(b'{"entities": {"Q1": {"id": "Q1", "labels": {}}}}', "identity")],
    )
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    parsed = client.get_json("https://www.wikidata.org/w/api.php?ids=Q1", key="k")

    assert parsed == {"entities": {"Q1": {"id": "Q1", "labels": {}}}}
    assert len(session.reads) == 1


# ---------------------------------------------------------------------------
# Augmentation client -- logging invariant
# ---------------------------------------------------------------------------


def test_augmentation_emits_existing_warning_per_throttle_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Augmentation must emit the existing throttle WARNING exactly once
    per throttled attempt, using the legacy message shape.

    The augmentation client catches ``urllib.error.HTTPError`` around
    the helper invocation only to emit the warning using ``error.code``;
    the helper has already notified the scheduler via the
    ``(host, delay)`` callback.
    """
    import pytest

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[_http_error(429, retry_after="5")])
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    with caplog.at_level(
        logging.WARNING, logger="osm_polygon_wikidata_only.augmentation.mediawiki"
    ):
        with pytest.raises(urllib.error.HTTPError):
            client.get_json("https://www.wikidata.org/w/api.php?ids=Q1", key="k1")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "Wikimedia throttled www.wikidata.org (HTTP 429); retrying after 5.0s"
    )
    assert scheduler.throttle_calls == [("www.wikidata.org", 5.0)]


def test_augmentation_emits_warning_for_503_with_error_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Augmentation must include the actual ``error.code`` in the warning."""
    import pytest

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[_http_error(503, retry_after="2")])
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    with caplog.at_level(
        logging.WARNING, logger="osm_polygon_wikidata_only.augmentation.mediawiki"
    ):
        with pytest.raises(urllib.error.HTTPError):
            client.get_json("https://www.wikidata.org/w/api.php?ids=Q2", key="k-503")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "HTTP 503" in warnings[0].getMessage()
    assert scheduler.throttle_calls == [("www.wikidata.org", 2.0)]


def test_augmentation_does_not_warn_on_non_throttle_http_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Augmentation must not emit the throttle warning on, e.g., 5xx other than 503."""
    import pytest

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[_http_error(502)])
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    with caplog.at_level(
        logging.WARNING, logger="osm_polygon_wikidata_only.augmentation.mediawiki"
    ):
        with pytest.raises(urllib.error.HTTPError):
            client.get_json("https://www.wikidata.org/w/api.php?ids=Q3", key="k-502")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == []
    assert scheduler.throttle_calls == []


def test_augmentation_http_date_delay_agrees_between_scheduler_and_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scheduler notification and warning must use the helper's parsed delay.

    For HTTP-date headers, re-parsing the Retry-After value would yield a
    different number of seconds (the gap between the server's target
    timestamp and ``datetime.now()`` shrinks as time passes). The
    augmentation client must rely on the delay captured from the
    helper's callback, never re-parse the header itself.
    """
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    import pytest

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    target = datetime.now(UTC) + timedelta(seconds=120)
    http_date = format_datetime(target, usegmt=True)
    error = _http_error(429, retry_after=http_date)
    settings = _make_settings(request_max_retries=1, request_base_delay_s=0.0)
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[error])
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    with caplog.at_level(
        logging.WARNING, logger="osm_polygon_wikidata_only.augmentation.mediawiki"
    ):
        with pytest.raises(urllib.error.HTTPError):
            client.get_json(
                "https://www.wikidata.org/w/api.php?ids=Q-http-date",
                key="k-http-date",
            )

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert scheduler.throttle_calls, "scheduler was not notified"
    scheduler_delay = scheduler.throttle_calls[0][1]
    # Warning renders with %.1f precision; the scheduler sees the
    # unformatted float. Round-trip must be exact after the warning's
    # precision is restored.
    rendered_delay = float(message.rsplit("after ", 1)[1].rstrip("s"))
    assert round(scheduler_delay, 1) == rendered_delay, (
        f"scheduler saw {scheduler_delay}s but warning logged {rendered_delay}s"
    )
    # Parsed value should be near the target 120s (within tolerance for
    # test execution time).
    assert 110.0 <= scheduler_delay <= 120.0


# ---------------------------------------------------------------------------
# Wikipedia / Wikidata -- no augmentation warning
# ---------------------------------------------------------------------------


def test_wikipedia_does_not_emit_augmentation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wikipedia client must only notify the scheduler, no warning logs."""

    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[_http_error(429, retry_after="7")])

    client = HttpWikipediaClient.__new__(HttpWikipediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session

    with caplog.at_level(
        logging.WARNING,
        logger="osm_polygon_wikidata_only.enrichment.wikipedia_client",
    ):
        result = client.fetch_article("en", "enwiki", "Monaco")

    assert result.status == "rate_limited"
    augmentation_records = [
        record for record in caplog.records if "Wikimedia throttled" in record.getMessage()
    ]
    assert augmentation_records == []
    assert scheduler.throttle_calls == [("en.wikipedia.org", 7.0)]


def test_wikidata_does_not_emit_augmentation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wikidata client must only notify the scheduler, no warning logs."""

    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[_http_error(503, retry_after="3")])

    client = HttpWikidataClient.__new__(HttpWikidataClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._endpoint = "https://www.wikidata.org/w/api.php"

    with caplog.at_level(
        logging.WARNING,
        logger="osm_polygon_wikidata_only.enrichment.wikidata_client",
    ):
        result = client.get_entity("Q1")

    assert result is None
    augmentation_records = [
        record for record in caplog.records if "Wikimedia throttled" in record.getMessage()
    ]
    assert augmentation_records == []
    assert scheduler.throttle_calls == [("www.wikidata.org", 3.0)]


# ---------------------------------------------------------------------------
# Non-object JSON error messages (exact pre-refactor strings preserved)
# ---------------------------------------------------------------------------


def test_wikipedia_non_object_error_message_is_exact() -> None:
    import pytest

    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[(b"[1, 2, 3]", "identity")])

    client = HttpWikipediaClient.__new__(HttpWikipediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session

    url = "https://en.wikipedia.org/w/api.php?action=query&titles=X"
    with pytest.raises(ValueError) as exc_info:
        client._http_get(url)
    assert str(exc_info.value) == f"Expected JSON object from {url}, got list"
    # Exception chaining is suppressed: the original marker must not
    # appear in the formatted traceback that an external observer sees.
    assert exc_info.value.__suppress_context__ is True


def test_wikidata_non_object_error_message_is_exact() -> None:
    import pytest

    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[(b"42", "identity")])

    client = HttpWikidataClient.__new__(HttpWikidataClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session

    url = "https://www.wikidata.org/w/api.php?ids=Q1&action=wbgetentities"
    with pytest.raises(ValueError) as exc_info:
        client._http_get(url)
    assert str(exc_info.value) == f"Expected JSON object from {url}, got int"
    assert exc_info.value.__suppress_context__ is True


def test_augmentation_non_object_error_message_is_exact() -> None:
    import pytest

    from osm_polygon_wikidata_only.augmentation.mediawiki import (
        AugmentationWikimediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession(responses=[(b'"a string"', "identity")])
    cache = _NoHitCache()

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    client._settings = settings
    client._scheduler = scheduler
    client._session = session
    client._cache = cache

    url = "https://www.wikidata.org/w/api.php?ids=Q1"
    with pytest.raises(ValueError) as exc_info:
        client.get_json(url, key="k-non-object")
    assert str(exc_info.value) == f"Expected JSON object from {url}"
    assert exc_info.value.__suppress_context__ is True


# ---------------------------------------------------------------------------
# Cache + scheduler fakes
# ---------------------------------------------------------------------------


class _HitCache:
    def __init__(self, parsed: object) -> None:
        self.cached_parsed = parsed
        self.stores: list[tuple[str, Any, dict[str, Any]]] = []

    def get(self, key: str) -> Any:
        return _CacheEntry("ok", self.cached_parsed, None)

    def set(self, key: str, payload: Any, **kwargs: Any) -> None:
        self.stores.append((key, payload, kwargs))


class _NoHitCache:
    def __init__(self) -> None:
        self.stores: list[tuple[str, Any, dict[str, Any]]] = []

    def get(self, key: str) -> Any:
        return None

    def set(self, key: str, payload: Any, **kwargs: Any) -> None:
        self.stores.append((key, payload, kwargs))


class _CacheEntry:
    def __init__(self, status: str, parsed_result: Any, request_url: str | None) -> None:
        self.status = status
        self.parsed_result = parsed_result
        self.request_url = request_url
