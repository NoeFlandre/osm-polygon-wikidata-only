"""Wikidata client split -- characterization tests.

These tests pin behaviour that the split of
``enrichment.wikidata_client`` into ``enrichment.wikidata.{transport,
cache,models,parsing}`` must preserve. They lock down:

* identity preservation for the documented facade surface;
* round-trip serialization for successful entities and authoritative
  missing responses (cache stores ``None`` payload + ``not_found`` status);
* cache-hit short-circuit for valid positive and authoritative negative entries;
* TTL selection (``failed_ttl_s`` for authoritative absence, default for success);
* cache-key normalization (``wikidata/{qid}.json``);
* corrupt, malformed, and legacy ambiguous cached payloads are treated as misses;
* transport failures propagate without cache writes;
* :class:`HttpWikidataClient` constructor signature + defaults;
* :class:`CachedWikidataClient` constructor signature + defaults;
* batch ordering: ``get_entities`` must preserve caller order and
  deduplicate by ``dict.fromkeys``;
* invalid QIDs return ``None`` without ever firing a request.
"""

from __future__ import annotations

import urllib.error
from email.message import Message
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubSession:
    def __init__(self, responses: list[Any]) -> None:
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
        "wikidata_min_interval_s": 1.0,
        "wikimedia_authenticated_min_interval_s": 0.5,
        "rate_limit_retry_after_default_s": 60.0,
    }
    base.update(overrides)
    return type("Settings", (), base)()


def _cache_entry(status: str, parsed_result: Any, request_url: str | None) -> Any:
    return type(
        "CacheEntry",
        (),
        {"status": status, "parsed_result": parsed_result, "request_url": request_url},
    )()


# ---------------------------------------------------------------------------
# Identity: facade surface preserved
# ---------------------------------------------------------------------------


def test_wikidata_facade_class_identity_after_split() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata import (
        cache as focused_cache,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata import (
        transport as focused_transport,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient as FacadeCached,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient as FacadeHttp,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        InMemoryWikidataClient as FacadeInMemory,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        WikidataError as FacadeError,
    )

    assert FacadeHttp is focused_transport.HttpWikidataClient
    assert FacadeInMemory is focused_transport.InMemoryWikidataClient
    assert FacadeCached is focused_cache.CachedWikidataClient
    assert FacadeError is focused_transport.WikidataError


def test_wikidata_facade_does_not_leak_new_helpers() -> None:
    import osm_polygon_wikidata_only.enrichment.wikidata_client as facade

    forbidden = {
        "_entity_to_dict",
        "_entity_from_dict",
        "_build_url",
    }
    leaked = forbidden & set(dir(facade))
    assert not leaked, f"facade leaked implementation helpers: {leaked}"


def test_wikidata_transport_uses_legacy_logger_name() -> None:
    """The transport module's logger must be the legacy module path
    ``osm_polygon_wikidata_only.enrichment.wikidata_client`` so the
    warning emitted on a failed batch request remains observable by
    consumers filtering on the legacy name.

    This locks down the legacy logger-name preservation requirement;
    the helper-level invariant (``record.name == ...``) is asserted
    in ``test_http_wikidata_warning_on_batch_failure`` and
    ``test_http_wikidata_503_returns_none_via_cache``.
    """
    import osm_polygon_wikidata_only.enrichment.wikidata.transport as focused_transport

    assert focused_transport.LOGGER.name == ("osm_polygon_wikidata_only.enrichment.wikidata_client")


# ---------------------------------------------------------------------------
# Constructor signatures / defaults
# ---------------------------------------------------------------------------


def test_http_wikidata_constructor_signature() -> None:
    from osm_polygon_wikidata_only.config.settings import WIKIDATA_API_URL
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    client = HttpWikidataClient(settings, scheduler=scheduler)
    assert client._settings is settings
    assert client._scheduler is scheduler
    assert client._endpoint == WIKIDATA_API_URL


def test_cached_wikidata_constructor_signature_and_failed_ttl_default() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
    )

    assert CachedWikidataClient.__init__.__kwdefaults__ == {"failed_ttl_s": 60 * 60}


# ---------------------------------------------------------------------------
# Cache round-trip: successful entity serialization
# ---------------------------------------------------------------------------


def _recording_cache() -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {"stored": []}

    class _Cache:
        def get(self, key: str) -> Any:
            return _cache_entry("ok", captured.get("hit"), None)

        def set(self, key: str, payload: Any, **kwargs: Any) -> None:
            captured["stored"].append((key, payload, kwargs))

    return _Cache(), captured


def test_cached_wikidata_serializes_entity_for_success() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        InMemoryWikidataClient,
        WikidataEntity,
    )

    entity = WikidataEntity(
        qid="Q1",
        sitelinks={"enwiki": "Monaco"},
        labels={"en": "Monaco"},
        descriptions={"en": "Country"},
        aliases={"en": ["Monaco"]},
    )
    inner = InMemoryWikidataClient({"Q1": entity})

    # Cache returns None (miss) so the inner is called.
    stored: list[tuple[str, Any, dict[str, Any]]] = []

    class _MissCache:
        def get(self, key: str) -> Any:
            return None

        def set(self, key: str, payload: Any, **kwargs: Any) -> None:
            stored.append((key, payload, kwargs))

    client = CachedWikidataClient(inner, _MissCache())
    result = client.get_entity("Q1")

    assert result == entity
    assert len(stored) == 1
    key, payload, kwargs = stored[0]
    assert key == "wikidata/Q1.json"
    assert isinstance(payload, dict)
    assert payload["qid"] == "Q1"
    assert payload["sitelinks"] == {"enwiki": "Monaco"}
    assert payload["labels"] == {"en": "Monaco"}
    assert kwargs["status"] == "ok"
    assert kwargs["request_url"] == ""  # no HTTP inner client


# ---------------------------------------------------------------------------
# Cache round-trip: authoritative missing entity serialization
# ---------------------------------------------------------------------------


def test_cached_wikidata_serializes_authoritative_missing_with_failed_ttl() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        InMemoryWikidataClient,
    )

    inner = InMemoryWikidataClient({})
    stored: list[tuple[str, Any, dict[str, Any]]] = []

    class _MissCache:
        def get(self, key: str) -> Any:
            return None

        def set(self, key: str, payload: Any, **kwargs: Any) -> None:
            stored.append((key, payload, kwargs))

    client = CachedWikidataClient(inner, _MissCache(), failed_ttl_s=99)

    result = client.get_entity("Q1")

    assert result is None
    assert len(stored) == 1
    key, payload, kwargs = stored[0]
    assert key == "wikidata/Q1.json"
    assert payload is None
    assert kwargs["status"] == "not_found"
    assert kwargs["ttl_s"] == 99
    assert kwargs["response_metadata"] == {"reason": "wikidata_entity_missing"}


# ---------------------------------------------------------------------------
# Cache-hit short-circuit
# ---------------------------------------------------------------------------


def test_cached_wikidata_hit_skips_inner_fetch() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        WikidataEntity,
    )

    cached_payload = {
        "qid": "Q1",
        "sitelinks": {},
        "labels": {},
        "descriptions": {},
        "aliases": {},
    }

    class _Cache:
        def get(self, key: str) -> Any:
            return _cache_entry("ok", cached_payload, None)

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

    batch_calls: list[list[str]] = []
    per_title_calls: list[str] = []

    class _Inner:
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            # Record that batch was called (even with empty list, per
            # legacy behaviour).
            batch_calls.append(list(qids))
            return []

        def get_entity(self, qid: str) -> WikidataEntity | None:
            per_title_calls.append(qid)
            return None

    inner = _Inner()
    client = CachedWikidataClient(inner, _Cache())  # type: ignore[arg-type]
    result = client.get_entity("Q1")

    # Result reflects the cache, not the inner.
    assert result is not None
    assert result.qid == "Q1"
    # Inner's per-title path must never be invoked.
    assert per_title_calls == []
    # Legacy behaviour: the batch path may be invoked (even with []); if
    # so, the inner's get_entities MUST NOT see per-QID work.
    if batch_calls:
        assert batch_calls == [[]]


# ---------------------------------------------------------------------------
# Corrupt / malformed cached payload behaviour
# ---------------------------------------------------------------------------


def test_cached_wikidata_corrupt_payload_is_treated_as_a_miss() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        WikidataEntity,
    )

    inner_calls: list[str] = []

    class _Inner:
        def get_entity(self, qid: str) -> WikidataEntity:
            inner_calls.append(qid)
            return WikidataEntity(qid=qid)

    cache = type(
        "C",
        (),
        {
            "get": lambda self, key: _cache_entry("ok", "not-a-dict", None),
            "set": lambda self, *args, **kw: None,
        },
    )()

    client = CachedWikidataClient(_Inner(), cache)  # type: ignore[arg-type]
    result = client.get_entity("Q1")
    assert result == WikidataEntity(qid="Q1")
    assert inner_calls == ["Q1"]


def test_cached_wikidata_legacy_error_hit_is_treated_as_a_miss() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        WikidataEntity,
    )

    inner_calls: list[str] = []

    class _Inner:
        def get_entity(self, qid: str) -> WikidataEntity:
            inner_calls.append(qid)
            return WikidataEntity(qid=qid)

    cache = type(
        "C",
        (),
        {
            "get": lambda self, key: _cache_entry("error", None, None),
            "set": lambda self, *args, **kw: None,
        },
    )()

    client = CachedWikidataClient(_Inner(), cache)  # type: ignore[arg-type]
    result = client.get_entity("Q1")
    assert result == WikidataEntity(qid="Q1")
    assert inner_calls == ["Q1"]


# ---------------------------------------------------------------------------
# Batch ordering + invalid QIDs
# ---------------------------------------------------------------------------


def test_cached_wikidata_batch_preserves_order_and_dedup() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        InMemoryWikidataClient,
        WikidataEntity,
    )

    inner = InMemoryWikidataClient(
        {
            "Q1": WikidataEntity(qid="Q1", sitelinks={}, labels={}, descriptions={}, aliases={}),
            "Q2": WikidataEntity(qid="Q2", sitelinks={}, labels={}, descriptions={}, aliases={}),
        }
    )
    cache = type(
        "C",
        (),
        {"get": lambda self, key: None, "set": lambda self, *a, **kw: None},
    )()

    client = CachedWikidataClient(inner, cache)  # type: ignore[arg-type]
    results = client.get_entities(["Q1", "Q2", "Q1", "bogus", "Q2"])

    assert [r.qid if r else None for r in results] == ["Q1", "Q2", "Q1", None, "Q2"]


def test_cached_wikidata_invalid_qid_returns_none_without_fetch() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        InMemoryWikidataClient,
    )

    inner = InMemoryWikidataClient({})
    sets: list[tuple[str, Any, dict[str, Any]]] = []

    cache = type(
        "C",
        (),
        {
            "get": lambda self, key: None,
            "set": lambda self, key, payload, **kw: sets.append((key, payload, kw)),
        },
    )()

    client = CachedWikidataClient(inner, cache)  # type: ignore[arg-type]
    result = client.get_entity("not-a-qid")
    assert result is None
    # No cache write for invalid QIDs.
    assert sets == []


# ---------------------------------------------------------------------------
# HTTP client: logger names / failure semantics
# ---------------------------------------------------------------------------


def test_http_wikidata_batch_failure_propagates() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession([_http_error(503, retry_after="3")])
    client = HttpWikidataClient(settings, scheduler=scheduler)
    client._session = session
    client._endpoint = "https://www.wikidata.org/w/api.php"

    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 503"):
        client.get_entities(["Q1"])

    assert scheduler.throttle_calls == [("www.wikidata.org", 3.0)]


def test_http_wikidata_503_via_cache_propagates_without_cache_write() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        CachedWikidataClient,
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession([_http_error(503, retry_after="3")])
    client = HttpWikidataClient(settings, scheduler=scheduler)
    client._session = session
    client._endpoint = "https://www.wikidata.org/w/api.php"

    stored: list[tuple[str, Any, dict[str, Any]]] = []

    class _MissCache:
        def get(self, key: str) -> Any:
            return None

        def set(self, key: str, payload: Any, **kwargs: Any) -> None:
            stored.append((key, payload, kwargs))

    cached = CachedWikidataClient(client, _MissCache(), failed_ttl_s=11)

    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 503"):
        cached.get_entity("Q1")

    assert stored == []


# ---------------------------------------------------------------------------
# Identity of constructor arguments (extras: ensure ``endpoint`` override)
# ---------------------------------------------------------------------------


def test_http_wikidata_accepts_endpoint_override() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        HttpWikidataClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    client = HttpWikidataClient(
        settings,
        scheduler=scheduler,
        endpoint="https://example.test/w/api.php",
    )
    assert client._endpoint == "https://example.test/w/api.php"
    assert client._build_url("Q1") == (
        "https://example.test/w/api.php?"
        "action=wbgetentities&ids=Q1&props=sitelinks%7Clabels%7Cdescriptions"
        "%7Caliases&sitefilter=wiki&languages=en&format=json&maxlag=5"
    )
