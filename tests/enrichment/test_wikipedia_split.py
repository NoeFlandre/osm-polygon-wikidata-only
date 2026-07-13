"""Wikipedia client split -- characterization tests.

These tests pin behaviour that the split of
``enrichment.wikipedia_client`` into ``enrichment.wikipedia.{transport,
cache,models,parsing}`` must preserve. They lock down:

* identity preservation for the documented facade surface;
* round-trip serialization for successful articles and failed
  responses;
* cache-hit short-circuit (no inner fetch);
* TTL selection (``failed_ttl_s`` for failures, default for success);
* cache-key normalization (slash / space encoding, policy suffix);
* corrupt / malformed cached payload behaviour;
* legacy logger name emission on Wikidata-style ``_build_url`` calls
  (none today -- this is a regression guard);
* :class:`HttpWikipediaClient` constructor signature + defaults;
* :class:`CachedWikipediaClient` constructor signature + defaults;
* Action API fallback (when ``fetch_article`` returns ``empty_text``
  with ``fetch_full_text=True``) and batch ``fetch_articles``
  per-title selection logic.
"""

from __future__ import annotations

import logging
import urllib.error
from email.message import Message
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes (mirror the style used by ``test_wikimedia_transport_clients.py``)
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
        "wikipedia_min_interval_s": 1.0,
        "wikimedia_authenticated_min_interval_s": 0.5,
        "rate_limit_retry_after_default_s": 60.0,
    }
    base.update(overrides)
    return type("Settings", (), base)()


# ---------------------------------------------------------------------------
# Identity: facade surface preserved
# ---------------------------------------------------------------------------


def test_wikipedia_facade_class_identity_after_split() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import (
        cache as focused_cache,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia import (
        transport as focused_transport,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient as FacadeCached,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient as FacadeHttp,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        InMemoryWikipediaClient as FacadeInMemory,
    )

    assert FacadeHttp is focused_transport.HttpWikipediaClient
    assert FacadeInMemory is focused_transport.InMemoryWikipediaClient
    assert FacadeCached is focused_cache.CachedWikipediaClient


def test_wikipedia_facade_does_not_leak_new_helpers() -> None:
    import osm_polygon_wikidata_only.enrichment.wikipedia_client as facade

    forbidden = {
        "_article_to_dict",
        "_article_from_dict",
        "_safe_title",
        "_build_url",
        "_build_parse_url",
    }
    leaked = forbidden & set(dir(facade))
    assert not leaked, f"facade leaked implementation helpers: {leaked}"


# ---------------------------------------------------------------------------
# Constructor signatures / defaults
# ---------------------------------------------------------------------------


def test_http_wikipedia_constructor_signature() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    client = HttpWikipediaClient(settings, scheduler=scheduler)
    assert client._settings is settings
    assert client._scheduler is scheduler


def test_cached_wikipedia_constructor_signature_and_failed_ttl_default() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
    )

    _make_settings()
    client = CachedWikipediaClient.__new__(CachedWikipediaClient)
    assert client.__init__.__kwdefaults__ == {"failed_ttl_s": 60 * 60}


# ---------------------------------------------------------------------------
# Cache round-trip: successful article serialization
# ---------------------------------------------------------------------------


def _cache_entry(status: str, parsed_result: Any, request_url: str | None) -> Any:
    return type(
        "CacheEntry",
        (),
        {"status": status, "parsed_result": parsed_result, "request_url": request_url},
    )()


def _recording_cache() -> Any:
    captured: dict[str, Any] = {"stored": []}

    class _Cache:
        def get(self, key: str) -> Any:
            return _cache_entry("ok", captured.get("hit"), None)

        def set(self, key: str, payload: Any, **kwargs: Any) -> None:
            captured["stored"].append((key, payload, kwargs))

    return _Cache(), captured


def test_cached_wikipedia_serializes_article_for_success() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
        InMemoryWikipediaClient,
        WikipediaArticle,
    )

    article = WikipediaArticle(
        language="en",
        site="enwiki",
        title="Monaco",
        page_id=42,
        revision_id=7,
        revision_timestamp="2024-01-01T00:00:00Z",
        url="https://en.wikipedia.org/wiki/Monaco",
        lead_text="lead",
        extract="extract body",
        full_text="full body",
        full_text_format="plain_text",
        thumbnail_url="thumb",
        thumbnail_width=320,
        thumbnail_height=240,
        categories=["Cat1"],
        license="CC BY-SA 4.0",
        attribution="attr",
        source_api="mediawiki_action_api",
        retrieved_at="2024-01-01T00:00:00Z",
    )
    inner = InMemoryWikipediaClient({("enwiki", "Monaco"): FetchResult("ok", article)})
    cache, captured = _recording_cache()
    client = CachedWikipediaClient(inner, cache)

    result = client.fetch_article("en", "enwiki", "Monaco")

    assert result == FetchResult("ok", article)
    stored = captured["stored"]
    assert len(stored) == 1
    key, payload, kwargs = stored[0]
    assert key == "wikipedia/full-text-v2/enwiki/Monaco.json"
    assert isinstance(payload, dict)
    assert payload["language"] == "en"
    assert payload["title"] == "Monaco"
    assert payload["page_id"] == 42
    assert payload["revision_id"] == 7
    assert kwargs["status"] == "ok"
    assert kwargs["request_url"] == ""  # no HTTP inner client


# ---------------------------------------------------------------------------
# Cache round-trip: failed result serialization
# ---------------------------------------------------------------------------


def test_cached_wikipedia_serializes_failure_with_failed_ttl() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
        InMemoryWikipediaClient,
    )

    inner = InMemoryWikipediaClient(
        {("enwiki", "Missing"): FetchResult("article_not_found", None, "missing")},
    )
    cache, captured = _recording_cache()
    client = CachedWikipediaClient(inner, cache, failed_ttl_s=123)

    result = client.fetch_article("en", "enwiki", "Missing")

    assert result.status == "article_not_found"
    stored = captured["stored"]
    assert len(stored) == 1
    key, payload, kwargs = stored[0]
    assert key == "wikipedia/full-text-v2/enwiki/Missing.json"
    assert payload == "article_not_found"
    assert kwargs["status"] == "error"
    assert kwargs["ttl_s"] == 123
    assert kwargs["response_metadata"]["status"] == "article_not_found"


# ---------------------------------------------------------------------------
# Cache-key normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Simple", "Simple"),
        ("With Space", "With_Space"),
        ("Has/Slash", "Has_Slash"),
        ("Both Kinds Of/Things", "Both_Kinds_Of_Things"),
    ],
)
def test_cached_wikipedia_cache_key_normalization(title: str, expected: str) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
    )

    full = CachedWikipediaClient._cache_key("enwiki", title, fetch_full_text=True)
    lead = CachedWikipediaClient._cache_key("enwiki", title, fetch_full_text=False)
    assert full == f"wikipedia/full-text-v2/enwiki/{expected}.json"
    assert lead == f"wikipedia/lead-only-v2/enwiki/{expected}.json"


# ---------------------------------------------------------------------------
# Cache-hit short-circuit
# ---------------------------------------------------------------------------


def test_cached_wikipedia_hit_skips_inner_fetch() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        WikipediaArticle,
    )

    sentinel_article = WikipediaArticle(
        language="en",
        site="enwiki",
        title="Hit",
        page_id=1,
        revision_id=1,
        revision_timestamp="",
        url="",
        lead_text="",
        extract="",
        full_text="",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="",
        attribution="",
        source_api="",
        retrieved_at="",
    )
    inner_calls: list[tuple[str, str]] = []

    class _Inner:
        def fetch_article(self, language, site, title, **_kw):
            inner_calls.append((language, site, title))
            raise AssertionError("inner must not be called on a cache hit")

    cached_payload = {
        "language": "en",
        "site": "enwiki",
        "title": "Hit",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "",
        "url": "",
        "lead_text": "",
        "extract": "",
        "full_text": "",
        "full_text_format": "plain_text",
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": [],
        "license": "",
        "attribution": "",
        "source_api": "",
        "retrieved_at": "",
    }
    cache = type(
        "C",
        (),
        {
            "get": lambda self, key: _cache_entry("ok", cached_payload, None),
            "set": lambda self, *args, **kw: None,
        },
    )()

    client = CachedWikipediaClient(_Inner(), cache)  # type: ignore[arg-type]
    result = client.fetch_article("en", "enwiki", "Hit")

    assert result.status == "ok"
    assert result.article == sentinel_article
    assert inner_calls == []


# ---------------------------------------------------------------------------
# Corrupt / malformed cached payload behaviour
# ---------------------------------------------------------------------------


def test_cached_wikipedia_corrupt_payload_falls_back_to_inner() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
        WikipediaArticle,
    )

    fresh = FetchResult(
        "ok",
        WikipediaArticle(
            language="en",
            site="enwiki",
            title="Fresh",
            page_id=1,
            revision_id=1,
            revision_timestamp="",
            url="",
            lead_text="",
            extract="",
            full_text="",
            full_text_format="plain_text",
            thumbnail_url="",
            thumbnail_width=None,
            thumbnail_height=None,
            categories=[],
            license="",
            attribution="",
            source_api="",
            retrieved_at="",
        ),
    )

    class _Inner:
        def fetch_article(self, *_args, **_kw):
            return fresh

    cache = type(
        "C",
        (),
        {
            # Simulates a cache hit whose payload is not a dict (corrupt).
            "get": lambda self, key: _cache_entry("ok", "not-a-dict", None),
            "set": lambda self, *args, **kw: None,
        },
    )()

    client = CachedWikipediaClient(_Inner(), cache)  # type: ignore[arg-type]
    result = client.fetch_article("en", "enwiki", "Fresh")
    assert result == fresh


def test_cached_wikipedia_error_status_hit_is_treated_as_miss() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
    )

    inner = type(
        "I",
        (),
        {"fetch_article": lambda self, *a, **kw: FetchResult("http_error", None, "boom")},
    )()

    cache = type(
        "C",
        (),
        {
            "get": lambda self, key: _cache_entry("error", None, None),
            "set": lambda self, *args, **kw: None,
        },
    )()

    client = CachedWikipediaClient(inner, cache)  # type: ignore[arg-type]
    result = client.fetch_article("en", "enwiki", "Boom")
    assert result.status == "http_error"


# ---------------------------------------------------------------------------
# Batch fetch: per-title selection
# ---------------------------------------------------------------------------


def test_cached_wikipedia_batch_returns_per_title_results() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
        InMemoryWikipediaClient,
    )

    inner = InMemoryWikipediaClient(
        {
            ("enwiki", "A"): FetchResult("article_not_found", None),
            ("enwiki", "B"): FetchResult("article_not_found", None),
        }
    )
    cache = type(
        "C",
        (),
        {"get": lambda self, key: None, "set": lambda self, *a, **kw: None},
    )()

    client = CachedWikipediaClient(inner, cache)  # type: ignore[arg-type]
    results = client.fetch_articles("en", "enwiki", ["A", "B"], fetch_full_text=True)

    assert set(results) == {"A", "B"}
    assert all(r.status == "article_not_found" for r in results.values())


def test_cached_wikipedia_batch_with_lead_only_uses_inner_batch_path() -> None:
    """When ``fetch_full_text=False`` the cache must use the inner batch path."""
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
    )

    class _BatchInner:
        def __init__(self) -> None:
            self.batch_called = False
            self.per_title_called = False

        def fetch_articles(self, language, site, titles, *, fetch_full_text=True):
            self.batch_called = True
            return {title: FetchResult("article_not_found", None) for title in titles}

        def fetch_article(self, *_args, **_kw):
            self.per_title_called = True
            return FetchResult("article_not_found", None)

    inner = _BatchInner()
    cache = type(
        "C",
        (),
        {"get": lambda self, key: None, "set": lambda self, *a, **kw: None},
    )()

    client = CachedWikipediaClient(inner, cache)  # type: ignore[arg-type]
    client.fetch_articles("en", "enwiki", ["A"], fetch_full_text=False)
    assert inner.batch_called
    assert not inner.per_title_called


# ---------------------------------------------------------------------------
# HTTP client: 429 -> rate_limited mapping (no augmentation warning path)
# ---------------------------------------------------------------------------


def test_http_wikipedia_429_returns_rate_limited_not_throttle_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Wikipedia HTTP client must surface 429 as a ``rate_limited``
    :class:`FetchResult`, must NOT emit the augmentation-style
    ``Wikimedia throttled`` warning (that lives in the augmentation
    client), and must notify the scheduler exactly once.

    Note: the Wikipedia transport itself does not emit any log
    records in the success / throttling paths, so there is no
    logger-name invariant to test at this layer (unlike Wikidata's
    batch-failure warning).
    """
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    session = _StubSession([_http_error(429, retry_after="7")])
    client = HttpWikipediaClient(settings, scheduler=scheduler)
    client._session = session

    with caplog.at_level(
        logging.WARNING,
        logger="osm_polygon_wikidata_only.enrichment.wikipedia_client",
    ):
        result = client.fetch_article("en", "enwiki", "X")

    assert result.status == "rate_limited"
    augmentation_records = [r for r in caplog.records if "Wikimedia throttled" in r.getMessage()]
    assert augmentation_records == []
    assert scheduler.throttle_calls == [("en.wikipedia.org", 7.0)]


# ---------------------------------------------------------------------------
# HTTP client: action API fallback on empty_text
# ---------------------------------------------------------------------------


def test_http_wikipedia_falls_back_to_parse_when_extract_empty() -> None:
    """When the initial response is ``empty_text`` and a revision id is
    available, the client must invoke the Action API parse fallback and
    synthesize the final result.
    """
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        HttpWikipediaClient,
    )

    settings = _make_settings()
    scheduler = _RecordingScheduler()
    query_body = (
        b'{"query": {"pages": {"0": {"pageid": 1, "title": "X", "fullurl": "u", '
        b'"revisions": [{"revid": 99, "timestamp": "t"}], '
        b'"extract": ""}}}}'
    )
    parse_body = b'{"parse": {"text": {"*": "<p>Body</p>"}}}'
    session = _StubSession([(query_body, "identity"), (parse_body, "identity")])
    client = HttpWikipediaClient(settings, scheduler=scheduler)
    client._session = session

    result = client.fetch_article("en", "enwiki", "X", fetch_full_text=True)

    assert result.status == "ok"
    assert result.article is not None
    assert result.article.source_api == "mediawiki_action_api_parse_fallback"
    assert "Body" in result.article.full_text
    assert len(session.reads) == 2
