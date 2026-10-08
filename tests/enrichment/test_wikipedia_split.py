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
from collections.abc import Iterable

import pytest

from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
    BatchWikipediaClient,
    FetchResult,
    WikipediaArticle,
    WikipediaClient,
)
from tests.enrichment._client_fakes import (
    _cache_entry,
    _make_settings,
    _MemoryCache,
    _RecordingScheduler,
    _StubSession,
)
from tests.helpers import http_error as _http_error

# ---------------------------------------------------------------------------
# Fakes (mirror the style used by ``test_wikimedia_transport_clients.py``)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Identity: facade surface preserved
# ---------------------------------------------------------------------------


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
    assert client.scheduler is scheduler


def test_cached_wikipedia_constructor_signature_and_failed_ttl_default() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
    )

    assert CachedWikipediaClient.__init__.__kwdefaults__ == {"failed_ttl_s": 60 * 60}


# ---------------------------------------------------------------------------
# Cache round-trip: successful article serialization
# ---------------------------------------------------------------------------


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
    cache = _MemoryCache()
    client = CachedWikipediaClient(inner, cache)

    result = client.fetch_article("en", "enwiki", "Monaco")

    assert result == FetchResult("ok", article)
    stored = cache.writes
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
    cache = _MemoryCache()
    client = CachedWikipediaClient(inner, cache, failed_ttl_s=123)

    result = client.fetch_article("en", "enwiki", "Missing")

    assert result.status == "article_not_found"
    stored = cache.writes
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
    inner_calls: list[tuple[str, str, str]] = []

    class _Inner(WikipediaClient):
        def fetch_article(
            self,
            language: str,
            site: str,
            title: str,
            *,
            wikidata_label: str = "",
            wikidata_description: str = "",
            wikidata_aliases: list[str] | None = None,
            fetch_full_text: bool = True,
        ) -> FetchResult:
            del wikidata_label, wikidata_description, wikidata_aliases, fetch_full_text
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
    cache = _MemoryCache(
        {"wikipedia/full-text-v2/enwiki/Hit.json": _cache_entry("ok", cached_payload, None)}
    )

    client = CachedWikipediaClient(_Inner(), cache)
    result = client.fetch_article("en", "enwiki", "Hit")

    assert result.status == "ok"
    assert result.article == sentinel_article
    assert inner_calls == []


# ---------------------------------------------------------------------------
# Parse-fallback text written before the block-boundary fix (#201)
# ---------------------------------------------------------------------------

_CURRENT_CLEANING = "html-block-boundaries-v1"
_FALLBACK_SOURCE = "mediawiki_action_api_parse_fallback"


def _stored_payload(source_api: str, marker: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "language": "en",
        "site": "enwiki",
        "title": "Stale",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "",
        "url": "",
        "lead_text": "stale lead",
        "extract": "",
        "full_text": "stale full",
        "full_text_format": "plain_text",
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": [],
        "license": "",
        "attribution": "",
        "source_api": source_api,
        "retrieved_at": "",
    }
    if marker is not None:
        payload["text_cleaning_version"] = marker
    return payload


def _fresh_article() -> WikipediaArticle:
    return WikipediaArticle(
        language="en",
        site="enwiki",
        title="Stale",
        page_id=1,
        revision_id=1,
        revision_timestamp="",
        url="",
        lead_text="fresh lead",
        extract="",
        full_text="fresh full",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="",
        attribution="",
        source_api=_FALLBACK_SOURCE,
        retrieved_at="",
    )


class _RecordingWikipedia(WikipediaClient):
    def __init__(self, article: WikipediaArticle) -> None:
        self._article = article
        self.calls: list[bool] = []

    def fetch_article(
        self,
        language: str,
        site: str,
        title: str,
        *,
        wikidata_label: str = "",
        wikidata_description: str = "",
        wikidata_aliases: list[str] | None = None,
        fetch_full_text: bool = True,
    ) -> FetchResult:
        del language, site, title, wikidata_label, wikidata_description, wikidata_aliases
        self.calls.append(fetch_full_text)
        return FetchResult("ok", self._article)


@pytest.mark.parametrize(
    ("fetch_full_text", "source_api", "marker", "expect_refetch"),
    [
        pytest.param(True, "mediawiki_action_api", None, False, id="normal-entry-hits"),
        pytest.param(
            True, "mediawiki_action_api", _CURRENT_CLEANING, False, id="normal-marked-entry-hits"
        ),
        pytest.param(True, _FALLBACK_SOURCE, None, True, id="fallback-without-marker-refetches"),
        pytest.param(
            True, _FALLBACK_SOURCE, "older-cleaning", True, id="fallback-older-marker-refetches"
        ),
        pytest.param(
            True, _FALLBACK_SOURCE, _CURRENT_CLEANING, False, id="fallback-current-marker-hits"
        ),
        pytest.param(
            False, _FALLBACK_SOURCE, None, True, id="lead-only-fallback-without-marker-refetches"
        ),
    ],
)
def test_cached_wikipedia_refetches_only_stale_parse_fallback_entries(
    fetch_full_text: bool, source_api: str, marker: str | None, expect_refetch: bool
) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import CachedWikipediaClient

    key = CachedWikipediaClient._cache_key("enwiki", "Stale", fetch_full_text)
    cache = _MemoryCache({key: _cache_entry("ok", _stored_payload(source_api, marker), None)})
    inner = _RecordingWikipedia(_fresh_article())

    result = CachedWikipediaClient(inner, cache).fetch_article(
        "en", "enwiki", "Stale", fetch_full_text=fetch_full_text
    )

    if expect_refetch:
        assert inner.calls == [fetch_full_text]
        assert result.article is not None
        assert result.article.full_text == "fresh full"
    else:
        assert inner.calls == []
        assert result.article is not None
        assert result.article.full_text == "stale full"


def test_cached_wikipedia_refetched_parse_fallback_is_stored_with_marker() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import CachedWikipediaClient

    key = CachedWikipediaClient._cache_key("enwiki", "Stale", fetch_full_text=True)
    cache = _MemoryCache({key: _cache_entry("ok", _stored_payload(_FALLBACK_SOURCE, None), None)})
    CachedWikipediaClient(_RecordingWikipedia(_fresh_article()), cache).fetch_article(
        "en", "enwiki", "Stale"
    )

    ((stored_key, stored_payload, stored_kwargs),) = cache.writes
    assert stored_key == key
    assert stored_kwargs["status"] == "ok"
    assert stored_payload["text_cleaning_version"] == _CURRENT_CLEANING

    # The refreshed entry is a hit on the next read, so the refetch runs once.
    next_inner = _RecordingWikipedia(_fresh_article())
    next_cache = _MemoryCache({key: _cache_entry("ok", stored_payload, None)})
    result = CachedWikipediaClient(next_inner, next_cache).fetch_article("en", "enwiki", "Stale")
    assert next_inner.calls == []
    assert result.article is not None
    assert result.article.full_text == "fresh full"


def test_cached_wikipedia_batch_refetches_stale_parse_fallback_entries() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import CachedWikipediaClient

    key = CachedWikipediaClient._cache_key("enwiki", "Stale", fetch_full_text=True)
    cache = _MemoryCache({key: _cache_entry("ok", _stored_payload(_FALLBACK_SOURCE, None), None)})
    inner = _RecordingWikipedia(_fresh_article())

    results = CachedWikipediaClient(inner, cache).fetch_articles(
        "en", "enwiki", ["Stale"], fetch_full_text=True
    )

    assert inner.calls == [True]
    assert results["Stale"].article is not None
    assert results["Stale"].article.full_text == "fresh full"


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

    class _Inner(WikipediaClient):
        def fetch_article(
            self,
            language: str,
            site: str,
            title: str,
            *,
            wikidata_label: str = "",
            wikidata_description: str = "",
            wikidata_aliases: list[str] | None = None,
            fetch_full_text: bool = True,
        ) -> FetchResult:
            del language, site, title, wikidata_label, wikidata_description
            del wikidata_aliases, fetch_full_text
            return fresh

    cache = _MemoryCache(
        {"wikipedia/full-text-v2/enwiki/Fresh.json": _cache_entry("ok", "not-a-dict", None)}
    )

    client = CachedWikipediaClient(_Inner(), cache)
    result = client.fetch_article("en", "enwiki", "Fresh")
    assert result == fresh


def test_cached_wikipedia_error_status_hit_is_treated_as_miss() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
    )

    class _Inner(WikipediaClient):
        def fetch_article(
            self,
            language: str,
            site: str,
            title: str,
            *,
            wikidata_label: str = "",
            wikidata_description: str = "",
            wikidata_aliases: list[str] | None = None,
            fetch_full_text: bool = True,
        ) -> FetchResult:
            del language, site, title, wikidata_label, wikidata_description
            del wikidata_aliases, fetch_full_text
            return FetchResult("http_error", None, "boom")

    cache = _MemoryCache(
        {"wikipedia/full-text-v2/enwiki/Boom.json": _cache_entry("error", None, None)}
    )

    client = CachedWikipediaClient(_Inner(), cache)
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
    cache = _MemoryCache()

    client = CachedWikipediaClient(inner, cache)
    results = client.fetch_articles("en", "enwiki", ["A", "B"], fetch_full_text=True)

    assert set(results) == {"A", "B"}
    assert all(r.status == "article_not_found" for r in results.values())


def test_cached_wikipedia_batch_with_lead_only_uses_inner_batch_path() -> None:
    """When ``fetch_full_text=False`` the cache must use the inner batch path."""
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        CachedWikipediaClient,
        FetchResult,
    )

    class _BatchInner(WikipediaClient, BatchWikipediaClient):
        def __init__(self) -> None:
            self.batch_called = False
            self.per_title_called = False

        def fetch_articles(
            self,
            language: str,
            site: str,
            titles: Iterable[str],
            *,
            fetch_full_text: bool = True,
        ) -> dict[str, FetchResult]:
            del language, site, fetch_full_text
            self.batch_called = True
            return {title: FetchResult("article_not_found", None) for title in titles}

        def fetch_article(
            self,
            language: str,
            site: str,
            title: str,
            *,
            wikidata_label: str = "",
            wikidata_description: str = "",
            wikidata_aliases: list[str] | None = None,
            fetch_full_text: bool = True,
        ) -> FetchResult:
            del language, site, title, wikidata_label, wikidata_description
            del wikidata_aliases, fetch_full_text
            self.per_title_called = True
            return FetchResult("article_not_found", None)

    inner = _BatchInner()
    cache = _MemoryCache()

    client = CachedWikipediaClient(inner, cache)
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
    client = HttpWikipediaClient(settings, scheduler=scheduler, session=session)

    with caplog.at_level(
        logging.WARNING,
        logger="osm_polygon_wikidata_only.enrichment.wikipedia_client",
    ):
        result = client.fetch_article("en", "enwiki", "X")

    assert result.status == "rate_limited"
    augmentation_records = [r for r in caplog.records if "Wikimedia throttled" in r.getMessage()]
    assert augmentation_records == []
    assert scheduler.throttle_calls == [("en.wikipedia.org", 7.0)]


@pytest.mark.parametrize(
    ("code", "fallback", "status"),
    [
        (404, False, "article_not_found"),
        (429, False, "rate_limited"),
        (503, True, "rate_limited"),
        (500, True, "http_error"),
    ],
)
def test_wikipedia_http_error_mapping_is_explicit(code: int, fallback: bool, status: str) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import transport

    result = transport._article_http_error(_http_error(code), fallback=fallback)

    assert result.status == status
    assert result.article is None


@pytest.mark.parametrize("fallback", [False, True])
def test_wikipedia_network_error_mapping_preserves_fallback_context(fallback: bool) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import transport

    result = transport._article_network_error(ConnectionError("offline"), fallback=fallback)

    assert result.status == "http_error"
    assert result.article is None
    expected = "parse fallback failed: offline" if fallback else "offline"
    assert result.error == expected


@pytest.mark.parametrize("has_article", [False, True])
def test_empty_fallback_result_preserves_available_article(has_article: bool) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import transport

    article = (
        WikipediaArticle(
            language="en",
            site="enwiki",
            title="Fallback",
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
        if has_article
        else None
    )
    result = transport._empty_fallback_result(transport.FetchResult("empty_text", article))

    assert result.status == "empty_text"
    assert result.article is article
    assert result.error == "extract and exact-revision parse were empty"


def test_wikipedia_lead_batch_error_falls_back_to_full_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import transport

    client = transport.HttpWikipediaClient(
        _make_settings(), scheduler=_RecordingScheduler(), session=_StubSession(responses=[])
    )
    fallback = {"Alpha": transport.FetchResult("article_not_found", None)}
    monkeypatch.setattr(client, "_fetch_full_text_batch", lambda *_args: fallback)
    monkeypatch.setattr(
        client,
        "_request_article_data",
        lambda *_args, **_kwargs: (None, transport.FetchResult("http_error", None)),
    )

    assert client._fetch_lead_batch("en", "enwiki", ["Alpha"]) == fallback


def test_wikipedia_lead_batch_parse_error_falls_back_to_full_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia import transport

    client = transport.HttpWikipediaClient(
        _make_settings(), scheduler=_RecordingScheduler(), session=_StubSession(responses=[])
    )
    fallback = {"Alpha": transport.FetchResult("article_not_found", None)}
    monkeypatch.setattr(client, "_fetch_full_text_batch", lambda *_args: fallback)
    monkeypatch.setattr(
        client,
        "_request_article_data",
        lambda *_args, **_kwargs: ({"query": {}}, None),
    )
    monkeypatch.setattr(
        transport,
        "_parse_wikipedia_batch_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad payload")),
    )

    assert client._fetch_lead_batch("en", "enwiki", ["Alpha"]) == fallback


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
    client = HttpWikipediaClient(settings, scheduler=scheduler, session=session)

    result = client.fetch_article("en", "enwiki", "X", fetch_full_text=True)

    assert result.status == "ok"
    assert result.article is not None
    assert result.article.source_api == "mediawiki_action_api_parse_fallback"
    assert "Body" in result.article.full_text
    assert len(session.reads) == 2
