"""Tests for the enrichment stack: QID validation, parsing, clients, linker, text cleaning."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment import article_linker
from osm_polygon_wikidata_only.enrichment.progress import EnrichmentProgress
from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
    html_to_plain_text,
    normalize_unicode,
    normalize_whitespace,
    strip_template_markers,
)
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    BatchWikidataClient,
    CachedWikidataClient,
    HttpWikidataClient,
    InMemoryWikidataClient,
    WikidataEntity,
    is_valid_qid,
    language_from_site,
    parse_wikidata_entity,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    BatchWikipediaClient,
    CachedWikipediaClient,
    FetchResult,
    HttpWikipediaClient,
    InMemoryWikipediaClient,
    WikipediaArticle,
    parse_wikipedia_response,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler


class RecordingSession:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.requests: list[urllib.request.Request] = []

    def read(
        self,
        request: urllib.request.Request,
        *,
        min_interval_anonymous_s: float = 0.0,
        min_interval_authenticated_s: float = 0.0,
    ) -> tuple[bytes, str]:
        self.requests.append(request)
        return json.dumps(self._payload).encode(), ""


class ThrottledSession:
    def read(
        self,
        request: urllib.request.Request,
        *,
        min_interval_anonymous_s: float = 0.0,
        min_interval_authenticated_s: float = 0.0,
    ) -> tuple[bytes, str]:
        headers = Message()
        headers["Retry-After"] = "17"
        raise urllib.error.HTTPError(request.full_url, 429, "limited", headers, None)


# --- QID validation ------------------------------------------------------


@pytest.mark.parametrize("qid", ["Q1", "Q42", "Q9999999"])
def test_is_valid_qid_accepts_well_formed(qid: str) -> None:
    assert is_valid_qid(qid)


@pytest.mark.parametrize(
    "qid",
    ["", "Q0", "Q-1", "q42", "Q42a", "Q 42", "P42", "X1", "Q"],
)
def test_is_valid_qid_rejects_garbage(qid: str) -> None:
    assert not is_valid_qid(qid)


# --- language_from_site --------------------------------------------------


def test_language_from_site_strips_wiki() -> None:
    assert language_from_site("enwiki") == "en"
    assert language_from_site("frwiki") == "fr"
    assert language_from_site("dewiki") == "de"
    assert language_from_site("zh_min_nanwiki") == "zh-min-nan"


def test_language_from_site_passthrough_for_unknown() -> None:
    assert language_from_site("en") == "en"


# --- parse_wikidata_entity ----------------------------------------------


def test_parse_wikidata_entity_extracts_sitelinks() -> None:
    data = {
        "entities": {
            "Q42": {
                "sitelinks": {
                    "enwiki": {"title": "Douglas Adams"},
                    "frwiki": {"title": "Douglas Adams"},
                    "simplewiki": {"title": "Douglas Adams"},
                    "zh_min_nanwiki": {"title": "Douglas Adams"},
                    "commonswiki": {"title": "Douglas Adams"},  # not a wiki
                },
                "labels": {"en": {"value": "Douglas Adams"}},
                "descriptions": {"en": {"value": "English writer"}},
            }
        }
    }
    entity = parse_wikidata_entity("Q42", data)
    assert entity is not None
    assert entity.qid == "Q42"
    assert entity.sitelinks == {
        "enwiki": "Douglas Adams",
        "frwiki": "Douglas Adams",
        "simplewiki": "Douglas Adams",
        "zh_min_nanwiki": "Douglas Adams",
    }
    assert entity.labels == {"en": "Douglas Adams"}


def test_parse_wikidata_entity_returns_none_for_missing() -> None:
    assert parse_wikidata_entity("Q42", {"entities": {"Q42": {"missing": "x"}}}) is None
    assert parse_wikidata_entity("Q42", {"entities": {}}) is None


def test_parse_wikidata_entity_drops_non_wikipedia_wikifunctions_sitelink() -> None:
    """Wikifunctions is hosted at ``wikifunctions.org`` and must not be fetched.

    Regression test for a real pipeline failure on Q150 (Wikidata):
    the entity returns ``wikifunctionswiki`` as a sitelink, which is not
    a Wikipedia host. Treating it as a language Wikipedia triggers a
    DNS lookup for ``wikifunctions.wikipedia.org`` that always fails and
    aborts the entire PBF with ``IncompleteEnrichmentError``.
    """
    data = {
        "entities": {
            "Q150": {
                "sitelinks": {
                    "enwiki": {"title": "Universe"},
                    "wikifunctionswiki": {"title": "Z10000"},
                },
                "labels": {"en": {"value": "Universe"}},
            }
        }
    }
    entity = parse_wikidata_entity("Q150", data)
    assert entity is not None
    assert entity.sitelinks == {"enwiki": "Universe"}


# --- InMemoryWikidataClient ---------------------------------------------


def test_in_memory_wikidata_client_returns_entity() -> None:
    entity = WikidataEntity(qid="Q42", sitelinks={"enwiki": "Foo"})
    client = InMemoryWikidataClient({"Q42": entity})
    assert client.get_entity("Q42") is entity


def test_in_memory_wikidata_client_rejects_invalid_qid() -> None:
    client = InMemoryWikidataClient({})
    assert client.get_entity("not-a-qid") is None


def test_in_memory_wikidata_client_returns_none_for_missing() -> None:
    client = InMemoryWikidataClient({})
    assert client.get_entity("Q9999999") is None


def test_http_wikidata_client_get_entities_parses_each_qid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HttpWikidataClient(Settings())
    monkeypatch.setattr(
        client,
        "_http_get",
        lambda url: {
            "entities": {
                "Q1": {"sitelinks": {"enwiki": {"title": "One"}}},
                "Q2": {"sitelinks": {"enwiki": {"title": "Two"}}},
            }
        },
    )
    assert [entity.qid if entity else None for entity in client.get_entities(["Q1", "Q2"])] == [
        "Q1",
        "Q2",
    ]


def test_http_wikidata_client_routes_requests_through_injected_session() -> None:
    session = RecordingSession({"entities": {}})
    client = HttpWikidataClient(Settings(), session=session)

    assert client._http_get(client._build_url("Q1")) == {"entities": {}}

    assert len(session.requests) == 1
    headers = dict(session.requests[0].header_items())
    assert headers["User-agent"] == Settings().user_agent
    assert headers["Accept-encoding"] == "gzip"


def test_http_wikidata_client_reports_429_as_host_scoped_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 from one host must be scoped to that host, never global."""
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)
    host_reports: list[tuple[str, float]] = []
    global_reports: list[float] = []
    monkeypatch.setattr(
        scheduler, "report_host_throttled", lambda host, delay: host_reports.append((host, delay))
    )
    monkeypatch.setattr(scheduler, "report_throttled", lambda delay: global_reports.append(delay))
    monkeypatch.setattr(scheduler, "pace_host", lambda *_, **__: None)
    client = HttpWikidataClient(Settings(), scheduler=scheduler, session=ThrottledSession())

    with pytest.raises(urllib.error.HTTPError):
        client._http_get(client._build_url("Q1"))

    assert host_reports == [("www.wikidata.org", 17.0)]
    assert global_reports == []


# --- CachedWikidataClient -----------------------------------------------


def test_cached_wikidata_client_serves_from_cache(tmp_path: Path) -> None:
    entity = WikidataEntity(qid="Q42", sitelinks={"enwiki": "Foo"})
    inner = InMemoryWikidataClient({"Q42": entity})
    cache = JsonFileCache(tmp_path)
    client = CachedWikidataClient(inner, cache)

    assert client.get_entity("Q42") is entity
    # Replace the underlying mapping; the cache should still serve the
    # first value because it was cached on first access.
    inner._mapping = {}  # type: ignore[attr-defined]
    out = client.get_entity("Q42")
    assert out is not None
    assert out.qid == "Q42"


def test_cached_wikidata_client_caches_failures(tmp_path: Path) -> None:
    client = CachedWikidataClient(InMemoryWikidataClient({}), JsonFileCache(tmp_path))
    assert client.get_entity("Q1") is None
    # Second call: no exception, no crash.
    assert client.get_entity("Q1") is None


def test_cached_wikidata_client_batches_only_cache_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = InMemoryWikidataClient({"Q1": WikidataEntity(qid="Q1")})
    client = CachedWikidataClient(inner, JsonFileCache(tmp_path))
    client.get_entity("Q1")
    calls: list[list[str]] = []

    def get_entities(qids: list[str]) -> list[WikidataEntity | None]:
        calls.append(qids)
        return [WikidataEntity(qid=qid) for qid in qids]

    monkeypatch.setattr(inner, "get_entities", get_entities, raising=False)
    result = client.get_entities(["Q1", "Q2"])
    assert [entity.qid if entity else None for entity in result] == ["Q1", "Q2"]
    assert calls == [["Q2"]]


# --- text_cleaning ------------------------------------------------------


def test_normalize_whitespace_collapses_runs() -> None:
    assert normalize_whitespace("a  b\n\nc\t\td") == "a b c d"


def test_strip_template_markers_removes_braces() -> None:
    out = strip_template_markers("Hello {{convert|1|m}} world")
    assert out == "Hello  world"


def test_normalize_unicode_nfc() -> None:
    # e + combining acute = NFD; the composed form is the NFC of "é".
    nfd = "é"
    nfc = normalize_unicode(nfd, "NFC")
    assert nfc == "é"
    assert len(nfc) == 1


def test_clean_article_text_does_it_all() -> None:
    raw = "Hello  {{convert|1|m}}  world\n\nfoo"
    out = clean_article_text(raw)
    assert "  " not in out
    assert "{{" not in out
    assert "Hello" in out and "foo" in out


def test_count_words() -> None:
    assert count_words("") == 0
    assert count_words("hello world") == 2
    assert count_words("  a  b  c  ") == 3


def test_estimate_tokens_floor_at_1_for_short_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") == 1
    assert estimate_tokens("a" * 40) == 10


def test_html_to_plain_text_keeps_rendered_article_text() -> None:
    html = "<div><p>First <b>paragraph</b>.</p><h2>History</h2><p>Second&nbsp;part.</p></div>"
    assert html_to_plain_text(html) == "First paragraph. History Second part."


# --- parse_wikipedia_response ------------------------------------------


def _sample_wiki_response(
    *,
    page_id: int = 123,
    revid: int = 999,
    extract: str = "Hello world.\n\nFirst paragraph. Second paragraph.",
    fullurl: str = "https://en.wikipedia.org/wiki/Hello",
    title: str = "Hello",
    thumb: dict | None = None,
) -> dict:
    return {
        "query": {
            "pages": {
                str(page_id): {
                    "pageid": page_id,
                    "title": title,
                    "extract": extract,
                    "revisions": [{"revid": revid, "timestamp": "2026-01-02T03:04:05Z"}],
                    "fullurl": fullurl,
                    **({"thumbnail": thumb} if thumb else {}),
                }
            }
        }
    }


def test_parse_wikipedia_response_returns_ok() -> None:
    data = _sample_wiki_response()
    res = parse_wikipedia_response("en", "enwiki", "Hello", data)
    assert res.status == "ok"
    assert res.article is not None
    assert res.article.page_id == 123
    assert res.article.revision_id == 999
    assert res.article.url.endswith("/wiki/Hello")
    assert "Hello world" in res.article.full_text


def test_parse_wikipedia_response_empty_text_is_empty_text() -> None:
    data = _sample_wiki_response(extract="")
    res = parse_wikipedia_response("en", "enwiki", "Hello", data)
    assert res.status == "empty_text"
    # Metadata is preserved on empty body so the linker can build an
    # Article row with fetch_status="empty_text".
    assert res.article is not None
    assert res.article.page_id == 123
    assert res.article.revision_id == 999
    assert res.article.full_text == ""


def test_parse_wikipedia_response_article_not_found() -> None:
    data = {"query": {"pages": {"-1": {"missing": "", "title": "Hello"}}}}
    res = parse_wikipedia_response("en", "enwiki", "Hello", data)
    assert res.status == "article_not_found"


def test_parse_wikipedia_response_thumbnail_metadata() -> None:
    data = _sample_wiki_response(thumb={"source": "https://x/y.jpg", "width": 200, "height": 100})
    res = parse_wikipedia_response("en", "enwiki", "Hello", data)
    assert res.article is not None
    assert res.article.thumbnail_url == "https://x/y.jpg"
    assert res.article.thumbnail_width == 200
    assert res.article.thumbnail_height == 100


# --- InMemoryWikipediaClient + Cached ---------------------------------


def _sample_article(
    language: str = "en", title: str = "Foo", body: str = "Some text."
) -> WikipediaArticle:
    return WikipediaArticle(
        language=language,
        site=f"{language}wiki",
        title=title,
        page_id=1,
        revision_id=10,
        revision_timestamp="2026-01-01T00:00:00Z",
        url=f"https://{language}.wikipedia.org/wiki/{title}",
        lead_text=body,
        extract=body,
        full_text=body,
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
        source_api="mediawiki_action_api",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def test_in_memory_wikipedia_client_returns_known() -> None:
    art = _sample_article()
    client = InMemoryWikipediaClient({("enwiki", "Foo"): FetchResult("ok", art)})
    res = client.fetch_article("en", "enwiki", "Foo")
    assert res.status == "ok"
    assert res.article is art


def test_in_memory_wikipedia_client_missing_article() -> None:
    client = InMemoryWikipediaClient({})
    res = client.fetch_article("en", "enwiki", "Nope")
    assert res.status == "article_not_found"


def test_http_wikipedia_client_fetch_articles_returns_results_by_requested_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HttpWikipediaClient(Settings())
    data = {
        "query": {
            "pages": {
                "1": _sample_wiki_response(page_id=1, title="Alpha")["query"]["pages"]["1"],
                "2": _sample_wiki_response(page_id=2, title="Beta")["query"]["pages"]["2"],
            }
        }
    }
    monkeypatch.setattr(client, "_http_get", lambda url: data)
    results = client.fetch_articles("en", "enwiki", ["Alpha", "Beta"], fetch_full_text=False)
    assert [results[title].article.title for title in ("Alpha", "Beta")] == ["Alpha", "Beta"]


def test_http_wikipedia_client_routes_requests_through_injected_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"query": {"pages": {}}}
    session = RecordingSession(payload)
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)
    monkeypatch.setattr(scheduler, "pace_host", lambda *_, **__: None)
    client = HttpWikipediaClient(Settings(), scheduler=scheduler, session=session)

    url = client._build_url("en", "Alpha", fetch_full_text=True)
    assert client._http_get(url) == payload

    assert len(session.requests) == 1
    headers = dict(session.requests[0].header_items())
    assert headers["User-agent"] == Settings().user_agent
    assert headers["Accept"] == "application/json"
    assert headers["Accept-encoding"] == "gzip"


def test_http_wikipedia_client_reports_429_as_host_scoped_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 from one host must be scoped to that host, never global."""
    scheduler = AdaptiveRequestScheduler(requests_per_minute=100_000)
    host_reports: list[tuple[str, float]] = []
    global_reports: list[float] = []
    monkeypatch.setattr(
        scheduler, "report_host_throttled", lambda host, delay: host_reports.append((host, delay))
    )
    monkeypatch.setattr(scheduler, "report_throttled", lambda delay: global_reports.append(delay))
    monkeypatch.setattr(scheduler, "pace_host", lambda *_, **__: None)
    client = HttpWikipediaClient(Settings(), scheduler=scheduler, session=ThrottledSession())

    with pytest.raises(urllib.error.HTTPError):
        client._http_get(client._build_url("en", "Alpha", fetch_full_text=True))

    assert host_reports == [("en.wikipedia.org", 17.0)]
    assert global_reports == []


def test_http_clients_request_maxlag_for_background_work() -> None:
    wiki_url = HttpWikipediaClient(Settings())._build_url("en", "Alpha", fetch_full_text=True)
    wd_url = HttpWikidataClient(Settings())._build_url("Q1")
    assert "maxlag=5" in wiki_url
    assert "maxlag=5" in wd_url


def test_full_text_article_batches_use_individual_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HttpWikipediaClient(Settings())
    calls: list[str] = []

    def fetch_article(language: str, site: str, title: str, **_: object) -> FetchResult:
        calls.append(title)
        return FetchResult("ok", _sample_article(language, title, "full text"))

    monkeypatch.setattr(client, "fetch_article", fetch_article)
    results = client.fetch_articles("en", "enwiki", ["Alpha", "Beta"], fetch_full_text=True)

    assert calls == ["Alpha", "Beta"]
    assert set(results) == {"Alpha", "Beta"}


def test_http_wikipedia_client_falls_back_to_exact_revision_parse_when_extract_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HttpWikipediaClient(Settings(request_base_delay_s=0))
    query = _sample_wiki_response(extract="", revid=540315, title="حکیم ترمذی")
    parsed = {
        "parse": {
            "title": "حکیم ترمذی",
            "pageid": 81195,
            "revid": 540315,
            "text": '<div class="mw-parser-output"><p>حکیم ترمذی مشہور صوفی نیں۔</p></div>',  # noqa: RUF001
        }
    }
    urls: list[str] = []

    def http_get(url: str) -> dict[str, object]:
        urls.append(url)
        return query if len(urls) == 1 else parsed

    monkeypatch.setattr(client, "_http_get", http_get)
    result = client.fetch_article("pnb", "pnbwiki", "حکیم ترمذی")
    assert result.status == "ok"
    assert result.article is not None
    assert result.article.revision_id == 540315
    assert "مشہور صوفی" in result.article.full_text
    assert "action=parse" in urls[1]
    assert "oldid=540315" in urls[1]


def test_cached_wikipedia_client_serves_from_cache(tmp_path: Path) -> None:
    art = _sample_article()
    inner = InMemoryWikipediaClient({("enwiki", "Foo"): FetchResult("ok", art)})
    cache = JsonFileCache(tmp_path)
    client = CachedWikipediaClient(inner, cache)

    res1 = client.fetch_article("en", "enwiki", "Foo")
    assert res1.status == "ok"
    inner._responses.clear()  # type: ignore[attr-defined]
    res2 = client.fetch_article("en", "enwiki", "Foo")
    assert res2.status == "ok"
    assert res2.article is not None
    assert res2.article.title == "Foo"


def test_cached_wikipedia_client_caches_failures(tmp_path: Path) -> None:
    inner = InMemoryWikipediaClient({})
    cache = JsonFileCache(tmp_path)
    client = CachedWikipediaClient(inner, cache)
    assert client.fetch_article("en", "enwiki", "X").status == "article_not_found"


def test_cached_wikipedia_client_does_not_reuse_transient_failure(tmp_path: Path) -> None:
    class RecoveringWikipedia(InMemoryWikipediaClient):
        calls = 0

        def fetch_article(
            self, language: str, site: str, title: str, **kwargs: object
        ) -> FetchResult:
            self.calls += 1
            if self.calls == 1:
                return FetchResult("http_error", None, "offline")
            return FetchResult("ok", _sample_article(language, title, "complete"))

    inner = RecoveringWikipedia({})
    client = CachedWikipediaClient(inner, JsonFileCache(tmp_path))
    assert client.fetch_article("en", "enwiki", "X").status == "http_error"
    assert client.fetch_article("en", "enwiki", "X").status == "ok"
    assert inner.calls == 2


def test_full_text_request_does_not_reuse_lead_only_cache(tmp_path: Path) -> None:
    class TextAwareWikipedia(InMemoryWikipediaClient):
        def __init__(self) -> None:
            super().__init__({})
            self.calls: list[bool] = []

        def fetch_article(
            self,
            language: str,
            site: str,
            title: str,
            *,
            fetch_full_text: bool = True,
            **kwargs: object,
        ) -> FetchResult:
            self.calls.append(fetch_full_text)
            text = "full body" if fetch_full_text else "lead"
            return FetchResult("ok", _sample_article(language, title, text))

    inner = TextAwareWikipedia()
    client = CachedWikipediaClient(inner, JsonFileCache(tmp_path))
    client.fetch_article("en", "enwiki", "X", fetch_full_text=False)
    result = client.fetch_article("en", "enwiki", "X", fetch_full_text=True)
    assert result.article is not None
    assert result.article.full_text == "full body"
    assert inner.calls == [False, True]


def test_cached_wikipedia_client_batches_only_cache_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    article = _sample_article(title="Foo")
    inner = InMemoryWikipediaClient({("enwiki", "Foo"): FetchResult("ok", article)})
    client = CachedWikipediaClient(inner, JsonFileCache(tmp_path))
    client.fetch_article("en", "enwiki", "Foo")
    calls: list[list[str]] = []

    def fetch_articles(
        language: str, site: str, titles: list[str], **_: object
    ) -> dict[str, FetchResult]:
        calls.append(titles)
        return {"Bar": FetchResult("ok", _sample_article(title="Bar"))}

    monkeypatch.setattr(inner, "fetch_articles", fetch_articles, raising=False)
    results = client.fetch_articles("en", "enwiki", ["Foo", "Bar"])
    assert [results[title].article.title for title in ("Foo", "Bar")] == ["Foo", "Bar"]
    assert calls == [["Bar"]]
    assert client.fetch_article("en", "enwiki", "X").status == "article_not_found"


# --- article_linker ---------------------------------------------------


def _entity_q42() -> WikidataEntity:
    return WikidataEntity(
        qid="Q42",
        sitelinks={"enwiki": "Foo", "frwiki": "Foo_fr"},
        labels={"en": "Foo", "fr": "Foo_fr"},
        descriptions={"en": "An entity"},
        aliases={"en": ["Foo alias"]},
    )


def test_link_qid_fetches_all_languages_by_default() -> None:
    wd = InMemoryWikidataClient({"Q42": _entity_q42()})
    en = InMemoryWikipediaClient(
        {
            ("enwiki", "Foo"): FetchResult("ok", _sample_article("en", "Foo", "en text")),
            ("frwiki", "Foo_fr"): FetchResult("ok", _sample_article("fr", "Foo_fr", "fr text")),
        }
    )
    summary = article_linker.link_qid("Q42", wikidata_client=wd, wikipedia_client=en)
    assert {a.language for a in summary.articles} == {"en", "fr"}


def test_link_qid_respects_language_filter() -> None:
    wd = InMemoryWikidataClient({"Q42": _entity_q42()})
    en = InMemoryWikipediaClient(
        {
            ("enwiki", "Foo"): FetchResult("ok", _sample_article("en", "Foo", "en text")),
            ("frwiki", "Foo_fr"): FetchResult("ok", _sample_article("fr", "Foo_fr", "fr text")),
        }
    )
    summary = article_linker.link_qid(
        "Q42", wikidata_client=wd, wikipedia_client=en, languages=("en",)
    )
    assert {a.language for a in summary.articles} == {"en"}


def test_link_qid_continues_when_one_language_fails() -> None:
    wd = InMemoryWikidataClient({"Q42": _entity_q42()})
    en = InMemoryWikipediaClient(
        {
            ("enwiki", "Foo"): FetchResult("article_not_found", None),
            ("frwiki", "Foo_fr"): FetchResult("ok", _sample_article("fr", "Foo_fr", "fr text")),
        }
    )
    summary = article_linker.link_qid("Q42", wikidata_client=wd, wikipedia_client=en)
    assert len(summary.articles) == 1
    assert summary.articles[0].language == "fr"
    assert summary.statuses["enwiki"] == "article_not_found"


def test_link_qid_invalid_qid_returns_empty() -> None:
    wd = InMemoryWikidataClient({})
    en = InMemoryWikipediaClient({})
    summary = article_linker.link_qid("not-a-qid", wikidata_client=wd, wikipedia_client=en)
    assert summary.entity is None
    assert summary.articles == []


def test_link_qid_wikidata_not_found() -> None:
    wd = InMemoryWikidataClient({})
    en = InMemoryWikipediaClient({})
    summary = article_linker.link_qid("Q9999", wikidata_client=wd, wikipedia_client=en)
    assert summary.entity is None


def test_link_qid_filters_non_wiki_sitelinks() -> None:
    entity = WikidataEntity(
        qid="Q1",
        sitelinks={
            "enwiki": "Foo",
            "commonswiki": "Foo",  # not a wiki
            "de.wikisource.org": "Foo",  # not a wiki suffix
        },
    )
    wd = InMemoryWikidataClient({"Q1": entity})
    en = InMemoryWikipediaClient(
        {("enwiki", "Foo"): FetchResult("ok", _sample_article("en", "Foo", "x"))}
    )
    summary = article_linker.link_qid("Q1", wikidata_client=wd, wikipedia_client=en)
    assert {a.language for a in summary.articles} == {"en"}


def test_link_qid_includes_long_and_compound_wikipedia_language_sites() -> None:
    entity = WikidataEntity(
        qid="Q1",
        sitelinks={"simplewiki": "Foo", "zh_min_nanwiki": "Foo_nan"},
    )
    wd = InMemoryWikidataClient({"Q1": entity})
    wiki = InMemoryWikipediaClient(
        {
            ("simplewiki", "Foo"): FetchResult("ok", _sample_article("simple", "Foo", "x")),
            ("zh_min_nanwiki", "Foo_nan"): FetchResult(
                "ok", _sample_article("zh-min-nan", "Foo_nan", "x")
            ),
        }
    )
    summary = article_linker.link_qid("Q1", wikidata_client=wd, wikipedia_client=wiki)
    assert {article.language for article in summary.articles} == {"simple", "zh-min-nan"}


def test_best_language_picks_preferred_then_falls_back() -> None:
    InMemoryWikidataClient({})
    InMemoryWikipediaClient({})
    s = article_linker.LinkSummary(qid="Q1", entity=None)
    s.articles = [
        _sample_article("de", "X"),
        _sample_article("fr", "X"),
    ]
    assert s.best_language() == "fr"
    s.articles = [_sample_article("de", "X"), _sample_article("es", "X")]
    assert s.best_language() == "de"  # de is preferred, then fr
    s.articles = []
    assert s.best_language() == ""


def test_dedup_repeated_qids_in_fetch_qids() -> None:
    wd = InMemoryWikidataClient({"Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "X"})})
    en = InMemoryWikipediaClient(
        {("enwiki", "X"): FetchResult("ok", _sample_article("en", "X", "x"))}
    )
    summaries = article_linker.fetch_qids(
        ["Q1", "Q1", "Q1"], wikidata_client=wd, wikipedia_client=en
    )
    # Deduplication happens at the caller; here we just verify that
    # repeated QIDs are not silently collapsed.
    assert len(summaries) == 3


def test_fetch_qids_uses_batch_clients_and_preserves_input_order() -> None:
    class BatchWd(InMemoryWikidataClient):
        calls = 0

        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            self.calls += 1
            return [self.get_entity(qid) for qid in qids]

    class BatchWiki(InMemoryWikipediaClient):
        calls = 0

        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            self.calls += 1
            return {
                title: self.fetch_article(language, site, title, fetch_full_text=fetch_full_text)
                for title in titles
            }

    entity = WikidataEntity(qid="Q1", sitelinks={"enwiki": "X", "frwiki": "X_fr"})
    wd = BatchWd({"Q1": entity})
    wiki = BatchWiki(
        {
            ("enwiki", "X"): FetchResult("ok", _sample_article("en", "X", "en")),
            ("frwiki", "X_fr"): FetchResult("ok", _sample_article("fr", "X_fr", "fr")),
        }
    )
    summaries = article_linker.fetch_qids(
        ["Q1", "Q1"],
        wikidata_client=wd,
        wikipedia_client=wiki,
    )
    assert [summary.qid for summary in summaries] == ["Q1", "Q1"]
    assert [[article.language for article in summary.articles] for summary in summaries] == [
        ["en", "fr"],
        ["en", "fr"],
    ]
    assert wd.calls == 1
    assert wiki.calls == 2


def test_fetch_qids_reports_batched_qid_site_and_article_progress() -> None:
    class BatchWd(InMemoryWikidataClient):
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [self.get_entity(qid) for qid in qids]

    class BatchWiki(InMemoryWikipediaClient):
        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            return {
                title: self.fetch_article(language, site, title, fetch_full_text=fetch_full_text)
                for title in titles
            }

    wd = BatchWd(
        {
            "Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "One", "frwiki": "Un"}),
            "Q2": WikidataEntity(qid="Q2", sitelinks={"enwiki": "Two"}),
        }
    )
    wiki = BatchWiki(
        {
            ("enwiki", "One"): FetchResult("ok", _sample_article("en", "One", "one")),
            ("enwiki", "Two"): FetchResult("ok", _sample_article("en", "Two", "two")),
            ("frwiki", "Un"): FetchResult("ok", _sample_article("fr", "Un", "un")),
        }
    )
    progress = EnrichmentProgress(total_qids=0)

    summaries = article_linker.fetch_qids(
        ["Q1", "Q2"],
        wikidata_client=wd,
        wikipedia_client=wiki,
        batch_size=1,
        progress=progress,
    )

    assert [summary.qid for summary in summaries] == ["Q1", "Q2"]
    snapshot = progress.snapshot()
    assert (snapshot.qids_completed, snapshot.qids_total) == (2, 2)
    assert (snapshot.sites_completed, snapshot.sites_total) == (2, 2)
    assert snapshot.articles_attempted == 3
    assert snapshot.phase == "wikipedia"


def test_fetch_qids_reports_qid_progress_for_compatibility_clients() -> None:
    wd = InMemoryWikidataClient(
        {
            "Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "One"}),
            "Q2": WikidataEntity(qid="Q2", sitelinks={"enwiki": "Two"}),
        }
    )
    wiki = InMemoryWikipediaClient(
        {
            ("enwiki", "One"): FetchResult("ok", _sample_article("en", "One", "one")),
            ("enwiki", "Two"): FetchResult("ok", _sample_article("en", "Two", "two")),
        }
    )
    progress = EnrichmentProgress(total_qids=0)

    summaries = article_linker.fetch_qids(
        ["Q1", "Q2"],
        wikidata_client=wd,
        wikipedia_client=wiki,
        progress=progress,
    )

    assert [summary.qid for summary in summaries] == ["Q1", "Q2"]
    snapshot = progress.snapshot()
    assert (snapshot.qids_completed, snapshot.qids_total) == (2, 2)
    assert (snapshot.sites_completed, snapshot.sites_total) == (0, 0)
    assert snapshot.articles_attempted == 0
    assert snapshot.phase == "wikidata"


def test_batch_client_capability_protocols_are_structural() -> None:
    class WikidataBatch:
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [None for _ in qids]

    class WikipediaBatch:
        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            return {}

    assert isinstance(WikidataBatch(), BatchWikidataClient)
    assert isinstance(WikipediaBatch(), BatchWikipediaClient)


def test_fetch_qids_chunks_same_site_title_batches_at_requested_limit() -> None:
    class BatchWd(InMemoryWikidataClient):
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [self.get_entity(qid) for qid in qids]

    class BatchWiki(InMemoryWikipediaClient):
        def __init__(self) -> None:
            super().__init__({})
            self.batch_sizes: list[int] = []

        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            self.batch_sizes.append(len(titles))
            return {
                title: FetchResult("ok", _sample_article("en", title, "text")) for title in titles
            }

    qids = [f"Q{index}" for index in range(1, 52)]
    wd = BatchWd({qid: WikidataEntity(qid=qid, sitelinks={"enwiki": qid}) for qid in qids})
    wiki = BatchWiki()
    summaries = article_linker.fetch_qids(
        qids,
        wikidata_client=wd,
        wikipedia_client=wiki,
        batch_size=50,
    )
    assert len(summaries) == 51
    assert wiki.batch_sizes == [50, 1]


def test_fetch_qids_prioritizes_other_sites_before_more_large_site_chunks() -> None:
    """One large site must not monopolize the executor's FIFO queue."""

    class BatchWd(InMemoryWikidataClient):
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [self.get_entity(qid) for qid in qids]

    first_chunk_started = threading.Event()
    other_site_started = threading.Event()
    release_large_site = threading.Event()

    class GatedBatchWiki(InMemoryWikipediaClient):
        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            if titles == ["Q1", "Q2"]:
                first_chunk_started.set()
                release_large_site.wait(timeout=5)
            elif titles == ["Q3", "Q4"]:
                release_large_site.wait(timeout=5)
            elif site == "frwiki":
                other_site_started.set()
            return {
                title: FetchResult("ok", _sample_article(language, title, title))
                for title in titles
            }

    qids = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    wd = BatchWd(
        {
            **{qid: WikidataEntity(qid=qid, sitelinks={"enwiki": qid}) for qid in qids[:4]},
            "Q5": WikidataEntity(qid="Q5", sitelinks={"frwiki": "Q5"}),
        }
    )
    progress = EnrichmentProgress(total_qids=0)
    result: list[article_linker.LinkSummary] = []

    fetcher = threading.Thread(
        target=lambda: result.extend(
            article_linker.fetch_qids(
                qids,
                wikidata_client=wd,
                wikipedia_client=GatedBatchWiki({}),
                batch_size=2,
                site_workers=2,
                progress=progress,
            )
        )
    )
    fetcher.start()

    assert first_chunk_started.wait(timeout=2)
    try:
        assert other_site_started.wait(timeout=2), (
            "another site must run before the large site's second chunk"
        )
    finally:
        release_large_site.set()
        fetcher.join(timeout=5)

    assert not fetcher.is_alive()
    assert [summary.qid for summary in result] == qids
    assert [[article.title for article in summary.articles] for summary in result] == [
        ["Q1"],
        ["Q2"],
        ["Q3"],
        ["Q4"],
        ["Q5"],
    ]
    snapshot = progress.snapshot()
    assert snapshot.sites_completed == snapshot.sites_total == 2
    assert snapshot.articles_attempted == 5


def test_fetch_qids_reports_completed_chunks_before_site_finishes() -> None:
    class BatchWd(InMemoryWikidataClient):
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [self.get_entity(qid) for qid in qids]

    blocked_chunk_started = threading.Event()
    other_chunk_returned = threading.Event()
    release_blocked_chunk = threading.Event()

    class GatedBatchWiki(InMemoryWikipediaClient):
        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            if titles == ["Q1", "Q2"]:
                blocked_chunk_started.set()
                release_blocked_chunk.wait(timeout=5)
            else:
                other_chunk_returned.set()
            return {
                title: FetchResult("ok", _sample_article(language, title, title))
                for title in titles
            }

    qids = ["Q1", "Q2", "Q3", "Q4"]
    wd = BatchWd({qid: WikidataEntity(qid=qid, sitelinks={"enwiki": qid}) for qid in qids})
    progress = EnrichmentProgress(total_qids=0)
    fetcher = threading.Thread(
        target=article_linker.fetch_qids,
        kwargs={
            "qids": qids,
            "wikidata_client": wd,
            "wikipedia_client": GatedBatchWiki({}),
            "batch_size": 2,
            "site_workers": 2,
            "progress": progress,
        },
    )
    fetcher.start()

    assert blocked_chunk_started.wait(timeout=2)
    assert other_chunk_returned.wait(timeout=2)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and progress.snapshot().articles_attempted == 0:
        time.sleep(0.01)
    try:
        snapshot = progress.snapshot()
        assert snapshot.articles_attempted == 2
        assert snapshot.sites_completed == 0
    finally:
        release_blocked_chunk.set()
        fetcher.join(timeout=5)

    assert not fetcher.is_alive()
    final = progress.snapshot()
    assert final.articles_attempted == 4
    assert final.sites_completed == final.sites_total == 1


def test_fetch_qids_reports_site_progress_as_sites_complete_not_in_input_order() -> None:
    """Regression: Wikipedia progress must update as fast sites finish.

    ``executor.map`` yields results in input order, so the previous
    implementation only advanced ``complete_site`` once the FIRST
    (language, site) finished all its HTTP calls. For regions like
    Antarctica where ``enwiki`` alone has hundreds of titles, the
    heartbeat would stay at ``Wikipedia 0/108 sites`` until that one
    site finished, even though all the small-language sites had
    already completed in the background.
    """

    class BatchWd(InMemoryWikidataClient):
        def get_entities(self, qids: list[str]) -> list[WikidataEntity | None]:
            return [self.get_entity(qid) for qid in qids]

    slow_release = threading.Event()

    class GatedBatchWiki(InMemoryWikipediaClient):
        def __init__(self) -> None:
            super().__init__({})
            self.completed_sites: list[str] = []
            self._lock = threading.Lock()

        def fetch_articles(
            self, language: str, site: str, titles: list[str], *, fetch_full_text: bool = True
        ) -> dict[str, FetchResult]:
            # The slow site (enwiki) blocks until the test releases it;
            # every other site returns immediately so it can finish
            # while enwiki is still waiting.
            if site == "enwiki":
                slow_release.wait(timeout=10)
            with self._lock:
                self.completed_sites.append(site)
            return {
                title: FetchResult("ok", _sample_article(language, title, "body"))
                for title in titles
            }

    # Many QIDs all sharing enwiki (the slow site) plus a few QIDs
    # with small sites that should finish quickly.
    slow_qids = [f"Q{index}" for index in range(1, 31)]
    fast_qids = [f"Q{index + 100}" for index in range(1, 6)]
    wd = BatchWd(
        {
            **{qid: WikidataEntity(qid=qid, sitelinks={"enwiki": qid}) for qid in slow_qids},
            **{qid: WikidataEntity(qid=qid, sitelinks={"frwiki": qid}) for qid in fast_qids},
        }
    )
    wiki = GatedBatchWiki()
    progress = EnrichmentProgress(total_qids=0)

    fetcher = threading.Thread(
        target=article_linker.fetch_qids,
        kwargs={
            "qids": slow_qids + fast_qids,
            "wikidata_client": wd,
            "wikipedia_client": wiki,
            "site_workers": 4,
            "progress": progress,
        },
    )
    fetcher.start()

    # Wait for the small sites (frwiki) to finish while enwiki is
    # still gated. Once frwiki is in completed_sites, give the
    # main thread a moment to advance the progress snapshot.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with wiki._lock:
            if "frwiki" in wiki.completed_sites:
                break
        time.sleep(0.02)
    else:  # pragma: no cover - guard against real regressions
        slow_release.set()
        fetcher.join(timeout=5)
        pytest.fail("frwiki never finished while enwiki was gated")

    # While enwiki is still gated, the progress must already reflect
    # the small site(s) that finished. This is exactly the heartbeat
    # scenario: the heartbeat should report "Wikipedia N/M sites" with
    # N>0 long before enwiki finishes.
    snapshot = progress.snapshot()
    assert snapshot.sites_total == 2
    assert snapshot.sites_completed == 1, (
        f"frwiki must advance progress while enwiki is still gated; snapshot={snapshot}"
    )

    # Release enwiki and let the fetch finish cleanly.
    slow_release.set()
    fetcher.join(timeout=5)
    assert not fetcher.is_alive()

    final = progress.snapshot()
    assert final.sites_completed == final.sites_total == 2
    assert final.articles_attempted == 35
