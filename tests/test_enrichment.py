"""Tests for the enrichment stack: QID validation, parsing, clients, linker, text cleaning."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.enrichment import article_linker
from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
    normalize_unicode,
    normalize_whitespace,
    strip_template_markers,
)
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    CachedWikidataClient,
    InMemoryWikidataClient,
    WikidataEntity,
    is_valid_qid,
    language_from_site,
    parse_wikidata_entity,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    CachedWikipediaClient,
    FetchResult,
    InMemoryWikipediaClient,
    WikipediaArticle,
    parse_wikipedia_response,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache

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
    assert entity.sitelinks == {"enwiki": "Douglas Adams", "frwiki": "Douglas Adams"}
    assert entity.labels == {"en": "Douglas Adams"}


def test_parse_wikidata_entity_returns_none_for_missing() -> None:
    assert parse_wikidata_entity("Q42", {"entities": {"Q42": {"missing": "x"}}}) is None
    assert parse_wikidata_entity("Q42", {"entities": {}}) is None


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
    assert res.article is None


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
