"""Edge and round-trip tests for MediaWiki response parsing."""

from __future__ import annotations

from typing import Any

import pytest

from osm_polygon_wikidata_only.enrichment.wikipedia.parsing import (
    parse_wikipedia_batch_response,
    parse_wikipedia_response,
    plain_text_from_parse_response,
    query_with_extract,
    revision_id_from_query,
)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"query": {"pages": {"1": {"revisions": [{"revid": 42}]}}}}, 42),
        ({}, 0),
        ({"query": {"pages": []}}, 0),
        ({"query": {"pages": {"1": {"revisions": [{}]}}}}, 0),
    ],
)
def test_revision_id_parser_handles_valid_and_malformed_queries(
    payload: dict[str, Any], expected: int
) -> None:
    assert revision_id_from_query(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"query": {"pages": {"1": "not a page"}}},
        {"query": {"pages": {"1": {"revisions": "not a list"}}}},
        {"query": {"pages": {"1": {"revisions": [None]}}}},
    ],
)
def test_revision_id_parser_rejects_non_mapping_revision_shapes(
    payload: dict[str, Any],
) -> None:
    assert revision_id_from_query(payload) == 0


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"parse": {"text": {"*": "<p>Hello</p>"}}}, "Hello"),
        ({"parse": {"text": "<p>Hello</p>"}}, "Hello"),
        ({"parse": {"text": 42}}, ""),
        ({}, ""),
    ],
)
def test_plain_text_parser_accepts_action_api_text_shapes(
    payload: dict[str, Any], expected: str
) -> None:
    assert plain_text_from_parse_response(payload) == expected


def test_plain_text_parser_rejects_non_mapping_parse_payload() -> None:
    assert plain_text_from_parse_response({"parse": ["not a mapping"]}) == ""


def test_query_with_extract_copies_only_the_first_page() -> None:
    source = {"query": {"pages": {"1": {"title": "One", "extract": "old"}}}}

    result = query_with_extract(source, "new")

    assert result == {"query": {"pages": {"1": {"title": "One", "extract": "new"}}}}
    assert source["query"]["pages"]["1"]["extract"] == "old"


@pytest.mark.parametrize("payload", [{}, {"query": {}}, {"query": {"pages": []}}])
def test_query_with_extract_returns_original_for_missing_pages(payload: dict[str, Any]) -> None:
    assert query_with_extract(payload, "new") is payload


def test_query_with_extract_returns_original_for_non_mapping_page() -> None:
    payload: dict[str, Any] = {"query": {"pages": {"1": "not a mapping"}}}

    assert query_with_extract(payload, "new") is payload


def test_batch_parser_resolves_normalization_and_redirect_aliases() -> None:
    response = {
        "query": {
            "normalized": [{"from": "old title", "to": "New title"}],
            "redirects": [{"from": "New title", "to": "Final title"}],
            "pages": {
                "7": {
                    "title": "Final title",
                    "pageid": 7,
                    "revisions": [{"revid": 9, "timestamp": "2024-01-01T00:00:00Z"}],
                    "extract": "A short article.",
                }
            },
        }
    }

    results = parse_wikipedia_batch_response(
        "en", "enwiki", ["old title", "missing"], response, fetch_full_text=True
    )

    assert results["old title"].status == "ok"
    assert results["old title"].article is not None
    assert results["old title"].article.title == "Final title"
    assert results["missing"].status == "article_not_found"


def test_batch_parser_breaks_alias_cycles_as_a_missing_page() -> None:
    response = {
        "query": {
            "normalized": [{"from": "A", "to": "B"}],
            "redirects": [{"from": "B", "to": "A"}],
            "pages": {},
        }
    }

    result = parse_wikipedia_batch_response("en", "enwiki", ["A"], response, fetch_full_text=False)

    assert result["A"].status == "article_not_found"


@pytest.mark.parametrize(
    "payload, message",
    [({}, "missing query"), ({"query": {"pages": []}}, "missing query.pages")],
)
def test_batch_parser_rejects_missing_response_sections(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_wikipedia_batch_response("en", "enwiki", ["Title"], payload, fetch_full_text=False)


@pytest.mark.parametrize(
    "payload, status, error",
    [
        ({}, "article_not_found", "no pages"),
        ({"query": {"pages": {}}}, "article_not_found", "no pages"),
        ({"query": {"pages": {"1": {"missing": ""}}}}, "article_not_found", "page missing"),
        (
            {"query": {"pages": {"1": {"pageid": 1, "revisions": []}}}},
            "parse_error",
            "no revisions",
        ),
    ],
)
def test_single_response_parser_classifies_incomplete_pages(
    payload: dict[str, Any], status: str, error: str
) -> None:
    result = parse_wikipedia_response("en", "enwiki", "Title", payload)

    assert result.status == status
    assert error in result.error


def test_single_response_parser_builds_fallback_url_and_plain_text() -> None:
    result = parse_wikipedia_response(
        "fr",
        "frwiki",
        "Titre",
        {
            "query": {
                "pages": {
                    "1": {
                        "title": "Titre avec espace",
                        "pageid": 1,
                        "revisions": [{"revid": 2, "timestamp": "2024-01-01T00:00:00Z"}],
                        "extract": "Lead.\n\nBody.",
                    }
                }
            }
        },
    )

    assert result.status == "ok"
    assert result.article is not None
    assert result.article.url.endswith("/Titre_avec_espace")
    assert result.article.lead_text == "Lead."
    assert result.article.full_text == "Lead. Body."


def test_single_response_parser_reports_non_mapping_query() -> None:
    result = parse_wikipedia_response("en", "enwiki", "Title", {"query": "invalid"})

    assert result.status == "parse_error"
    assert result.error == "missing query.pages"


def test_single_response_parser_preserves_article_for_empty_text() -> None:
    result = parse_wikipedia_response(
        "en",
        "enwiki",
        "Empty",
        {
            "query": {
                "pages": {
                    "1": {
                        "title": "Empty",
                        "pageid": 1,
                        "revisions": [{"revid": 2}],
                        "extract": "",
                    }
                }
            }
        },
    )

    assert result.status == "empty_text"
    assert result.article is not None
    assert result.article.title == "Empty"
