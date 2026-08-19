"""Focused branch and contract tests for pure enrichment helpers."""

from __future__ import annotations

import inspect

import pytest

from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    html_to_plain_text,
    normalize_unicode,
    strip_template_markers,
)
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    _is_language_wiki,
    language_from_site,
    parse_wikidata_entity,
    qids_from_osm_tag,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.parsing import (
    _parse_wikipedia_page,
    parse_wikipedia_batch_response,
    parse_wikipedia_response,
    plain_text_from_parse_response,
    query_with_extract,
    revision_id_from_query,
)


def _wiki_page(
    title: str = "Alpha",
    *,
    page_id: int = 1,
    extract: str = "Lead.\n\nBody.",
) -> dict[str, object]:
    return {
        "pageid": page_id,
        "title": title,
        "extract": extract,
        "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}],
    }


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"query": {"pages": []}},
        {"query": {"pages": {}}},
        {"query": {"pages": {"1": "not a page"}}},
        {"query": {"pages": {"1": {"revisions": []}}}},
        {"query": {"pages": {"1": {"revisions": ["not a revision"]}}}},
    ],
)
def test_revision_id_from_query_returns_zero_for_malformed_shapes(
    data: dict[str, object],
) -> None:
    assert revision_id_from_query(data) == 0


def test_revision_id_from_query_reads_first_revision() -> None:
    data = {"query": {"pages": {"1": {"revisions": [{"revid": 73}]}}}}

    assert revision_id_from_query(data) == 73


def test_revision_id_from_query_defaults_missing_revision_id_to_zero() -> None:
    data = {"query": {"pages": {"1": {"revisions": [{}]}}}}

    assert revision_id_from_query(data) == 0


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({}, ""),
        ({"parse": "not an object"}, ""),
        ({"parse": {}}, ""),
        ({"parse": {"text": 123}}, ""),
        ({"parse": {"text": "<p>Hello&nbsp;world</p>"}}, "Hello world"),
        ({"parse": {"text": {}}}, ""),
        ({"parse": {"text": {"*": "<p>Hello</p>"}}}, "Hello"),
    ],
)
def test_plain_text_from_parse_response_handles_action_api_shapes(
    data: dict[str, object], expected: str
) -> None:
    assert plain_text_from_parse_response(data) == expected


def test_query_with_extract_returns_a_copy_without_mutating_input() -> None:
    original = {"query": {"pages": {"1": {"pageid": 1, "extract": "old"}}}}

    result = query_with_extract(original, "new")

    assert result == {"query": {"pages": {"1": {"pageid": 1, "extract": "new"}}}}
    assert original["query"]["pages"]["1"]["extract"] == "old"  # type: ignore[index]


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"query": {"pages": {}}},
        {"query": {"pages": {"1": "not a page"}}},
    ],
)
def test_query_with_extract_preserves_malformed_response(data: dict[str, object]) -> None:
    assert query_with_extract(data, "new") is data


def test_parse_wikipedia_batch_response_resolves_aliases_and_missing_titles() -> None:
    data = {
        "query": {
            "normalized": [{"from": "alpha", "to": "Alpha"}],
            "redirects": [{"from": "Old Alpha", "to": "Alpha"}],
            "pages": {"1": _wiki_page()},
        }
    }

    results = parse_wikipedia_batch_response(
        "en", "enwiki", ["alpha", "Old Alpha", "Missing"], data, fetch_full_text=False
    )

    assert results["alpha"].status == "ok"
    assert results["Old Alpha"].status == "ok"
    assert results["Missing"].status == "article_not_found"
    assert results["Missing"].error == "page missing"
    assert results["alpha"].article is not None
    assert results["alpha"].article.language == "en"
    assert results["alpha"].article.site == "enwiki"


def test_parse_wikipedia_batch_response_ignores_bad_aliases_and_breaks_cycles() -> None:
    data = {
        "query": {
            "normalized": ["bad", {"from": "A", "to": "B"}],
            "redirects": [
                {"from": "B", "to": "A"},
                {"from": 1, "to": "B"},
                {"from": "C", "to": "D"},
            ],
            "pages": {"2": "not a page", "1": _wiki_page("A")},
        }
    }

    result = parse_wikipedia_batch_response("en", "enwiki", ["A"], data, fetch_full_text=True)

    assert result["A"].status == "ok"


def test_parse_wikipedia_batch_response_uses_requested_title_when_page_title_is_missing() -> None:
    data = {
        "query": {
            "pages": {
                "Requested": {
                    "pageid": 1,
                    "extract": "Text",
                    "revisions": [{"revid": 1}],
                }
            }
        }
    }

    result = parse_wikipedia_batch_response(
        "en", "enwiki", ["Requested"], data, fetch_full_text=True
    )

    assert result["Requested"].article is not None
    assert result["Requested"].article.title == "Requested"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "missing query in batch response"),
        ({"query": {"pages": []}}, "missing query.pages in batch response"),
    ],
)
def test_parse_wikipedia_batch_response_rejects_missing_containers(
    data: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError) as caught:
        parse_wikipedia_batch_response("en", "enwiki", [], data, fetch_full_text=False)
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("data", "status", "error"),
    [
        ({"query": 1}, "parse_error", "missing query.pages"),
        ({"query": {"pages": {}}}, "article_not_found", "no pages in response"),
        (
            {"query": {"pages": {"1": {"missing": "", "title": "A"}}}},
            "article_not_found",
            "page missing",
        ),
        (
            {
                "query": {
                    "pages": {
                        "1": {
                            **_wiki_page(),
                            "missing": "still marked missing",
                        }
                    }
                }
            },
            "article_not_found",
            "page missing",
        ),
        (
            {"query": {"pages": {"1": {"pageid": 1, "revisions": []}}}},
            "parse_error",
            "no revisions",
        ),
    ],
)
def test_parse_wikipedia_response_reports_terminal_shape_errors(
    data: dict[str, object], status: str, error: str
) -> None:
    result = parse_wikipedia_response("en", "enwiki", "A", data)

    assert result.status == status
    assert result.error == error


def test_parse_wikipedia_response_builds_encoded_fallback_url() -> None:
    result = parse_wikipedia_response(
        "fr",
        "frwiki",
        "Fallback",
        {
            "query": {
                "pages": {
                    "1": {
                        **_wiki_page("A title/with slash", extract="Text"),
                        "fullurl": "",
                    }
                }
            }
        },
    )

    assert result.article is not None
    assert result.article.url == "https://fr.wikipedia.org/wiki/A_title/with_slash"


def test_parse_wikipedia_response_preserves_the_article_contract() -> None:
    result = parse_wikipedia_response(
        "en",
        "enwiki",
        "Requested title",
        {
            "query": {
                "pages": {
                    "7": {
                        **_wiki_page("Canonical title", page_id=7, extract="Lead.\n\nBody."),
                        "fullurl": "https://example.test/canonical",
                        "thumbnail": {
                            "source": "https://example.test/thumb.jpg",
                            "width": 320,
                            "height": 240,
                        },
                    }
                }
            }
        },
    )

    assert result.status == "ok"
    assert result.error == ""
    assert result.article is not None
    article = result.article
    assert article.language == "en"
    assert article.site == "enwiki"
    assert article.title == "Canonical title"
    assert article.page_id == 7
    assert article.revision_id == 42
    assert article.revision_timestamp == "2026-01-01T00:00:00Z"
    assert article.url == "https://example.test/canonical"
    assert article.lead_text == "Lead."
    assert article.extract == "Lead. Body."
    assert article.full_text == "Lead. Body."
    assert article.full_text_format == "plain_text"
    assert article.thumbnail_url == "https://example.test/thumb.jpg"
    assert article.thumbnail_width == 320
    assert article.thumbnail_height == 240
    assert article.categories == []
    assert article.license == "CC BY-SA 4.0"
    assert "Canonical title" in article.attribution
    assert "revision 42" in article.attribution
    assert article.source_api == "mediawiki_action_api"
    assert article.retrieved_at


def test_parse_wikipedia_response_reports_empty_text_reason() -> None:
    result = parse_wikipedia_response(
        "en",
        "enwiki",
        "A",
        {"query": {"pages": {"1": {"pageid": 1, "revisions": [{"revid": 1}], "extract": ""}}}},
    )

    assert result.status == "empty_text"
    assert result.error == "no extract returned by API"
    assert result.article is not None
    assert result.article.lead_text == ""


def test_parse_wikipedia_response_defaults_missing_optional_fields() -> None:
    result = parse_wikipedia_response(
        "en",
        "enwiki",
        "A",
        {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "title": "A",
                        "revisions": [{}],
                        "extract": "Lead",
                        "thumbnail": {},
                    }
                }
            }
        },
    )

    assert result.article is not None
    assert result.article.revision_id == 0
    assert result.article.revision_timestamp == ""
    assert result.article.thumbnail_url == ""


def test_parse_wikipedia_response_uses_the_first_lead_paragraph_and_limit() -> None:
    first_paragraph = "A" * 501
    extract = f"{first_paragraph}\n\nSecond paragraph\n\nThird paragraph"
    result = parse_wikipedia_response(
        "en", "enwiki", "A", {"query": {"pages": {"1": _wiki_page(extract=extract)}}}
    )

    assert result.article is not None
    assert result.article.lead_text == "A" * 500


def test_parse_wikipedia_response_keeps_all_lines_in_the_first_lead_paragraph() -> None:
    result = parse_wikipedia_response(
        "en",
        "enwiki",
        "A",
        {"query": {"pages": {"1": _wiki_page(extract="First line\nsecond line\n\nBody")}}},
    )

    assert result.article is not None
    assert result.article.lead_text == "First line second line"


def test_parse_wikipedia_response_signature_keeps_compatibility_defaults() -> None:
    parameters = inspect.signature(parse_wikipedia_response).parameters

    assert parameters["wikidata_label"].default == ""
    assert parameters["wikidata_description"].default == ""
    assert parameters["fetch_full_text"].default is True


@pytest.mark.parametrize(
    ("page", "status", "error"),
    [
        ({"missing": "", "pageid": 1}, "article_not_found", "page missing"),
        ({"pageid": 1, "revisions": []}, "parse_error", "no revisions"),
    ],
)
def test_selected_page_parser_reports_shape_errors(
    page: dict[str, object], status: str, error: str
) -> None:
    result = _parse_wikipedia_page("en", "enwiki", "Requested", page)

    assert result.status == status
    assert result.error == error


def test_selected_page_parser_preserves_all_article_fields() -> None:
    result = _parse_wikipedia_page(
        "en",
        "enwiki",
        "Requested",
        {
            "pageid": 7,
            "title": "Canonical title",
            "fullurl": "https://example.test/canonical",
            "extract": "First line\nsecond line\n\nBody\n\nThird",
            "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}],
            "thumbnail": {
                "source": "https://example.test/thumb.jpg",
                "width": 320,
                "height": 240,
            },
        },
    )

    assert result.status == "ok"
    assert result.error == ""
    assert result.article is not None
    assert result.article.language == "en"
    assert result.article.site == "enwiki"
    assert result.article.title == "Canonical title"
    assert result.article.page_id == 7
    assert result.article.revision_id == 42
    assert result.article.revision_timestamp == "2026-01-01T00:00:00Z"
    assert result.article.url == "https://example.test/canonical"
    assert result.article.lead_text == "First line second line"
    assert result.article.extract == "First line second line Body Third"
    assert result.article.full_text == "First line second line Body Third"
    assert result.article.full_text_format == "plain_text"
    assert result.article.thumbnail_url == "https://example.test/thumb.jpg"
    assert result.article.thumbnail_width == 320
    assert result.article.thumbnail_height == 240
    assert result.article.categories == []
    assert result.article.license == "CC BY-SA 4.0"
    assert "Canonical title" in result.article.attribution
    assert "revision 42" in result.article.attribution
    assert result.article.source_api == "mediawiki_action_api"
    assert result.article.retrieved_at


def test_selected_page_parser_defaults_missing_optional_fields() -> None:
    result = _parse_wikipedia_page(
        "en",
        "enwiki",
        "Requested",
        {"pageid": 1, "revisions": [{}], "extract": "Text", "thumbnail": {}},
    )

    assert result.article is not None
    assert result.article.title == "Requested"
    assert result.article.revision_id == 0
    assert result.article.revision_timestamp == ""
    assert result.article.thumbnail_url == ""


def test_selected_page_parser_handles_empty_extract_and_missing_url() -> None:
    result = _parse_wikipedia_page(
        "fr",
        "frwiki",
        "Requested",
        {
            "pageid": 1,
            "title": "A title/with slash",
            "fullurl": "",
            "revisions": [{"revid": 1}],
            "extract": "",
        },
    )

    assert result.status == "empty_text"
    assert result.error == "no extract returned by API"
    assert result.article is not None
    assert result.article.lead_text == ""
    assert result.article.extract == ""
    assert result.article.full_text == ""
    assert result.article.url == "https://fr.wikipedia.org/wiki/A_title/with_slash"


def test_selected_page_parser_enforces_the_lead_length_limit() -> None:
    result = _parse_wikipedia_page(
        "en",
        "enwiki",
        "Requested",
        {
            "pageid": 1,
            "title": "A",
            "revisions": [{"revid": 1}],
            "extract": f"{'A' * 501}\n\nBody",
        },
    )

    assert result.article is not None
    assert result.article.lead_text == "A" * 500


def test_qids_from_osm_tag_trims_and_deduplicates_in_source_order() -> None:
    assert qids_from_osm_tag(" Q2 ; Q1 ; Q2 ") == ("Q2", "Q1")


@pytest.mark.parametrize("value", ["", "Q1;", "Q1;not-a-qid", "Q0", ";"])
def test_qids_from_osm_tag_rejects_any_invalid_component(value: str) -> None:
    assert qids_from_osm_tag(value) == ()


@pytest.mark.parametrize(
    ("site", "expected"),
    [
        ("enwiki", True),
        ("be_x_oldwiki", True),
        ("x-_wiki", True),
        ("zh-min-wiki", True),
        ("wiki", False),
        ("commonswiki", False),
        ("ENwiki", False),
        ("en.wiki", False),
        ("en wiki", False),
        ("wikifunctionswiki", False),
    ],
)
def test_language_wiki_filter_is_strict(site: str, expected: bool) -> None:
    assert _is_language_wiki(site) is expected


def test_parse_wikidata_entity_keeps_values_and_ignores_empty_fields() -> None:
    entity = parse_wikidata_entity(
        "Q42",
        {
            "entities": {
                "Q42": {
                    "sitelinks": {
                        "enwiki": {"title": "Douglas Adams"},
                        "frwiki": {},
                        "commonswiki": {"title": "ignored"},
                    },
                    "labels": {"en": {"value": "Douglas Adams"}, "fr": {}},
                    "descriptions": {"en": {"value": "Writer"}, "fr": {}},
                    "aliases": {"en": [{"value": "Douglas"}, {"value": ""}, {}]},
                }
            }
        },
    )

    assert entity == WikidataEntity(
        qid="Q42",
        sitelinks={"enwiki": "Douglas Adams"},
        labels={"en": "Douglas Adams", "fr": ""},
        descriptions={"en": "Writer", "fr": ""},
        aliases={"en": ["Douglas"]},
    )


def test_language_from_site_handles_legacy_and_nonwiki_keys() -> None:
    assert language_from_site("be_x_oldwiki") == "be-tarask"
    assert language_from_site("zh_min_nanwiki") == "zh-min-nan"
    assert language_from_site("en") == "en"


def test_cleaning_handles_empty_unicode_and_non_template_text() -> None:
    assert clean_article_text("") == ""
    assert normalize_unicode("e\u0301") == "é"
    assert strip_template_markers("plain text") == "plain text"


def test_html_to_plain_text_ignores_script_style_and_adds_boundaries() -> None:
    html = "<script>hidden()</script><style>.x{display:none}</style><ul><li>One</li><li>Two</li></ul><br>Three"

    assert html_to_plain_text(html) == "One Two Three"
