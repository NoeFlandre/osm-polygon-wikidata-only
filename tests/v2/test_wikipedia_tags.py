import pytest

from osm_polygon_wikidata_only.v2.wikipedia_tags import (
    WikipediaTagRef,
    WikipediaTagRejection,
    _append_tag_value,
    _consume_tag_values,
    _from_url,
    _normalize_title_reference,
    _normalize_url_reference,
    _normalize_value,
    _tag_language,
    _title_reference_parts,
    _wikipedia_host,
    parse_wikipedia_tags,
)


def test_base_key_accepts_all_language_codes() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia": "ja:東京;zh-yue:香港;sr-Latn:Beograd",
        }
    )
    assert rejected == ()
    assert [(ref.language, ref.title) for ref in refs] == [
        ("ja", "東京"),
        ("sr-latn", "Beograd"),
        ("zh-yue", "香港"),
    ]


def test_language_normalization_handles_underscore_and_case() -> None:
    refs, rejected = parse_wikipedia_tags({"wikipedia": "SR_Latn:Belgrade"})

    assert rejected == ()
    assert [(ref.language, ref.title) for ref in refs] == [("sr-latn", "Belgrade")]


def test_language_specific_keys_do_not_need_a_fixed_allow_list() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia:de": "Berlin",
            "wikipedia:fr": "Paris",
            "wikipedia:zh-yue": "香港",
        }
    )
    assert rejected == ()
    assert [(ref.language, ref.title) for ref in refs] == [
        ("de", "Berlin"),
        ("fr", "Paris"),
        ("zh-yue", "香港"),
    ]


def test_urls_and_title_normalization_are_deterministic() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia": "https://en.wikipedia.org/wiki/New_York_City;en:New%20York%20City",
        }
    )
    assert rejected == ()
    assert refs == (
        WikipediaTagRef(
            language="en",
            title="New York City",
            raw_key="wikipedia",
            raw_value="https://en.wikipedia.org/wiki/New_York_City",
        ),
    )


def test_duplicates_and_irrelevant_tags_are_removed() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "name": "Paris",
            "wikipedia": "en:Paris",
            "wikipedia:en": "Paris",
        }
    )
    assert rejected == ()
    assert len(refs) == 1
    assert refs[0].language == "en"
    assert refs[0].title == "Paris"


def test_malformed_references_are_reported_without_raising() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia": "Title without language;bad::value",
            "wikipedia:not a language": "Title",
        }
    )
    assert refs == ()
    assert all(isinstance(item, WikipediaTagRejection) for item in rejected)
    assert len(rejected) == 3
    assert [item.reason for item in rejected] == [
        "missing language prefix",
        "empty language or title",
        "invalid language code",
    ]


def test_empty_values_and_whitespace_are_rejected() -> None:
    refs, rejected = parse_wikipedia_tags({"wikipedia": " ; en: "})
    assert refs == ()
    assert [item.reason for item in rejected] == ["empty value", "empty language or title"]


def test_urls_must_point_to_a_matching_wikipedia_language_and_page() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia": (
                "https://en.wikipedia.org/wiki/Paris;"
                "https://en.wikipedia.org/wiki/Paris#History;"
                "https://fr.wikipedia.org/wiki/Paris;"
                "https://en.m.wikipedia.org/wiki/London;"
                "https://en.wikipedia.org/not-a-wiki-page"
            ),
            "wikipedia:de": "https://fr.wikipedia.org/wiki/de",
        }
    )

    assert [(ref.language, ref.title) for ref in refs] == [
        ("en", "London"),
        ("en", "Paris"),
        ("fr", "Paris"),
    ]
    assert [item.reason for item in rejected] == [
        "invalid language code",
        "language disagrees with URL",
    ]


def test_normalization_returns_an_empty_reason_for_valid_references() -> None:
    assert _normalize_value("en", "New_York") == (("en", "New York"), "")
    assert _normalize_value(None, "en:New%20York") == (("en", "New York"), "")
    assert _normalize_value("en", "https://en.wikipedia.org/wiki/New_York") == (
        ("en", "New York"),
        "",
    )


def test_wikipedia_tag_helpers_cover_host_and_url_validation() -> None:
    assert _wikipedia_host(None) is None
    assert _wikipedia_host("EN.WIKIPEDIA.ORG") == "en.wikipedia.org"
    assert _wikipedia_host("example.org") is None
    assert _from_url("not a URL") is None
    assert _normalize_url_reference(None, ("en", "Paris")) == (("en", "Paris"), "")
    assert _normalize_url_reference("fr", ("en", "Paris")) == (
        None,
        "language disagrees with URL",
    )


@pytest.mark.parametrize(
    ("key_language", "value", "expected"),
    [
        ("en", "London", (("en", "London"), "")),
        (None, "en:London", (("en", "London"), "")),
        (None, "London", (None, "missing language prefix")),
        (None, "not-a-language:London", (None, "invalid language code")),
    ],
)
def test_title_reference_helpers_cover_language_forms(
    key_language: str | None,
    value: str,
    expected: tuple[tuple[str, str] | None, str],
) -> None:
    assert _title_reference_parts(key_language, value) == expected
    assert _normalize_title_reference(key_language, value) == expected


def test_tag_value_helpers_share_the_parser_contract() -> None:
    refs: list[WikipediaTagRef] = []
    rejected: list[WikipediaTagRejection] = []
    seen: set[tuple[str, str]] = set()
    _append_tag_value("wikipedia", "en:Paris;", "en:Paris", None, seen, refs, rejected)
    _append_tag_value("wikipedia", "en:Paris;", "", None, seen, refs, rejected)
    _append_tag_value("wikipedia", "en:Paris;", "bad", None, seen, refs, rejected)
    _append_tag_value("wikipedia", "en:Paris;", "en:Paris", None, seen, refs, rejected)
    _consume_tag_values("wikipedia", "en:Paris;;fr:Paris", None, seen, refs, rejected)

    assert [(ref.language, ref.title) for ref in refs] == [("en", "Paris"), ("fr", "Paris")]
    assert [item.reason for item in rejected] == [
        "empty value",
        "missing language prefix",
        "empty value",
    ]


def test_tag_language_helper_distinguishes_base_valid_and_invalid_keys() -> None:
    assert _tag_language("wikipedia") == (None, None)
    assert _tag_language("wikipedia:en") == ("en", None)
    assert _tag_language("wikipedia:not a language") == (None, "invalid language code")


def test_non_wikipedia_urls_and_empty_titles_are_rejected() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia": "https://example.org/wiki/Paris;en: :invalid",
        }
    )

    assert refs == ()
    assert [item.reason for item in rejected] == [
        "empty language or title",
        "invalid language code",
    ]


def test_invalid_language_key_does_not_hide_later_valid_keys() -> None:
    refs, rejected = parse_wikipedia_tags(
        {
            "wikipedia:not-a-language": "ignored",
            "wikipedia:en": "London",
        }
    )

    assert [(ref.language, ref.title) for ref in refs] == [("en", "London")]
    assert rejected == (
        WikipediaTagRejection(
            raw_key="wikipedia:not-a-language",
            raw_value="ignored",
            reason="invalid language code",
        ),
    )
