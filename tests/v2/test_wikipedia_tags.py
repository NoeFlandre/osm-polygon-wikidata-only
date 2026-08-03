from osm_polygon_wikidata_only.v2.wikipedia_tags import (
    WikipediaTagRef,
    WikipediaTagRejection,
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
