from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import pytest

from osm_polygon_wikidata_only.v2.sentence_logic import (
    DEFAULT_SENTENCE_BATCH_SIZE,
    SAT_SUPPORTED_LANGUAGES,
    SENTENCE_COLUMNS,
    _normalize_pieces,
    is_sat_supported_language,
    sentence_schema,
    split_sections,
)


def _section(section_id: str, language: str, text: str) -> dict[str, object]:
    return {
        "section_id": section_id,
        "document_id": f"document-{section_id}",
        "article_id": f"article-{section_id}",
        "project": "wikipedia",
        "language": language,
        "text": text,
        "content_hash": f"source-{section_id}",
        "section_index": 0,
    }


@dataclass
class _FakeSegmenter:
    pieces: dict[str, list[str]]
    model_id: str = "segment-any-text/sat-3l-sm"
    version: str = "test"

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def split(self, texts: list[str], *, language: str) -> list[list[str]]:
        self.calls.append((language, tuple(texts)))
        return [self.pieces[text] for text in texts]


def test_supported_language_set_is_explicit_and_exact() -> None:
    assert len(SAT_SUPPORTED_LANGUAGES) == 85
    assert is_sat_supported_language("en")
    assert is_sat_supported_language("zh")
    assert not is_sat_supported_language("xx")
    assert not is_sat_supported_language("zh-hans")


def test_sentence_schema_preserves_column_names_and_integer_types() -> None:
    schema = sentence_schema()
    integer_columns = {
        "page_id",
        "revision_id",
        "section_index",
        "level",
        "sentence_index",
        "start_char",
        "end_char",
        "text_length_chars",
        "text_length_words",
        "text_length_tokens_estimate",
    }

    assert schema.names == list(SENTENCE_COLUMNS)
    assert [str(field.type) for field in schema] == [
        "int64" if column in integer_columns else "string" for column in SENTENCE_COLUMNS
    ]


def test_split_sections_routes_only_supported_languages_and_keeps_others_unsplit() -> None:
    sections = [
        _section("fr-1", "fr", "Un. Deux."),
        _section("xx-1", "xx", "One unknown section. Still one row."),
        _section("en-1", "en", "First. Second."),
    ]
    segmenter = _FakeSegmenter(
        {
            "Un. Deux.": ["Un. ", "Deux."],
            "First. Second.": ["First. ", "Second."],
        }
    )

    rows, summary = split_sections(sections, segmenter=segmenter, batch_size=1)

    assert segmenter.calls == [
        ("en", ("First. Second.",)),
        ("fr", ("Un. Deux.",)),
    ]
    assert [row["text"] for row in rows] == [
        "Un. ",
        "Deux.",
        "One unknown section. Still one row.",
        "First. ",
        "Second.",
    ]
    assert [row["segmentation_status"] for row in rows] == [
        "split",
        "split",
        "unsupported_language",
        "split",
        "split",
    ]
    assert rows[2]["sentence_index"] == 0
    assert rows[2]["start_char"] == 0
    assert rows[2]["end_char"] == len("One unknown section. Still one row.")
    assert summary.sections == 3
    assert summary.split_sections == 2
    assert summary.unsplit_sections == 1
    assert summary.sentence_rows == 5
    assert summary.supported_languages == ("en", "fr")
    assert summary.unsupported_languages == ("xx",)

    assert set(rows[0]) == set(SENTENCE_COLUMNS)
    assert rows[0]["section_id"] == "fr-1"
    assert rows[0]["document_id"] == "document-fr-1"
    assert rows[0]["article_id"] == "article-fr-1"
    assert rows[0]["project"] == "wikipedia"
    assert rows[0]["language"] == "fr"
    assert rows[0]["sentence_index"] == 0
    assert rows[0]["start_char"] == 0
    assert rows[0]["end_char"] == 4
    assert rows[0]["text_length_chars"] == 4
    assert rows[0]["text_length_words"] == 1
    assert rows[0]["text_length_tokens_estimate"] == 1
    assert rows[0]["source_content_hash"] == "source-fr-1"
    assert rows[0]["segmenter"] == "sat-3l-sm"
    assert rows[0]["segmenter_version"] == "test"
    assert rows[0]["model_id"] == "segment-any-text/sat-3l-sm"
    assert rows[0]["segmentation_status"] == "split"

    unsupported = rows[2]
    assert set(unsupported) == set(SENTENCE_COLUMNS)
    assert unsupported["section_id"] == "xx-1"
    assert unsupported["language"] == "xx"
    assert unsupported["sentence_index"] == 0
    assert unsupported["start_char"] == 0
    assert unsupported["end_char"] == len("One unknown section. Still one row.")
    assert unsupported["text"] == "One unknown section. Still one row."
    assert unsupported["text_length_chars"] == 35
    assert unsupported["text_length_words"] == 6
    assert unsupported["text_length_tokens_estimate"] == 8
    assert unsupported["content_hash"] == (
        "ad97efaf59930088965f6ddc8b8595e7c5fa805ebddd86231a325f16ff75f700"
    )
    assert unsupported["source_content_hash"] == "source-xx-1"
    assert unsupported["segmenter"] == "unsplit"
    assert unsupported["segmenter_version"] == ""
    assert unsupported["model_id"] == ""
    assert unsupported["segmentation_status"] == "unsupported_language"


def test_sentence_rows_preserve_all_context_and_derived_metadata() -> None:
    section = _section("rich", "en", "Alpha. Beta!")
    section.update(
        {
            "wikidata": "Q42",
            "site": "enwiki",
            "page_id": 7,
            "revision_id": 8,
            "section_index": 2,
            "heading": "Heading",
            "anchor": "heading",
            "level": 1,
            "parent_section_id": "parent",
            "section_path": "[0, 2]",
            "license": "CC BY-SA 4.0",
            "attribution": "Wikipedia",
        }
    )
    segmenter = _FakeSegmenter({"Alpha. Beta!": ["Alpha. ", "Beta!"]})

    rows, _ = split_sections([section], segmenter=segmenter)

    assert rows == [
        {
            "sentence_id": "03c782478ba7eaeea1a443a740d264095d43f87fe7804d1ffa16434832ba96c8",
            "section_id": "rich",
            "document_id": "document-rich",
            "article_id": "article-rich",
            "wikidata": "Q42",
            "project": "wikipedia",
            "language": "en",
            "site": "enwiki",
            "page_id": 7,
            "revision_id": 8,
            "section_index": 2,
            "heading": "Heading",
            "anchor": "heading",
            "level": 1,
            "parent_section_id": "parent",
            "section_path": "[0, 2]",
            "sentence_index": 0,
            "start_char": 0,
            "end_char": 7,
            "text": "Alpha. ",
            "text_length_chars": 7,
            "text_length_words": 1,
            "text_length_tokens_estimate": 1,
            "content_hash": "b0eaa01be4fcb09444635f8bfaa68e4716c0505b906ea6ed9370b73d3e498fbe",
            "source_content_hash": "source-rich",
            "segmenter": "sat-3l-sm",
            "segmenter_version": "test",
            "model_id": "segment-any-text/sat-3l-sm",
            "segmentation_status": "split",
            "license": "CC BY-SA 4.0",
            "attribution": "Wikipedia",
        },
        {
            "sentence_id": "0b309cf991d2370d9f1400e65c0fe74ee3ce4e18cc7c8eb4e79837c1a4de5169",
            "section_id": "rich",
            "document_id": "document-rich",
            "article_id": "article-rich",
            "wikidata": "Q42",
            "project": "wikipedia",
            "language": "en",
            "site": "enwiki",
            "page_id": 7,
            "revision_id": 8,
            "section_index": 2,
            "heading": "Heading",
            "anchor": "heading",
            "level": 1,
            "parent_section_id": "parent",
            "section_path": "[0, 2]",
            "sentence_index": 1,
            "start_char": 7,
            "end_char": 12,
            "text": "Beta!",
            "text_length_chars": 5,
            "text_length_words": 1,
            "text_length_tokens_estimate": 1,
            "content_hash": "c37437ff125d00c6ac682e64ac80cbf3597e0adf44f7d5856f38a3d53727eb03",
            "source_content_hash": "source-rich",
            "segmenter": "sat-3l-sm",
            "segmenter_version": "test",
            "model_id": "segment-any-text/sat-3l-sm",
            "segmentation_status": "split",
            "license": "CC BY-SA 4.0",
            "attribution": "Wikipedia",
        },
    ]


def test_split_sections_uses_the_default_batch_size() -> None:
    assert signature(split_sections).parameters["batch_size"].default == DEFAULT_SENTENCE_BATCH_SIZE
    assert DEFAULT_SENTENCE_BATCH_SIZE == 256


def test_split_sections_batches_supported_sections() -> None:
    sections = [_section(f"en-{index}", "en", f"Sentence {index}.") for index in range(3)]
    segmenter = _FakeSegmenter(
        {str(section["text"]): [str(section["text"])] for section in sections}
    )

    rows, summary = split_sections(sections, segmenter=segmenter, batch_size=2)

    assert [len(texts) for _, texts in segmenter.calls] == [2, 1]
    assert len(rows) == 3
    assert summary.sections == 3


def test_split_sections_batches_supported_languages_together_for_mixed_segmenter() -> None:
    class _MixedFakeSegmenter(_FakeSegmenter):
        supports_mixed_languages = True

    sections = [
        _section("fr-1", "fr", "Un. Deux."),
        _section("en-1", "en", "First. Second."),
    ]
    segmenter = _MixedFakeSegmenter(
        {
            "Un. Deux.": ["Un. ", "Deux."],
            "First. Second.": ["First. ", "Second."],
        }
    )

    rows, summary = split_sections(sections, segmenter=segmenter, batch_size=2)

    assert segmenter.calls == [("mixed", ("Un. Deux.", "First. Second."))]
    assert [row["text"] for row in rows] == ["Un. ", "Deux.", "First. ", "Second."]
    assert summary.supported_languages == ("en", "fr")


def test_split_sections_uses_source_text_when_source_hash_is_missing() -> None:
    section = _section("fallback", "en", "Alpha. Beta!")
    section.pop("content_hash")
    segmenter = _FakeSegmenter({"Alpha. Beta!": ["Alpha. Beta!"]})

    rows, _ = split_sections([section], segmenter=segmenter)

    assert rows[0]["source_content_hash"] == (
        "b6f8d1e40c91f4e9dab279fc8204ca2565fbc8565e0d3c598c2bde2bcaa2a4be"
    )


@pytest.mark.parametrize(("value", "expected"), [(None, ""), (123, "123")])
def test_split_sections_normalizes_non_string_source_text(value: object, expected: str) -> None:
    section = _section("text-value", "xx", "")
    section["text"] = value

    rows, _ = split_sections([section], segmenter=_FakeSegmenter({}))

    assert rows[0]["text"] == expected
    assert rows[0]["end_char"] == len(expected)


def test_split_sections_treats_missing_language_as_unsupported() -> None:
    section = _section("missing-language", "xx", "Unknown.")
    section.pop("language")

    rows, summary = split_sections([section], segmenter=_FakeSegmenter({}))

    assert rows[0]["segmentation_status"] == "unsupported_language"
    assert summary.unsupported_languages == ("",)


def test_split_sections_uses_empty_section_id_fallback_for_sentence_id() -> None:
    section = _section("no-section", "en", "One.")
    section.pop("section_id")
    segmenter = _FakeSegmenter({"One.": ["One."]})

    rows, _ = split_sections([section], segmenter=segmenter)

    assert rows[0]["sentence_id"] == (
        "6750295dc7b004c9996d7119510a7e243652fd197dfd3d6c3007351a2459e7da"
    )


def test_normalize_pieces_converts_non_string_model_output() -> None:
    assert _normalize_pieces([123]) == ["123"]  # type: ignore[list-item]


def test_split_sections_rejects_model_output_that_loses_source_text() -> None:
    section = _section("en-1", "en", "First. Second.")
    segmenter = _FakeSegmenter({section["text"]: ["First."]})

    with pytest.raises(ValueError) as error:
        split_sections([section], segmenter=segmenter)

    assert str(error.value) == (
        "Sentence segmentation for section 'en-1' does not reconstruct the source text"
    )


def test_split_sections_counts_every_unsupported_section() -> None:
    sections = [
        _section("xx-1", "xx", "Unknown one."),
        _section("xx-2", "xx", "Unknown two."),
    ]

    rows, summary = split_sections(
        sections,
        segmenter=_FakeSegmenter({}),
    )

    assert len(rows) == 2
    assert summary.unsplit_sections == 2
    assert summary.unsupported_languages == ("xx",)


def test_split_sections_handles_empty_supported_text_without_model_call() -> None:
    section = _section("en-empty", "en", "")
    segmenter = _FakeSegmenter({})

    rows, summary = split_sections([section], segmenter=segmenter)

    assert rows == []
    assert segmenter.calls == []
    assert summary.split_sections == 1
    assert summary.sentence_rows == 0


@pytest.mark.parametrize("batch_size", [0, -1])
def test_split_sections_rejects_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError) as error:
        split_sections([], segmenter=_FakeSegmenter({}), batch_size=batch_size)

    assert str(error.value) == "batch_size must be positive"


def test_split_sections_rejects_wrong_result_count() -> None:
    class _WrongCountSegmenter(_FakeSegmenter):
        def split(self, texts: list[str], *, language: str) -> list[list[str]]:
            self.calls.append((language, tuple(texts)))
            return []

    with pytest.raises(ValueError, match=r"returned 0 result\(s\) for 1"):
        split_sections(
            [_section("en-1", "en", "One.")],
            segmenter=_WrongCountSegmenter({}),
        )


def test_split_sections_rejects_empty_sentence_piece() -> None:
    section = _section("en-1", "en", "One.")
    segmenter = _FakeSegmenter({"One.": ["", "One."]})

    with pytest.raises(ValueError) as error:
        split_sections([section], segmenter=segmenter)

    assert str(error.value) == "Sentence segmenter returned an empty sentence"
