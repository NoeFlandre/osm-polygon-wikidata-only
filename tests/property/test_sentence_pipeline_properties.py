from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from osm_polygon_wikidata_only.domain.ids import content_hash
from osm_polygon_wikidata_only.v2.sentence_logic import (
    SAT_MODEL_ID,
    SAT_SUPPORTED_LANGUAGES,
    split_sections,
)

_UNSUPPORTED_LANGUAGES = ("xx", "zh-hans", "en-US", "")
# All non-surrogate Unicode scalar values are UTF-8 encodable; this includes
# whitespace without a rejection-heavy filter or a handpicked alphabet.
_UNICODE_CHARACTERS = st.characters(blacklist_categories=("Cs",))
_SUPPORTED_LANGUAGE_SET = frozenset(SAT_SUPPORTED_LANGUAGES)


@dataclass(frozen=True)
class _GeneratedSection:
    section_id: str
    language: str
    text: str


@dataclass
class _DeterministicSegmenter:
    model_id: str = SAT_MODEL_ID
    version: str = "hypothesis-fake"
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def split(
        self,
        texts: Sequence[str],
        *,
        language: str,
    ) -> list[list[str]]:
        self.calls.append((language, tuple(texts)))
        return [_lossless_pieces(text) for text in texts]


def _lossless_pieces(text: str) -> list[str]:
    step = 1 + len(text) % 4
    return [text[start : start + step] for start in range(0, len(text), step)]


def _expected_supported_language(language: str) -> bool:
    return language in _SUPPORTED_LANGUAGE_SET


def _section(section: _GeneratedSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "document_id": f"document-{section.section_id}",
        "article_id": f"article-{section.section_id}",
        "project": "wikipedia",
        "language": section.language,
        "site": f"{section.language}wiki",
        "text": section.text,
        "content_hash": content_hash(section.text),
    }


@st.composite
def _sections(draw: st.DrawFn) -> list[_GeneratedSection]:
    language = st.one_of(
        st.sampled_from(SAT_SUPPORTED_LANGUAGES),
        st.sampled_from(_UNSUPPORTED_LANGUAGES),
    )
    languages = draw(st.lists(language, min_size=1, max_size=8))
    texts = draw(
        st.lists(
            st.text(alphabet=_UNICODE_CHARACTERS, min_size=0, max_size=48),
            min_size=len(languages),
            max_size=len(languages),
        )
    )
    return [
        _GeneratedSection(f"section-{index}", language, text)
        for index, (language, text) in enumerate(zip(languages, texts, strict=True))
    ]


def _rows_for_section(rows: list[dict[str, Any]], section_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["section_id"] == section_id]


@settings(max_examples=50, derandomize=True, database=None, deadline=None)
@given(sections=_sections())
def test_generated_rows_preserve_each_source_and_offset(sections: list[_GeneratedSection]) -> None:
    segmenter = _DeterministicSegmenter()
    rows, summary = split_sections(
        [_section(section) for section in sections],
        segmenter=segmenter,
        batch_size=2,
    )

    expected_row_count = 0
    expected_ids: list[str] = []
    supported_languages = {
        section.language for section in sections if _expected_supported_language(section.language)
    }
    unsupported_languages = {
        section.language
        for section in sections
        if not _expected_supported_language(section.language)
    }

    for section in sections:
        section_rows = _rows_for_section(rows, section.section_id)
        expected_pieces = (
            _lossless_pieces(section.text)
            if _expected_supported_language(section.language)
            else [section.text]
        )
        expected_row_count += len(expected_pieces)
        expected_ids.extend([section.section_id] * len(expected_pieces))

        assert [row["text"] for row in section_rows] == expected_pieces
        assert "".join(row["text"] for row in section_rows) == section.text

        offset = 0
        for sentence_index, row in enumerate(section_rows):
            assert row["sentence_index"] == sentence_index
            assert row["start_char"] == offset
            assert row["end_char"] == offset + len(row["text"])
            assert row["text"] == section.text[row["start_char"] : row["end_char"]]
            assert row["text_length_chars"] == len(row["text"])
            assert row["source_content_hash"] == content_hash(section.text)
            assert row["content_hash"] == content_hash(row["text"])
            offset = row["end_char"]
        assert offset == len(section.text)

        if _expected_supported_language(section.language):
            assert all(row["segmentation_status"] == "split" for row in section_rows)
            assert all(row["segmenter"] == "sat-3l-sm" for row in section_rows)
        else:
            assert len(section_rows) == 1
            assert section_rows[0]["segmentation_status"] == "unsupported_language"
            assert section_rows[0]["segmenter"] == "unsplit"
            assert section_rows[0]["model_id"] == ""

    assert [row["section_id"] for row in rows] == expected_ids
    assert len(rows) == expected_row_count
    assert summary.sections == len(sections)
    assert summary.split_sections == sum(
        _expected_supported_language(section.language) for section in sections
    )
    assert summary.unsplit_sections == sum(
        not _expected_supported_language(section.language) for section in sections
    )
    assert summary.sentence_rows == expected_row_count
    assert summary.supported_languages == tuple(sorted(supported_languages))
    assert summary.unsupported_languages == tuple(sorted(unsupported_languages))

    called_languages = {language for language, _ in segmenter.calls}
    nonempty_supported_languages = {
        section.language
        for section in sections
        if _expected_supported_language(section.language) and section.text
    }
    assert called_languages == nonempty_supported_languages
    assert all(language not in unsupported_languages for language, _ in segmenter.calls)


def test_supported_empty_section_is_filtered_before_segmenter_call() -> None:
    segmenter = _DeterministicSegmenter()
    rows, summary = split_sections(
        [_section(_GeneratedSection("empty", "en", ""))],
        segmenter=segmenter,
    )

    assert rows == []
    assert summary.sections == 1
    assert summary.split_sections == 1
    assert summary.unsplit_sections == 0
    assert summary.sentence_rows == 0
    assert summary.supported_languages == ("en",)
    assert summary.unsupported_languages == ()
    assert segmenter.calls == []


@settings(max_examples=50, derandomize=True, database=None, deadline=None)
@given(
    sections=_sections(),
    first_batch_size=st.integers(min_value=1, max_value=4),
    second_batch_size=st.integers(min_value=1, max_value=4),
)
def test_generated_output_is_invariant_to_supported_batch_boundaries(
    sections: list[_GeneratedSection],
    first_batch_size: int,
    second_batch_size: int,
) -> None:
    first_rows, first_summary = split_sections(
        [_section(section) for section in sections],
        segmenter=_DeterministicSegmenter(),
        batch_size=first_batch_size,
    )
    second_rows, second_summary = split_sections(
        [_section(section) for section in sections],
        segmenter=_DeterministicSegmenter(),
        batch_size=second_batch_size,
    )

    assert first_rows == second_rows
    assert first_summary == second_summary
