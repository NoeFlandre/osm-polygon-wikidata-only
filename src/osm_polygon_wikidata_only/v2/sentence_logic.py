"""Pure sentence-row construction and exact SaT language routing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.models import stable_id
from osm_polygon_wikidata_only.domain.ids import content_hash
from osm_polygon_wikidata_only.enrichment.text_cleaning import count_words, estimate_tokens

SAT_MODEL_NAME = "sat-3l-sm"
SAT_MODEL_ID = "segment-any-text/sat-3l-sm"
DEFAULT_SENTENCE_BATCH_SIZE = 256

# This is the exact ISO-code set documented by the SaT model card.  Matching
# is intentionally exact: language variants not in this set remain unsplit.
SAT_SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
)
_SAT_SUPPORTED_LANGUAGE_SET = frozenset(SAT_SUPPORTED_LANGUAGES)

SENTENCE_COLUMNS: tuple[str, ...] = (
    "sentence_id",
    "section_id",
    "document_id",
    "article_id",
    "wikidata",
    "project",
    "language",
    "site",
    "page_id",
    "revision_id",
    "section_index",
    "heading",
    "anchor",
    "level",
    "parent_section_id",
    "section_path",
    "sentence_index",
    "start_char",
    "end_char",
    "text",
    "text_length_chars",
    "text_length_words",
    "text_length_tokens_estimate",
    "content_hash",
    "source_content_hash",
    "segmenter",
    "segmenter_version",
    "model_id",
    "segmentation_status",
    "license",
    "attribution",
)
_INTEGER_COLUMNS = frozenset(
    {
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
)
_SECTION_CONTEXT_COLUMNS = (
    "section_id",
    "document_id",
    "article_id",
    "wikidata",
    "project",
    "language",
    "site",
    "page_id",
    "revision_id",
    "section_index",
    "heading",
    "anchor",
    "level",
    "parent_section_id",
    "section_path",
    "license",
    "attribution",
)


class SentenceSegmenter(Protocol):
    """Batch interface implemented by the concrete SaT adapter."""

    model_id: str
    version: str

    def split(self, texts: Sequence[str], *, language: str) -> Sequence[Sequence[str]]:
        """Return lossless sentence pieces for each input text."""


@dataclass(frozen=True, slots=True)
class SentenceSplitSummary:
    """Counts and language routing recorded for one deterministic run."""

    sections: int
    split_sections: int
    unsplit_sections: int
    sentence_rows: int
    supported_languages: tuple[str, ...]
    unsupported_languages: tuple[str, ...]


@lru_cache(maxsize=1)
def sentence_schema() -> pa.Schema:
    """Return the stable Parquet schema for sentence sidecars."""
    return pa.schema(
        [
            pa.field(
                column,
                pa.int64() if column in _INTEGER_COLUMNS else pa.string(),
            )
            for column in SENTENCE_COLUMNS
        ]
    )


def is_sat_supported_language(language: str) -> bool:
    """Return whether ``language`` is an exact SaT-3l-sm language code."""
    return language in _SAT_SUPPORTED_LANGUAGE_SET


def split_sections(
    sections: Sequence[dict[str, Any]],
    *,
    segmenter: SentenceSegmenter,
    batch_size: int = DEFAULT_SENTENCE_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], SentenceSplitSummary]:
    """Split supported sections and emit one explicit row for other languages."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rows_by_position, supported_positions, unsupported_languages, unsupported_count = (
        _classify_sections(sections)
    )
    _populate_supported_rows(
        sections,
        rows_by_position,
        supported_positions,
        segmenter=segmenter,
        batch_size=batch_size,
    )

    rows = _rows_in_input_order(rows_by_position, section_count=len(sections))
    summary = SentenceSplitSummary(
        sections=len(sections),
        split_sections=sum(len(positions) for positions in supported_positions.values()),
        unsplit_sections=unsupported_count,
        sentence_rows=len(rows),
        supported_languages=tuple(sorted(supported_positions)),
        unsupported_languages=tuple(sorted(unsupported_languages)),
    )
    return rows, summary


def _classify_sections(
    sections: Sequence[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, list[int]], set[str], int]:
    rows_by_position: dict[int, list[dict[str, Any]]] = {}
    supported_positions: dict[str, list[int]] = defaultdict(list)
    unsupported_languages: set[str] = set()
    unsupported_count = 0
    for position, section in enumerate(sections):
        language = _language(section)
        if is_sat_supported_language(language):
            supported_positions[language].append(position)
            rows_by_position[position] = []
            continue
        unsupported_languages.add(language)
        unsupported_count += 1
        rows_by_position[position] = [_unsplit_row(section)]
    return rows_by_position, supported_positions, unsupported_languages, unsupported_count


def _populate_supported_rows(
    sections: Sequence[dict[str, Any]],
    rows_by_position: dict[int, list[dict[str, Any]]],
    supported_positions: dict[str, list[int]],
    *,
    segmenter: SentenceSegmenter,
    batch_size: int,
) -> None:
    if getattr(segmenter, "supports_mixed_languages", False):  # pragma: no mutate
        positions = sorted(
            position for values in supported_positions.values() for position in values
        )
        _populate_positions(
            sections,
            rows_by_position,
            language="mixed",
            positions=positions,
            segmenter=segmenter,
            batch_size=batch_size,
        )
        return
    for language in sorted(supported_positions):
        _populate_language_rows(
            sections,
            rows_by_position,
            language=language,
            positions=supported_positions[language],
            segmenter=segmenter,
            batch_size=batch_size,
        )


def _populate_language_rows(
    sections: Sequence[dict[str, Any]],
    rows_by_position: dict[int, list[dict[str, Any]]],
    *,
    language: str,
    positions: Sequence[int],
    segmenter: SentenceSegmenter,
    batch_size: int,
) -> None:
    _populate_positions(
        sections,
        rows_by_position,
        language=language,
        positions=positions,
        segmenter=segmenter,
        batch_size=batch_size,
    )


def _populate_positions(
    sections: Sequence[dict[str, Any]],
    rows_by_position: dict[int, list[dict[str, Any]]],
    *,
    language: str,
    positions: Sequence[int],
    segmenter: SentenceSegmenter,
    batch_size: int,
) -> None:
    nonempty_positions = [position for position in positions if _text(sections[position])]
    for start in range(0, len(nonempty_positions), batch_size):
        _populate_batch(
            sections,
            rows_by_position,
            language=language,
            positions=nonempty_positions[start : start + batch_size],
            segmenter=segmenter,
        )


def _populate_batch(
    sections: Sequence[dict[str, Any]],
    rows_by_position: dict[int, list[dict[str, Any]]],
    *,
    language: str,
    positions: Sequence[int],
    segmenter: SentenceSegmenter,
) -> None:
    texts = [_text(sections[position]) for position in positions]
    batches = list(segmenter.split(texts, language=language))
    if len(batches) != len(positions):
        raise ValueError(
            f"Sentence segmenter returned {len(batches)} result(s) for "
            f"{len(positions)} input section(s)"
        )
    for index, position in enumerate(positions):
        pieces = batches[index]
        rows_by_position[position] = _split_rows(
            sections[position],
            pieces,
            model_id=segmenter.model_id,
            segmenter_version=segmenter.version,
        )


def _rows_in_input_order(
    rows_by_position: dict[int, list[dict[str, Any]]], *, section_count: int
) -> list[dict[str, Any]]:
    return [row for position in range(section_count) for row in rows_by_position[position]]


def _language(section: dict[str, Any]) -> str:
    return str(section.get("language") or "")


def _text(section: dict[str, Any]) -> str:
    value = section.get("text")
    return value if isinstance(value, str) else str(value or "")


def _unsplit_row(section: dict[str, Any]) -> dict[str, Any]:
    text = _text(section)
    return _sentence_row(
        section,
        sentence_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        segmenter="unsplit",
        segmenter_version="",
        model_id="",
        segmentation_status="unsupported_language",
    )


def _split_rows(
    section: dict[str, Any],
    pieces: Sequence[str],
    *,
    model_id: str,
    segmenter_version: str,
) -> list[dict[str, Any]]:
    source = _text(section)
    normalized_pieces = _normalize_pieces(pieces)
    _validate_reconstruction(section, source, normalized_pieces)
    rows: list[dict[str, Any]] = []
    offset = 0
    for sentence_index, piece in enumerate(normalized_pieces):
        _validate_piece(piece)
        end = offset + len(piece)
        rows.append(
            _sentence_row(
                section,
                sentence_index=sentence_index,
                start_char=offset,
                end_char=end,
                text=piece,
                segmenter=SAT_MODEL_NAME,
                segmenter_version=segmenter_version,
                model_id=model_id,
                segmentation_status="split",
            )
        )
        offset = end
    return rows


def _normalize_pieces(pieces: Sequence[str]) -> list[str]:
    return [piece if isinstance(piece, str) else str(piece) for piece in pieces]


def _validate_reconstruction(section: dict[str, Any], source: str, pieces: Sequence[str]) -> None:
    if "".join(pieces) != source:
        raise ValueError(
            f"Sentence segmentation for section {section.get('section_id')!r} "
            "does not reconstruct the source text"
        )


def _validate_piece(piece: str) -> None:
    if not piece:
        raise ValueError("Sentence segmenter returned an empty sentence")


def _sentence_row(
    section: dict[str, Any],
    *,
    sentence_index: int,
    start_char: int,
    end_char: int,
    text: str,
    segmenter: str,
    segmenter_version: str,
    model_id: str,
    segmentation_status: str,
) -> dict[str, Any]:
    source_text = _text(section)
    source_content_hash = str(section.get("content_hash") or content_hash(source_text))
    row = {column: section.get(column) for column in _SECTION_CONTEXT_COLUMNS}
    row.update(
        {
            "sentence_id": stable_id(
                "v2-sentence",
                str(section.get("section_id") or ""),
                source_content_hash,
                start_char,
                end_char,
            ),
            "sentence_index": sentence_index,
            "start_char": start_char,
            "end_char": end_char,
            "text": text,
            "text_length_chars": len(text),
            "text_length_words": count_words(text),
            "text_length_tokens_estimate": estimate_tokens(text),
            "content_hash": content_hash(text),
            "source_content_hash": source_content_hash,
            "segmenter": segmenter,
            "segmenter_version": segmenter_version,
            "model_id": model_id,
            "segmentation_status": segmentation_status,
        }
    )
    return row


__all__ = [
    "DEFAULT_SENTENCE_BATCH_SIZE",
    "SAT_MODEL_ID",
    "SAT_MODEL_NAME",
    "SAT_SUPPORTED_LANGUAGES",
    "SENTENCE_COLUMNS",
    "SentenceSegmenter",
    "SentenceSplitSummary",
    "is_sat_supported_language",
    "sentence_schema",
    "split_sections",
]
