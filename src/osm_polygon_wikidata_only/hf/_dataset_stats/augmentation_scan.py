"""Row-level scanning of augmentation sidecar Parquet files."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .cache import file_fingerprint as _file_fingerprint
from .cache import relative_path as _relative_path
from .models import (
    PerFileSummary,
)
from .scanning import safe_table

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")

# Document columns actually used by the scanner.
DOCUMENT_COLUMNS: tuple[str, ...] = (
    "document_id",
    "wikidata",
    "project",
    "language",
    "full_text",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
)
SECTION_COLUMNS: tuple[str, ...] = (
    "section_id",
    "document_id",
    "wikidata",
    "project",
    "language",
    "text",
    "text_length_chars",
    "text_length_words",
    "text_length_tokens_estimate",
)
FACT_COLUMNS: tuple[str, ...] = (
    "fact_id",
    "wikidata",
    "property_id",
    "property_label_en",
    "property_labels",
    "value_type",
    "value_entity_id",
    "value_label_en",
    "value_labels",
    "value_text",
    "qualifiers",
    "references",
)

KIND_DOCUMENT = "documents"
KIND_SECTION = "sections"
KIND_FACT = "facts"


# ---------------------------------------------------------------------------
# JSON helper detection
# ---------------------------------------------------------------------------


def _has_json_content(value: object) -> bool:
    """A column cell counts as "present JSON" when it is a valid
    non-empty JSON array or object."""
    if value is None:
        return False
    if isinstance(value, str):
        return _json_text_has_content(value)
    return _json_collection_has_content(value)


def _json_text_has_content(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return False
    return _json_collection_has_content(parsed)


def _json_collection_has_content(value: object) -> bool:
    return isinstance(value, (list, dict)) and len(value) > 0


def _section_row_metrics(
    text: Any, chars: Any, words: Any, tokens: Any
) -> tuple[int, int, int, int, int]:
    """Return non-empty/empty flags and numeric text lengths for one row."""
    text_value = text if isinstance(text, str) else ""
    non_empty = int(bool(text_value and text_value.strip()))
    empty_or_null = 1 - non_empty
    return (
        non_empty,
        empty_or_null,
        _length_or_zero(chars),
        _length_or_zero(words),
        _length_or_zero(tokens),
    )


def _length_or_zero(value: Any) -> int:
    """Convert an optional stored length to an integer."""
    return int(str(value)) if value is not None else 0


def _optional_string(value: Any) -> tuple[str, ...]:
    """Return a one-item string tuple when a cell has a value."""
    return (str(value),) if value else ()


def _section_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract set/counter updates for one section row."""
    return (
        _optional_string(arrays.get("section_id", [None])[index]),
        _optional_string(arrays.get("document_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("language", [None])[index]),
    )


def _document_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract document, subject, and language updates for one row."""
    return (
        _optional_string(arrays.get("document_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("language", [None])[index]),
    )


def _json_field_counts(value: object) -> tuple[int, int]:
    """Return ``(available, unavailable)`` counts for one JSON cell."""
    if _has_json_content(value):
        return 1, 0
    if isinstance(value, str) and value.strip():
        return 0, 1
    return 0, 0


def _fact_row_identity(
    arrays: dict[str, list[Any]], index: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract fact, subject, and property updates for one row."""
    return (
        _optional_string(arrays.get("fact_id", [None])[index]),
        _optional_string(arrays.get("wikidata", [None])[index]),
        _optional_string(arrays.get("property_id", [None])[index]),
    )


def _nonempty_text(value: object) -> int:
    """Return one when a value is a non-blank string."""
    return int(isinstance(value, str) and bool(value.strip()))


def _present_string(value: object) -> tuple[str, ...]:
    """Return one non-empty string value without coercing other types."""
    return (value,) if isinstance(value, str) and value else ()


def _property_label_update(
    property_id: object, property_label: object
) -> tuple[tuple[str, str], ...]:
    """Return a first-seen property-label candidate for one fact row."""
    if not property_id or not isinstance(property_label, str):
        return ()
    return ((str(property_id), property_label.strip()),)


def _record_property_updates(
    property_ids: tuple[str, ...],
    property_label: object,
    property_counts: Counter[str],
    property_labels: dict[str, str],
) -> None:
    """Merge one row's property count and first-seen label."""
    property_counts.update(property_ids)
    for property_id, label in _property_label_update(
        property_ids[0] if property_ids else None, property_label
    ):
        property_labels.setdefault(property_id, label)


# ---------------------------------------------------------------------------
# Per-file scanning
# ---------------------------------------------------------------------------


def _scan_documents_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikipedia/documents`` or ``wikivoyage/documents``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(DOCUMENT_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_DOCUMENT,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in DOCUMENT_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    document_ids: set[str] = set()
    qids: set[str] = set()
    languages: Counter[str] = Counter()
    non_empty = 0
    empty_or_null = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0

    for i in range(rows):
        row_document_ids, row_qids, row_languages = _document_row_identity(arrays, i)
        full_text = arrays.get("full_text", [None])[i]
        document_ids.update(row_document_ids)
        qids.update(row_qids)
        languages.update(row_languages)
        chars = arrays.get("article_length_chars", [None])[i]
        words = arrays.get("article_length_words", [None])[i]
        tokens = arrays.get("article_length_tokens_estimate", [None])[i]
        row_non_empty, row_empty, row_chars, row_words, row_tokens = _section_row_metrics(
            full_text, chars, words, tokens
        )
        non_empty += row_non_empty
        empty_or_null += row_empty
        total_chars += row_chars
        total_words += row_words
        total_tokens += row_tokens

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_DOCUMENT,
        rows=rows,
        non_empty=non_empty,
        empty_or_null=empty_or_null,
        total_chars=total_chars,
        total_words=total_words,
        total_tokens_estimate=total_tokens,
        document_ids=frozenset(document_ids),
        qids=frozenset(qids),
        languages=dict(languages),
    )


def _scan_sections_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikipedia/sections`` or ``wikivoyage/sections``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(SECTION_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_SECTION,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in SECTION_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    document_ids: set[str] = set()
    section_ids: set[str] = set()
    qids: set[str] = set()
    languages: Counter[str] = Counter()
    non_empty = 0
    empty_or_null = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0

    for i in range(rows):
        row_section_ids, row_document_ids, row_qids, row_languages = _section_row_identity(
            arrays, i
        )
        text = arrays.get("text", [None])[i]
        section_ids.update(row_section_ids)
        document_ids.update(row_document_ids)
        qids.update(row_qids)
        languages.update(row_languages)
        chars = arrays.get("text_length_chars", [None])[i]
        words = arrays.get("text_length_words", [None])[i]
        tokens = arrays.get("text_length_tokens_estimate", [None])[i]
        row_non_empty, row_empty, row_chars, row_words, row_tokens = _section_row_metrics(
            text, chars, words, tokens
        )
        non_empty += row_non_empty
        empty_or_null += row_empty
        total_chars += row_chars
        total_words += row_words
        total_tokens += row_tokens

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_SECTION,
        rows=rows,
        non_empty=non_empty,
        empty_or_null=empty_or_null,
        total_chars=total_chars,
        total_words=total_words,
        total_tokens_estimate=total_tokens,
        document_ids=frozenset(document_ids),
        section_ids=frozenset(section_ids),
        qids=frozenset(qids),
        languages=dict(languages),
    )


def _scan_facts_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary:
    """Aggregate one ``wikidata/facts``."""
    rel = _relative_path(processed_dir, parquet_path)
    fp = _file_fingerprint(parquet_path)
    file_size = parquet_path.stat().st_size
    table = safe_table(parquet_path, list(FACT_COLUMNS))
    if table is None:
        return PerFileSummary(
            relative_path=rel,
            fingerprint=fp,
            file_size_bytes=file_size,
            kind=KIND_FACT,
            scan_failed=True,
        )

    arrays: dict[str, list[Any]] = {
        col: table.column(col).to_pylist() for col in FACT_COLUMNS if col in table.schema.names
    }

    rows = table.num_rows
    fact_ids: set[str] = set()
    subjects: set[str] = set()
    properties: set[str] = set()
    property_labels: dict[str, str] = {}
    property_counts: Counter[str] = Counter()
    with_prop_en = 0
    with_value_en = 0
    with_qualifiers = 0
    with_references = 0
    unavailable_qualifiers = 0
    unavailable_references = 0
    value_types: Counter[str] = Counter()

    for i in range(rows):
        row_fact_ids, row_subjects, row_properties = _fact_row_identity(arrays, i)
        property_label_en = arrays.get("property_label_en", [None])[i]
        value_type = arrays.get("value_type", [None])[i]
        value_label_en = arrays.get("value_label_en", [None])[i]
        qualifiers = arrays.get("qualifiers", [None])[i]
        references = arrays.get("references", [None])[i]
        fact_ids.update(row_fact_ids)
        subjects.update(row_subjects)
        properties.update(row_properties)
        with_prop_en += _nonempty_text(property_label_en)
        with_value_en += _nonempty_text(value_label_en)
        qualifier_count, unavailable_qualifier_count = _json_field_counts(qualifiers)
        with_qualifiers += qualifier_count
        unavailable_qualifiers += unavailable_qualifier_count
        reference_count, unavailable_reference_count = _json_field_counts(references)
        with_references += reference_count
        unavailable_references += unavailable_reference_count
        value_types.update(_present_string(value_type))
        _record_property_updates(
            row_properties, property_label_en, property_counts, property_labels
        )

    return PerFileSummary(
        relative_path=rel,
        fingerprint=fp,
        file_size_bytes=file_size,
        kind=KIND_FACT,
        fact_rows=rows,
        fact_ids=frozenset(fact_ids),
        subject_qids=frozenset(subjects),
        property_ids=frozenset(properties),
        property_labels=dict(property_labels),
        property_counts=dict(property_counts),
        with_property_en_label=with_prop_en,
        with_value_en_label=with_value_en,
        with_qualifiers=with_qualifiers,
        with_references=with_references,
        unavailable_qualifiers=unavailable_qualifiers,
        unavailable_references=unavailable_references,
        value_type_counts=dict(value_types),
    )


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _kind_for_rel(rel: str) -> str:
    """Return the augmentation kind for a sidecar path relative to
    ``<processed>/``."""
    for kind, prefixes in (
        (KIND_DOCUMENT, ("wikipedia/documents/", "wikivoyage/documents/")),
        (KIND_SECTION, ("wikipedia/sections/", "wikivoyage/sections/")),
        (KIND_FACT, ("wikidata/facts/",)),
    ):
        if rel.startswith(prefixes):
            return kind
    return ""


def scan_one_file(processed_dir: Path, parquet_path: Path) -> PerFileSummary | None:
    """Dispatch a single sidecar file to its specialized scanner."""
    try:
        rel = _relative_path(processed_dir, parquet_path)
    except ValueError:
        return None
    kind = _kind_for_rel(rel)
    if kind == KIND_DOCUMENT:
        return _scan_documents_file(processed_dir, parquet_path)
    if kind == KIND_SECTION:
        return _scan_sections_file(processed_dir, parquet_path)
    if kind == KIND_FACT:
        return _scan_facts_file(processed_dir, parquet_path)
    return None
