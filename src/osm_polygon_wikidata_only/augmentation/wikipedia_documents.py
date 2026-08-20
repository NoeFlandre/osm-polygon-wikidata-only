"""Canonical Wikipedia document contract: lossless 32-column schema, model, and conversion.

This module provides the canonical schema used to migrate the legacy
``articles/`` table into ``wikipedia/documents/`` without losing a
field. The smaller shared ``Document`` model remains in use internally
for section extraction and Wikivoyage.

The canonical 32-column layout preserves every article field plus
``document_id`` (deterministic identity) and ``project`` (always
``"wikipedia"``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    article_schema,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import is_valid_qid

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Canonical 32-column layout: document_id and project inserted at positions
# 0 and 3 respectively (after wikidata, before language), with all 30 article
# columns preserved in their original ARTICLE_COLUMNS order.
WIKIPEDIA_DOCUMENT_COLUMNS: tuple[str, ...] = (
    "document_id",
    "article_id",
    "wikidata",
    "project",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "retrieved_at",
    "wikidata_label",
    "wikidata_description",
    "wikidata_aliases",
    "lead_text",
    "extract",
    "full_text",
    "full_text_format",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
    "thumbnail_url",
    "thumbnail_width",
    "thumbnail_height",
    "categories",
    "license",
    "attribution",
    "source_api",
    "fetch_status",
    "fetch_error",
    "content_hash",
)

WIKIPEDIA_DOCUMENT_DESCRIPTIONS: dict[str, str] = {
    "document_id": (
        "Deterministic document identifier "
        "(`<wikidata>:<project>:<language>:<page_id>:<revision_id>`)."
    ),
    "project": "Wiki project name: always `wikipedia` for this table.",
    **{col: ARTICLE_DESCRIPTIONS[col] for col in ARTICLE_COLUMNS},
}


def wikipedia_document_schema() -> pa.Schema:
    """Build the canonical 32-column PyArrow schema.

    Inherited fields are taken directly from ``article_schema()`` including
    their original field metadata (description, etc.).  The two new fields
    (``document_id``, ``project``) carry description metadata matching
    ``WIKIPEDIA_DOCUMENT_DESCRIPTIONS``.
    """
    art = article_schema()
    fields: list[pa.Field] = []
    for col in WIKIPEDIA_DOCUMENT_COLUMNS:
        if col in ("document_id", "project"):
            fields.append(
                pa.field(
                    col,
                    pa.string(),
                    metadata={b"description": WIKIPEDIA_DOCUMENT_DESCRIPTIONS[col].encode()},
                )
            )
        else:
            # Preserve the original pa.Field including its metadata
            fields.append(art.field(col))
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class WikipediaDocumentConversionError(Exception):
    """Raised when an article row cannot be converted to a WikipediaDocument."""


# ---------------------------------------------------------------------------
# Identity validation
# ---------------------------------------------------------------------------


def _validate_qid(value: object, field: str) -> str:
    """Validate that value is a valid Wikidata QID string (Q[1-9][0-9]*)."""
    if not isinstance(value, str):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected str, got {type(value).__name__}"
        )
    if not is_valid_qid(value):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': invalid Wikidata QID '{value}' (must match Q[1-9][0-9]*)"
        )
    return value


def _validate_language(value: object, field: str) -> str:
    """Validate that value is a non-empty language string without colons or whitespace."""
    language = _require_language_string(value, field)
    _validate_language_format(language, field)
    return language


def _require_language_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected str, got {type(value).__name__}"
        )
    return value


def _validate_language_format(value: str, field: str) -> None:
    _validate_language_whitespace(value, field)
    _validate_language_colon(value, field)


def _validate_language_whitespace(value: str, field: str) -> None:
    if not value or value.strip() != value or any(c.isspace() for c in value):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': language must be non-empty and have no whitespace, got {value!r}"
        )


def _validate_language_colon(value: str, field: str) -> None:
    if ":" in value:
        raise WikipediaDocumentConversionError(
            f"Field '{field}': language must not contain ':', got '{value}'"
        )


def _validate_positive_int(value: object, field: str) -> int:
    """Validate that value is a positive int (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected int (not bool), got {type(value).__name__}"
        )
    if value <= 0:
        raise WikipediaDocumentConversionError(f"Field '{field}': must be positive, got {value}")
    return value


# ---------------------------------------------------------------------------
# Strict type validation helpers
# ---------------------------------------------------------------------------


def _require_exact_str(value: object, field: str) -> str:
    """Require value is already ``str``. No coercion."""
    if not isinstance(value, str):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected str, got {type(value).__name__}"
        )
    return value


def _require_exact_int(value: object, field: str) -> int:
    """Require value is already ``int`` (excluding ``bool``). No coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected int (not bool), got {type(value).__name__}"
        )
    return value


def _require_optional_int(value: object, field: str) -> int | None:
    """Require value is ``None`` or ``int`` (excluding ``bool``). No coercion."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WikipediaDocumentConversionError(
            f"Field '{field}': expected int | None (not bool), got {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WikipediaDocument:
    """Frozen, slotted model representing a canonical 32-column Wikipedia document."""

    document_id: str
    article_id: str
    wikidata: str
    project: str
    language: str
    site: str
    title: str
    url: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    retrieved_at: str
    wikidata_label: str
    wikidata_description: str
    wikidata_aliases: str
    lead_text: str
    extract: str
    full_text: str
    full_text_format: str
    article_length_chars: int
    article_length_words: int
    article_length_tokens_estimate: int
    thumbnail_url: str
    thumbnail_width: int | None
    thumbnail_height: int | None
    categories: str
    license: str
    attribution: str
    source_api: str
    fetch_status: str
    fetch_error: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict in canonical column order."""
        return {col: getattr(self, col) for col in WIKIPEDIA_DOCUMENT_COLUMNS}


# ---------------------------------------------------------------------------
# Row Conversion
# ---------------------------------------------------------------------------


def _make_document_id(wikidata: str, language: str, page_id: int, revision_id: int) -> str:
    """Build the deterministic document_id."""
    return f"{wikidata}:wikipedia:{language}:{page_id}:{revision_id}"


def _make_article_id(wikidata: str, language: str, page_id: int, revision_id: int) -> str:
    """Build the expected article_id for consistency checking."""
    return f"{wikidata}:{language}:{page_id}:{revision_id}"


def wikipedia_document_from_article_row(row: Mapping[str, object]) -> WikipediaDocument:
    """Convert an article row to a WikipediaDocument.

    Pure function: no I/O, no clock, no mutation of input. Preserves all
    30 article values exactly as-is, sets ``project = "wikipedia"``,
    computes ``document_id``, and validates every field strictly.

    **Strict validation policy:**

    - String fields must already be ``str``.
    - Required integer fields must already be ``int`` (not ``bool``).
    - Nullable ``thumbnail_width`` / ``thumbnail_height`` must be
      ``int | None`` (not ``bool``).
    - Unknown extra keys are rejected to prevent silently dropping data.
    - Identity fields are validated: valid QID, non-empty language without
      ``:`` or whitespace, positive ``page_id`` and ``revision_id``,
      consistent ``article_id``.

    Parameters
    ----------
    row:
        A mapping with exactly the 30 ``ARTICLE_COLUMNS`` keys.

    Returns
    -------
    WikipediaDocument
        Frozen, slotted model instance.

    Raises
    ------
    WikipediaDocumentConversionError
        On missing fields, extra keys, type mismatches, null identity
        components, or inconsistent ``article_id``.
    """
    article_cols_set = set(ARTICLE_COLUMNS)

    # Validate no unknown keys
    extra_keys = sorted(set(row.keys()) - article_cols_set)
    if extra_keys:
        raise WikipediaDocumentConversionError(
            f"Unknown extra key(s) in article row: {extra_keys}. "
            f"Only ARTICLE_COLUMNS keys are accepted."
        )

    # Validate all required fields are present
    missing = sorted(article_cols_set - set(row.keys()))
    if missing:
        raise WikipediaDocumentConversionError(
            f"Missing required field(s) in article row: {missing}"
        )

    # Validate and bind identity fields strictly
    wikidata = _validate_qid(row["wikidata"], "wikidata")
    language = _validate_language(row["language"], "language")
    page_id = _validate_positive_int(row["page_id"], "page_id")
    revision_id = _validate_positive_int(row["revision_id"], "revision_id")

    # Validate and bind article_id consistency
    article_id = _require_exact_str(row["article_id"], "article_id")
    expected_article_id = _make_article_id(wikidata, language, page_id, revision_id)
    if article_id != expected_article_id:
        raise WikipediaDocumentConversionError(
            f"Inconsistent article_id: expected '{expected_article_id}', got '{article_id}'"
        )

    # Validate and bind remaining fields to correctly typed locals (no type ignores)
    site = _require_exact_str(row["site"], "site")
    title = _require_exact_str(row["title"], "title")
    url = _require_exact_str(row["url"], "url")
    revision_timestamp = _require_exact_str(row["revision_timestamp"], "revision_timestamp")
    retrieved_at = _require_exact_str(row["retrieved_at"], "retrieved_at")
    wikidata_label = _require_exact_str(row["wikidata_label"], "wikidata_label")
    wikidata_description = _require_exact_str(row["wikidata_description"], "wikidata_description")
    wikidata_aliases = _require_exact_str(row["wikidata_aliases"], "wikidata_aliases")
    lead_text = _require_exact_str(row["lead_text"], "lead_text")
    extract = _require_exact_str(row["extract"], "extract")
    full_text = _require_exact_str(row["full_text"], "full_text")
    full_text_format = _require_exact_str(row["full_text_format"], "full_text_format")
    article_length_chars = _require_exact_int(row["article_length_chars"], "article_length_chars")
    article_length_words = _require_exact_int(row["article_length_words"], "article_length_words")
    article_length_tokens_estimate = _require_exact_int(
        row["article_length_tokens_estimate"], "article_length_tokens_estimate"
    )
    thumbnail_url = _require_exact_str(row["thumbnail_url"], "thumbnail_url")
    thumbnail_width = _require_optional_int(row["thumbnail_width"], "thumbnail_width")
    thumbnail_height = _require_optional_int(row["thumbnail_height"], "thumbnail_height")
    categories = _require_exact_str(row["categories"], "categories")
    license = _require_exact_str(row["license"], "license")
    attribution = _require_exact_str(row["attribution"], "attribution")
    source_api = _require_exact_str(row["source_api"], "source_api")
    fetch_status = _require_exact_str(row["fetch_status"], "fetch_status")
    fetch_error = _require_exact_str(row["fetch_error"], "fetch_error")
    content_hash = _require_exact_str(row["content_hash"], "content_hash")

    document_id = _make_document_id(wikidata, language, page_id, revision_id)

    return WikipediaDocument(
        document_id=document_id,
        article_id=article_id,
        wikidata=wikidata,
        project="wikipedia",
        language=language,
        site=site,
        title=title,
        url=url,
        page_id=page_id,
        revision_id=revision_id,
        revision_timestamp=revision_timestamp,
        retrieved_at=retrieved_at,
        wikidata_label=wikidata_label,
        wikidata_description=wikidata_description,
        wikidata_aliases=wikidata_aliases,
        lead_text=lead_text,
        extract=extract,
        full_text=full_text,
        full_text_format=full_text_format,
        article_length_chars=article_length_chars,
        article_length_words=article_length_words,
        article_length_tokens_estimate=article_length_tokens_estimate,
        thumbnail_url=thumbnail_url,
        thumbnail_width=thumbnail_width,
        thumbnail_height=thumbnail_height,
        categories=categories,
        license=license,
        attribution=attribution,
        source_api=source_api,
        fetch_status=fetch_status,
        fetch_error=fetch_error,
        content_hash=content_hash,
    )


# ---------------------------------------------------------------------------
# Table Conversion
# ---------------------------------------------------------------------------


def build_wikipedia_document_table(article_table: pa.Table) -> pa.Table:
    """Convert a PyArrow article table to a canonical Wikipedia document table.

    **Strict input validation:**

    - Rejects missing columns, wrong types, unknown extra columns,
      duplicate column names, and incorrect field ordering.
    - Rejects any mismatch in column description metadata.
    - The input must be exactly the established ``article_schema()`` layout.

    Parameters
    ----------
    article_table:
        Input table with the ``article_schema()`` layout.

    Returns
    -------
    pa.Table
        Output table with the ``wikipedia_document_schema()`` layout,
        sorted deterministically by ``document_id``.

    Raises
    ------
    WikipediaDocumentConversionError
        On schema/metadata mismatches, duplicate IDs, or conversion failures.
    """
    expected = article_schema()
    _validate_article_table_schema(article_table, expected)
    if article_table.num_rows == 0:
        return _empty_wikipedia_document_table()
    _validate_unique_article_ids(article_table)
    docs = _convert_article_rows(article_table.to_pylist())
    return _sort_documents(pa.Table.from_pylist(docs, schema=wikipedia_document_schema()))


def _validate_article_table_schema(article_table: pa.Table, expected: pa.Schema) -> None:
    actual = article_table.schema
    _validate_column_names(actual, expected)
    _validate_column_order(actual, expected)
    _validate_column_fields(actual, expected)


def _validate_column_names(actual: pa.Schema, expected: pa.Schema) -> None:
    _validate_no_duplicate_columns(actual)
    _validate_expected_columns(actual, expected)


def _validate_no_duplicate_columns(actual: pa.Schema) -> None:
    name_counts = Counter(actual.names)
    duplicated = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicated:
        raise WikipediaDocumentConversionError(
            f"Duplicate column name(s) in input table: {duplicated}"
        )


def _validate_expected_columns(actual: pa.Schema, expected: pa.Schema) -> None:
    expected_names = set(expected.names)
    actual_names = set(actual.names)
    extra = sorted(actual_names - expected_names)
    if extra:
        raise WikipediaDocumentConversionError(f"Unknown extra column(s) in input table: {extra}")
    missing = sorted(expected_names - actual_names)
    if missing:
        raise WikipediaDocumentConversionError(
            f"Missing required column(s) in input table: {missing}"
        )


def _validate_column_order(actual: pa.Schema, expected: pa.Schema) -> None:
    if tuple(actual.names) != tuple(expected.names):
        raise WikipediaDocumentConversionError(
            f"Input table column order does not match article_schema(). "
            f"Expected {list(expected.names)}, got {list(actual.names)}"
        )


def _validate_column_fields(actual: pa.Schema, expected: pa.Schema) -> None:
    for index, expected_field in enumerate(expected):
        actual_field = actual.field(index)
        _validate_column_type(actual_field, expected_field)
        _validate_column_metadata(actual_field, expected_field)


def _validate_column_type(actual: pa.Field, expected: pa.Field) -> None:
    if actual.type != expected.type:
        raise WikipediaDocumentConversionError(
            f"Type mismatch for column '{expected.name}': "
            f"expected {expected.type}, got {actual.type}"
        )


def _validate_column_metadata(actual: pa.Field, expected: pa.Field) -> None:
    if actual.metadata != expected.metadata:
        raise WikipediaDocumentConversionError(
            f"Metadata mismatch for column '{expected.name}': "
            f"expected {expected.metadata}, got {actual.metadata}"
        )


def _empty_wikipedia_document_table() -> pa.Table:
    schema = wikipedia_document_schema()
    return pa.table(
        {col: pa.array([], type=schema.field(col).type) for col in WIKIPEDIA_DOCUMENT_COLUMNS},
        schema=schema,
    )


def _validate_unique_article_ids(article_table: pa.Table) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for article_id in article_table.column("article_id").to_pylist():
        if article_id in seen:
            duplicates.add(article_id)
        seen.add(article_id)
    if duplicates:
        raise WikipediaDocumentConversionError(
            f"Duplicate article_id values found: {sorted(duplicates)}"
        )


def _convert_article_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    doc_id_set: set[str] = set()
    for row in rows:
        doc = wikipedia_document_from_article_row(row)
        _validate_unique_document_id(doc.document_id, doc_id_set)
        doc_id_set.add(doc.document_id)
        docs.append(doc.to_dict())
    return docs


def _validate_unique_document_id(document_id: str, seen: set[str]) -> None:
    if document_id in seen:
        raise WikipediaDocumentConversionError(f"Duplicate document_id generated: '{document_id}'")


def _sort_documents(table: pa.Table) -> pa.Table:
    return table.sort_by([("document_id", "ascending")])


__all__ = [
    "WIKIPEDIA_DOCUMENT_COLUMNS",
    "WIKIPEDIA_DOCUMENT_DESCRIPTIONS",
    "WikipediaDocument",
    "WikipediaDocumentConversionError",
    "build_wikipedia_document_table",
    "wikipedia_document_from_article_row",
    "wikipedia_document_schema",
]
