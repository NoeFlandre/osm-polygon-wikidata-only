"""Stable PyArrow schemas for augmentation tables."""

from __future__ import annotations

import pyarrow as pa

DOCUMENT_COLUMNS = (
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
    "full_text",
    "full_text_format",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
    "license",
    "attribution",
    "source_api",
    "fetch_status",
    "fetch_error",
    "content_hash",
)
SECTION_COLUMNS = (
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
    "text",
    "text_length_chars",
    "text_length_words",
    "text_length_tokens_estimate",
    "content_hash",
    "license",
    "attribution",
)
FACT_COLUMNS = (
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
    "numeric_value",
    "unit_entity_id",
    "rank",
    "qualifiers",
    "references",
    "retrieved_at",
    "source_api",
)


def _schema(columns: tuple[str, ...]) -> pa.Schema:
    integers = {
        "page_id",
        "revision_id",
        "section_index",
        "level",
        "article_length_chars",
        "article_length_words",
        "article_length_tokens_estimate",
        "text_length_chars",
        "text_length_words",
        "text_length_tokens_estimate",
    }
    return pa.schema(
        [
            pa.field(
                column,
                pa.int64()
                if column in integers
                else pa.float64()
                if column == "numeric_value"
                else pa.string(),
            )
            for column in columns
        ]
    )


def document_schema() -> pa.Schema:
    return _schema(DOCUMENT_COLUMNS)


def section_schema() -> pa.Schema:
    return _schema(SECTION_COLUMNS)


def fact_schema() -> pa.Schema:
    return _schema(FACT_COLUMNS)


__all__ = [
    "DOCUMENT_COLUMNS",
    "FACT_COLUMNS",
    "SECTION_COLUMNS",
    "document_schema",
    "fact_schema",
    "section_schema",
]
