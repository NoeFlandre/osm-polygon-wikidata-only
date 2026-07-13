"""Schema freezing tests.

Asserts the documented Parquet column ordering, types, and field
names for every public table. Uses in-memory data only — no fixture
Parquet files are required for the column list / type assertions.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
    article_schema,
    polygon_article_schema,
    polygon_schema,
)


def _column_types(schema: pa.Schema) -> dict[str, str]:
    return {field.name: str(field.type) for field in schema}


def test_polygon_schema_columns() -> None:
    assert tuple(field.name for field in polygon_schema()) == POLYGON_COLUMNS
    assert set(POLYGON_COLUMNS) == set(POLYGON_DESCRIPTIONS)


def test_article_schema_columns() -> None:
    assert tuple(field.name for field in article_schema()) == ARTICLE_COLUMNS
    assert set(ARTICLE_COLUMNS) == set(ARTICLE_DESCRIPTIONS)


def test_polygon_article_schema_columns() -> None:
    assert tuple(field.name for field in polygon_article_schema()) == POLYGON_ARTICLE_COLUMNS
    assert set(POLYGON_ARTICLE_COLUMNS) == set(POLYGON_ARTICLE_DESCRIPTIONS)


def test_polygon_schema_field_types() -> None:
    types = _column_types(polygon_schema())
    assert types["polygon_id"] == "string"
    assert types["lat"] == "double"
    assert types["lon"] == "double"
    assert types["has_wikipedia"] == "bool"
    assert types["wikipedia_language_count"] == "int64"
    assert types["area_m2"] == "double"


def test_article_schema_field_types() -> None:
    types = _column_types(article_schema())
    assert types["article_id"] == "string"
    assert types["page_id"] == "int64"
    assert types["full_text"] == "string"
    assert types["thumbnail_width"] == "int64"
    assert types["thumbnail_height"] == "int64"


def test_polygon_article_schema_field_types() -> None:
    types = _column_types(polygon_article_schema())
    assert types["is_best_language"] == "bool"
    assert types["page_id"] == "int64"
    assert types["revision_id"] == "int64"


def test_document_schema_columns() -> None:
    assert tuple(field.name for field in document_schema()) == DOCUMENT_COLUMNS


def test_section_schema_columns() -> None:
    assert tuple(field.name for field in section_schema()) == SECTION_COLUMNS


def test_fact_schema_columns() -> None:
    assert tuple(field.name for field in fact_schema()) == FACT_COLUMNS


def test_section_schema_field_types() -> None:
    types = _column_types(section_schema())
    assert types["section_index"] == "int64"
    assert types["level"] == "int64"
    assert types["text_length_words"] == "int64"


def test_fact_schema_field_types() -> None:
    types = _column_types(fact_schema())
    assert types["numeric_value"] == "double"
    assert types["rank"] == "string"
    assert types["value_entity_id"] == "string"
