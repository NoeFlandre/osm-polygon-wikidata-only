"""Tests for the new multi-table dataset schema."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    FETCH_STATUSES,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
    article_schema,
    empty_row,
    polygon_article_schema,
    polygon_schema,
)


def test_each_polygon_column_has_a_description() -> None:
    assert set(POLYGON_COLUMNS) == set(POLYGON_DESCRIPTIONS.keys())


def test_each_article_column_has_a_description() -> None:
    assert set(ARTICLE_COLUMNS) == set(ARTICLE_DESCRIPTIONS.keys())


def test_has_geometry() -> None:
    assert "geometry" in POLYGON_COLUMNS
    assert "geometry" in POLYGON_DESCRIPTIONS
    assert polygon_schema().field("geometry").type == pa.string()


def test_each_link_column_has_a_description() -> None:
    assert set(POLYGON_ARTICLE_COLUMNS) == set(POLYGON_ARTICLE_DESCRIPTIONS.keys())


def test_polygon_schema_columns() -> None:
    schema = polygon_schema()
    assert [f.name for f in schema] == list(POLYGON_COLUMNS)


def test_article_schema_columns() -> None:
    schema = article_schema()
    assert [f.name for f in schema] == list(ARTICLE_COLUMNS)


def test_link_schema_columns() -> None:
    schema = polygon_article_schema()
    assert [f.name for f in schema] == list(POLYGON_ARTICLE_COLUMNS)


def test_polygon_schema_uses_correct_types() -> None:
    schema = polygon_schema()
    by_name = {f.name: f.type for f in schema}
    assert by_name["lat"] == pa.float64()
    assert by_name["area_m2"] == pa.float64()
    assert by_name["has_wikipedia"] == pa.bool_()
    assert by_name["osm_id"] == pa.int64()
    assert by_name["wikidata"] == pa.string()


def test_article_schema_uses_correct_types() -> None:
    schema = article_schema()
    by_name = {f.name: f.type for f in schema}
    assert by_name["page_id"] == pa.int64()
    assert by_name["article_length_chars"] == pa.int64()
    assert by_name["full_text"] == pa.string()


def test_fetch_statuses_include_ok_and_failures() -> None:
    assert "ok" in FETCH_STATUSES
    assert "article_not_found" in FETCH_STATUSES
    assert "rate_limited" in FETCH_STATUSES
    assert "empty_text" in FETCH_STATUSES


def test_empty_row_for_polygons() -> None:
    row = empty_row(POLYGON_COLUMNS)
    assert row["lat"] == 0.0
    assert row["area_m2"] == 0.0
    assert row["has_wikipedia"] is False
    assert row["wikidata"] == ""
    assert len(row) == len(POLYGON_COLUMNS)


def test_empty_row_for_articles() -> None:
    row = empty_row(ARTICLE_COLUMNS)
    assert row["page_id"] == 0
    assert row["full_text"] == ""
    assert row["fetch_status"] == ""
