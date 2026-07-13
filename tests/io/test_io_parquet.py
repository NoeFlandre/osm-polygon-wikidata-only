"""Tests for io.parquet."""

from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
)
from osm_polygon_wikidata_only.io.parquet import (
    read_table,
    write_articles,
    write_polygon_articles,
    write_polygons,
    write_table,
)


def _sample_polygon() -> dict:
    return {
        "polygon_id": "monaco-latest:way:1",
        "region": "monaco",
        "source_pbf": "monaco-latest.osm.pbf",
        "osm_type": "way",
        "osm_id": 1,
        "wikidata": "Q235",
        "name": "Monaco",
        "tags": json.dumps({"name": "Monaco"}, sort_keys=True),
        "tag_keys": json.dumps(["name"]),
        "tag_count": 1,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [7.42, 43.73]}),
        "lat": 43.73,
        "lon": 7.42,
        "bbox": json.dumps([7.42, 43.73, 7.43, 43.74]),
        "area_m2": 1_000.0,
        "area_km2": 0.001,
        "area_bucket": "100m2-1k_m2",
        "has_name": True,
        "has_wikidata": True,
        "has_wikipedia": False,
        "wikipedia_language_count": 0,
        "wikipedia_languages": "[]",
        "wikipedia_article_count": 0,
        "has_english_wikipedia": False,
        "has_french_wikipedia": False,
        "text_available": False,
        "best_language": "",
        "extraction_version": "0.1.0",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def test_write_polygons_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "polygons" / "monaco.parquet"
    n = write_polygons(out, [_sample_polygon()])
    assert n == 1
    table = read_table(out)
    assert table.num_rows == 1
    assert table.column("wikidata").to_pylist() == ["Q235"]


def test_write_table_handles_empty_input(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.domain.schema import polygon_schema

    out = tmp_path / "empty.parquet"
    n = write_table(out, [], columns=POLYGON_COLUMNS, schema=polygon_schema())
    assert n == 0
    table = read_table(out)
    assert table.num_rows == 0
    # Schema columns are preserved on an empty file.
    assert [f.name for f in table.schema] == list(POLYGON_COLUMNS)


def test_write_polygon_articles_writes_proper_schema(tmp_path: Path) -> None:
    out = tmp_path / "links.parquet"
    rows = [
        {
            "polygon_id": "monaco-latest:way:1",
            "article_id": "Q235:en:1:1",
            "wikidata": "Q235",
            "language": "en",
            "source_pbf": "monaco-latest.osm.pbf",
            "region": "monaco",
            "osm_type": "way",
            "osm_id": 1,
            "page_id": 1,
            "revision_id": 1,
            "is_best_language": True,
        }
    ]
    n = write_polygon_articles(out, rows)
    assert n == 1
    table = read_table(out)
    assert set(table.column_names) == set(POLYGON_ARTICLE_COLUMNS)


def test_write_articles_handles_optional_ints(tmp_path: Path) -> None:
    out = tmp_path / "articles.parquet"
    rows = [
        {
            "article_id": "Q1:en:1:1",
            "wikidata": "Q1",
            "language": "en",
            "site": "enwiki",
            "title": "T",
            "url": "https://en.wikipedia.org/wiki/T",
            "page_id": 1,
            "revision_id": 1,
            "revision_timestamp": "2026-01-01T00:00:00Z",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "wikidata_label": "T",
            "wikidata_description": "",
            "wikidata_aliases": "[]",
            "lead_text": "",
            "extract": "",
            "full_text": "hello world",
            "full_text_format": "plain_text",
            "article_length_chars": 11,
            "article_length_words": 2,
            "article_length_tokens_estimate": 2,
            "thumbnail_url": "",
            "thumbnail_width": None,
            "thumbnail_height": None,
            "categories": "[]",
            "license": "CC BY-SA",
            "attribution": "Wikipedia",
            "source_api": "mediawiki_action_api",
            "fetch_status": "ok",
            "fetch_error": "",
            "content_hash": "deadbeef",
        }
    ]
    n = write_articles(out, rows)
    assert n == 1
    table = read_table(out)
    assert table.column("full_text").to_pylist() == ["hello world"]
    assert table.column("thumbnail_width").to_pylist() == [None]
