"""Dataset schema: column lists, parquet schemas, and helpers.

The single source of truth for what each of the three Parquet tables
contains. Used by:

* :mod:`io.parquet` to write tables with the right column types;
* :mod:`hf.dataset_card` to render the card;
* unit tests to assert the schema is complete.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

# Order matters: parquet columns appear in this order.
POLYGON_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "region",
    "source_pbf",
    "osm_type",
    "osm_id",
    "wikidata",
    "name",
    "tags",
    "tag_keys",
    "tag_count",
    "osm_primary_tag",
    "centroid",
    "lat",
    "lon",
    "bbox",
    "area_m2",
    "area_km2",
    "area_bucket",
    "has_name",
    "has_wikidata",
    "has_wikipedia",
    "wikipedia_language_count",
    "wikipedia_languages",
    "wikipedia_article_count",
    "has_english_wikipedia",
    "has_french_wikipedia",
    "text_available",
    "best_language",
    "extraction_version",
    "extracted_at",
)


ARTICLE_COLUMNS: tuple[str, ...] = (
    "article_id",
    "wikidata",
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


POLYGON_ARTICLE_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "article_id",
    "wikidata",
    "language",
    "source_pbf",
    "region",
    "osm_type",
    "osm_id",
    "page_id",
    "revision_id",
    "is_best_language",
)


# Column descriptions used by the dataset card and module docstrings.
# Order matches the column lists above.
POLYGON_DESCRIPTIONS: dict[str, str] = {
    "polygon_id": "Deterministic ID: `<source_pbf_stem>:<osm_type>:<osm_id>`.",
    "region": "Geofabrik region slug parsed from the source PBF filename (e.g. `monaco`).",
    "source_pbf": "Source PBF filename (e.g. `monaco-latest.osm.pbf`).",
    "osm_type": "OSM element type: `way` or `relation`.",
    "osm_id": "OpenStreetMap numeric identifier of the element.",
    "wikidata": "Wikidata Q-id from the OSM `wikidata=*` tag.",
    "name": "Convenience: `tags.name` if present, empty string otherwise.",
    "tags": "Deterministic JSON object of all OSM tags except `wikidata`.",
    "tag_keys": "Deterministic JSON list of sorted OSM tag keys.",
    "tag_count": "Number of OSM tags (excluding `wikidata`).",
    "osm_primary_tag": "Best single tag for coarse analysis, e.g. `landuse=forest`.",
    "centroid": "Polygon centroid as a GeoJSON Point string (`[lon, lat]`).",
    "lat": "Centroid latitude in decimal degrees (WGS84).",
    "lon": "Centroid longitude in decimal degrees (WGS84).",
    "bbox": "Bounding box as JSON list `[min_lon, min_lat, max_lon, max_lat]`.",
    "area_m2": "Polygon area in square meters (WGS84 equirectangular approximation).",
    "area_km2": "Polygon area in square kilometers.",
    "area_bucket": "Human-readable size bucket (e.g. `1-10km2`).",
    "has_name": "True if the polygon has a `name` tag.",
    "has_wikidata": "True if the polygon has a `wikidata` tag (always true by filter).",
    "has_wikipedia": "True if at least one linked Wikipedia article was fetched.",
    "wikipedia_language_count": "Number of Wikipedia languages for which articles were linked.",
    "wikipedia_languages": 'Deterministic JSON list of language codes (e.g. `["en","fr"]`).',
    "wikipedia_article_count": "Number of unique article revisions linked to this polygon.",
    "has_english_wikipedia": "True if `en` is among the available languages.",
    "has_french_wikipedia": "True if `fr` is among the available languages.",
    "text_available": "True if at least one linked article has non-empty full text.",
    "best_language": "Deterministic preferred language code (e.g. `en`).",
    "extraction_version": "Package version that produced the row.",
    "extracted_at": "ISO-8601 UTC timestamp at the moment the row was extracted.",
}


ARTICLE_DESCRIPTIONS: dict[str, str] = {
    "article_id": "Deterministic ID: `<wikidata>:<language>:<page_id>:<revision_id>`.",
    "wikidata": "Wikidata Q-id this article is linked to.",
    "language": "Wikipedia language code, e.g. `en`.",
    "site": "Wikidata sitelink site, e.g. `enwiki`.",
    "title": "Article title as returned by the Wikipedia API.",
    "url": "Canonical Wikipedia article URL.",
    "page_id": "MediaWiki page ID.",
    "revision_id": "Exact revision ID used for text extraction.",
    "revision_timestamp": "ISO-8601 timestamp of the revision.",
    "retrieved_at": "ISO-8601 UTC timestamp when this pipeline fetched the article.",
    "wikidata_label": "Best Wikidata label for the article's language, fallback English.",
    "wikidata_description": "Best Wikidata description for the article's language.",
    "wikidata_aliases": "Deterministic JSON list of Wikidata aliases.",
    "lead_text": "Lead section of the article, plain text.",
    "extract": "Short summary/extract if the API provided one.",
    "full_text": "Full cleaned article text, plain text.",
    "full_text_format": "Encoding of `full_text`; always `plain_text`.",
    "article_length_chars": "Length of `full_text` in characters.",
    "article_length_words": "Approximate whitespace-token count of `full_text`.",
    "article_length_tokens_estimate": "Rough token count: `chars / 4`.",
    "thumbnail_url": "Thumbnail URL only (no image bytes stored).",
    "thumbnail_width": "Thumbnail width in pixels, if known.",
    "thumbnail_height": "Thumbnail height in pixels, if known.",
    "categories": "Deterministic JSON list of category titles, or `[]`.",
    "license": "License string, e.g. `CC BY-SA`.",
    "attribution": "Attribution string for the article.",
    "source_api": "Which API was queried: `mediawiki_action_api` / `wikipedia_rest_api`.",
    "fetch_status": "One of: `ok`, `article_not_found`, `http_error`, `rate_limited`, `parse_error`, `empty_text`.",
    "fetch_error": "Short diagnostic on failure, empty string on success.",
    "content_hash": "Stable SHA-256 of `full_text` for change tracking.",
}


POLYGON_ARTICLE_DESCRIPTIONS: dict[str, str] = {
    "polygon_id": "FK to `polygons.polygon_id`.",
    "article_id": "FK to `articles.article_id`.",
    "wikidata": "Wikidata Q-id (denormalized for fast filtering).",
    "language": "Wikipedia language code.",
    "source_pbf": "Source PBF filename (denormalized).",
    "region": "Geofabrik region slug.",
    "osm_type": "OSM element type.",
    "osm_id": "OSM numeric identifier.",
    "page_id": "MediaWiki page ID of the linked article.",
    "revision_id": "MediaWiki revision ID of the linked article.",
    "is_best_language": "True if this row's language matches the polygon's `best_language`.",
}


# Allowed values for ``Article.fetch_status``. Enforced as a literal
# type at the call site; declared here for documentation and tests.
FETCH_STATUSES: frozenset[str] = frozenset(
    {
        "ok",
        "invalid_qid",
        "wikidata_not_found",
        "no_wikipedia_sitelinks",
        "article_not_found",
        "http_error",
        "rate_limited",
        "parse_error",
        "empty_text",
    }
)


def _arrow_schema(columns: tuple[str, ...], descriptions: dict[str, str]) -> pa.schema:
    """Build a pyarrow schema with the conventional column types.

    All columns are stored as strings / numeric scalars because we want
    byte-stable, simple parquet files that any tool can read.
    """
    fields: list[pa.Field] = []
    for col in columns:
        if (
            col.endswith("_count")
            or col.endswith("_chars")
            or col.endswith("_words")
            or col.endswith("_tokens_estimate")
        ) or col in {"page_id", "revision_id", "osm_id", "thumbnail_width", "thumbnail_height"}:
            dtype = pa.int64()
        elif col in {"lat", "lon", "area_m2", "area_km2"}:
            dtype = pa.float64()
        elif col.startswith("has_") or col == "is_best_language" or col == "text_available":
            dtype = pa.bool_()
        else:
            dtype = pa.string()
        fields.append(pa.field(col, dtype, metadata={"description": descriptions.get(col, "")}))
    return pa.schema(fields)


def polygon_schema() -> pa.schema:
    return _arrow_schema(POLYGON_COLUMNS, POLYGON_DESCRIPTIONS)


def article_schema() -> pa.schema:
    return _arrow_schema(ARTICLE_COLUMNS, ARTICLE_DESCRIPTIONS)


def polygon_article_schema() -> pa.schema:
    return _arrow_schema(POLYGON_ARTICLE_COLUMNS, POLYGON_ARTICLE_DESCRIPTIONS)


def empty_row(columns: tuple[str, ...]) -> dict[str, Any]:
    """Build a row of empty values for the given column set.

    Useful for building an empty parquet file with the correct schema
    when a PBF produces no rows.
    """
    row: dict[str, Any] = {}
    for col in columns:
        if (
            col.endswith("_count")
            or col.endswith("_chars")
            or col.endswith("_words")
            or col.endswith("_tokens_estimate")
        ) or col in {"page_id", "revision_id", "osm_id", "thumbnail_width", "thumbnail_height"}:
            row[col] = 0
        elif col in {"lat", "lon", "area_m2", "area_km2"}:
            row[col] = 0.0
        elif col.startswith("has_") or col == "is_best_language" or col == "text_available":
            row[col] = False
        else:
            row[col] = ""
    return row


__all__ = [
    "ARTICLE_COLUMNS",
    "ARTICLE_DESCRIPTIONS",
    "FETCH_STATUSES",
    "POLYGON_ARTICLE_COLUMNS",
    "POLYGON_ARTICLE_DESCRIPTIONS",
    "POLYGON_COLUMNS",
    "POLYGON_DESCRIPTIONS",
    "article_schema",
    "empty_row",
    "polygon_article_schema",
    "polygon_schema",
]
