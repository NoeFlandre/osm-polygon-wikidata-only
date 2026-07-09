"""Domain dataclasses for the multi-table dataset.

Three tables are produced per PBF:

* :class:`Polygon` — one row per retained OSM polygon.
* :class:`Article` — one row per (wikidata, language, page_id, revision).
* :class:`PolygonArticleLink` — one row per (polygon, article) join.

The dataclasses are deliberately flat (no nested dicts) so they map
1:1 to parquet columns. JSON-encoded list/object fields use
deterministic serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ids import article_id, polygon_id


@dataclass(frozen=True)
class Polygon:
    """One retained OSM polygon with analysis-friendly metadata."""

    polygon_id: str
    region: str
    source_pbf: str
    osm_type: str  # "way" or "relation"
    osm_id: int
    wikidata: str
    name: str
    tags: str  # deterministic JSON of OSM tags minus wikidata
    tag_keys: str  # deterministic JSON list of sorted tag keys
    tag_count: int
    osm_primary_tag: str
    centroid: str  # GeoJSON Point string
    lat: float
    lon: float
    bbox: str  # deterministic JSON list [min_lon, min_lat, max_lon, max_lat]
    area_m2: float
    area_km2: float
    area_bucket: str
    geometry: str = ""  # deterministic GeoJSON Polygon/MultiPolygon string

    # Wikipedia coverage (filled during enrichment).
    has_name: bool = False
    has_wikidata: bool = True
    has_wikipedia: bool = False
    wikipedia_language_count: int = 0
    wikipedia_languages: str = "[]"  # deterministic JSON list
    wikipedia_article_count: int = 0
    has_english_wikipedia: bool = False
    has_french_wikipedia: bool = False
    text_available: bool = False
    best_language: str = ""

    extraction_version: str = ""
    extracted_at: str = ""

    @staticmethod
    def make(
        *,
        source_pbf_stem: str,
        region: str,
        source_pbf: str,
        osm_type: str,
        osm_id: int,
        wikidata: str,
        name: str,
        tags: str,
        tag_keys: str,
        tag_count: int,
        osm_primary_tag: str,
        centroid: str,
        lat: float,
        lon: float,
        bbox: str,
        geometry: str = "",
        area_m2: float,
        area_km2: float,
        area_bucket: str,
        has_name: bool,
        has_wikidata: bool,
        extraction_version: str,
        extracted_at: str,
    ) -> Polygon:
        return Polygon(
            polygon_id=polygon_id(source_pbf_stem, osm_type, osm_id),
            region=region,
            source_pbf=source_pbf,
            osm_type=osm_type,
            osm_id=osm_id,
            wikidata=wikidata,
            name=name,
            tags=tags,
            tag_keys=tag_keys,
            tag_count=tag_count,
            osm_primary_tag=osm_primary_tag,
            centroid=centroid,
            lat=lat,
            lon=lon,
            bbox=bbox,
            geometry=geometry,
            area_m2=area_m2,
            area_km2=area_km2,
            area_bucket=area_bucket,
            has_name=has_name,
            has_wikidata=has_wikidata,
            extraction_version=extraction_version,
            extracted_at=extracted_at,
        )


@dataclass(frozen=True)
class Article:
    """One Wikipedia article fetched for a Wikidata item."""

    article_id: str
    wikidata: str
    language: str
    site: str  # e.g. "enwiki"
    title: str
    url: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    retrieved_at: str
    wikidata_label: str
    wikidata_description: str
    wikidata_aliases: str  # deterministic JSON list
    lead_text: str
    extract: str
    full_text: str
    full_text_format: str  # e.g. "plain_text"
    article_length_chars: int
    article_length_words: int
    article_length_tokens_estimate: int
    thumbnail_url: str
    thumbnail_width: int | None
    thumbnail_height: int | None
    categories: str  # deterministic JSON list
    license: str
    attribution: str
    source_api: str  # "wikidata" or "mediawiki_action_api" or "wikipedia_rest_api"
    fetch_status: str
    fetch_error: str
    content_hash: str

    @staticmethod
    def make(
        *,
        wikidata: str,
        language: str,
        page_id: int,
        revision_id: int,
    ) -> str:
        return article_id(wikidata, language, page_id, revision_id)


@dataclass(frozen=True)
class PolygonArticleLink:
    """One link between a polygon and an article row."""

    polygon_id: str
    article_id: str
    wikidata: str
    language: str
    source_pbf: str
    region: str
    osm_type: str
    osm_id: int
    page_id: int
    revision_id: int
    is_best_language: bool


@dataclass
class ManifestStats:
    """Per-PBF statistics for the manifest."""

    polygon_count: int = 0
    unique_wikidata_count: int = 0
    article_count: int = 0
    language_count: int = 0
    languages: list[str] = field(default_factory=list)
    rows_with_wikipedia: int = 0
    rows_with_full_text: int = 0
    total_full_text_chars: int = 0
    area_bucket_counts: dict[str, int] = field(default_factory=dict)
    top_tag_keys: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon_count": self.polygon_count,
            "unique_wikidata_count": self.unique_wikidata_count,
            "article_count": self.article_count,
            "language_count": self.language_count,
            "languages": sorted(self.languages),
            "rows_with_wikipedia": self.rows_with_wikipedia,
            "rows_with_full_text": self.rows_with_full_text,
            "total_full_text_chars": self.total_full_text_chars,
            "area_bucket_counts": dict(sorted(self.area_bucket_counts.items())),
            "top_tag_keys": dict(sorted(self.top_tag_keys.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
