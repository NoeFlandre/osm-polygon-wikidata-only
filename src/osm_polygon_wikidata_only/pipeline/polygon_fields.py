"""Shared field construction for admitted V1 and V2 polygon candidates.

Admission and output schemas belong to the extractors. This adapter only
assembles their common metadata from already validated geometry and raw tags.
"""

from __future__ import annotations

from typing import Any, TypedDict

from osm_polygon_wikidata_only import VERSION
from osm_polygon_wikidata_only.domain.analysis import area_bucket, bbox_from_geom, osm_primary_tag
from osm_polygon_wikidata_only.domain.geometry import PolygonGeometry, centroid_geojson
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso


class CommonPolygonFields(TypedDict):
    """Fields shared by ``Polygon.make`` and the V2 row dictionary."""

    region: str
    source_pbf: str
    osm_type: str
    osm_id: int
    name: str
    tags: str
    tag_keys: str
    tag_count: int
    osm_primary_tag: str
    centroid: str
    lat: float
    lon: float
    bbox: str
    geometry: str
    area_m2: float
    area_km2: float
    area_bucket: str
    has_name: bool
    extraction_version: str
    extracted_at: str


def build_common_polygon_fields(
    *,
    osm_type: str,
    osm_id: int,
    tags: dict[str, str],
    geom: dict[str, Any],
    computed: PolygonGeometry,
    region: str,
    source_pbf: str,
    extracted_at: str | None,
) -> CommonPolygonFields:
    """Build common fields without filtering candidates or changing their tags."""
    cleaned_tags = {key: value for key, value in tags.items() if key != "wikidata"}
    name = cleaned_tags.get("name", "")
    return {
        "region": region,
        "source_pbf": source_pbf,
        "osm_type": osm_type,
        "osm_id": osm_id,
        "name": name,
        "tags": json_dumps(cleaned_tags),
        "tag_keys": json_dumps(sorted(cleaned_tags)),
        "tag_count": len(cleaned_tags),
        "osm_primary_tag": osm_primary_tag(cleaned_tags),
        "centroid": centroid_geojson(computed.lon, computed.lat),
        "lat": computed.lat,
        "lon": computed.lon,
        "bbox": json_dumps(bbox_from_geom(geom)),
        "geometry": json_dumps(geom),
        "area_m2": computed.area_m2,
        "area_km2": computed.area_m2 / 1_000_000.0,
        "area_bucket": area_bucket(computed.area_m2),
        "has_name": bool(name),
        "extraction_version": VERSION,
        "extracted_at": extracted_at or utc_now_iso(),
    }
