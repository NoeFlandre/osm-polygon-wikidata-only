"""Polygon extraction: PBF -> :class:`Polygon` rows.

This module is the bridge between :mod:`io.pbf_reader` (which yields
raw polygon candidates from osmium) and the :class:`Polygon` domain
model. It computes geometry, analysis metadata, and stable IDs but
does not perform any HTTP work.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from osm_polygon_wikidata_only import __version__
from osm_polygon_wikidata_only.domain.analysis import area_bucket, bbox_from_geom, osm_primary_tag
from osm_polygon_wikidata_only.domain.geometry import (
    GeometryError,
    PolygonGeometry,
    centroid_geojson,
    compute_polygon_geometry,
)
from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.io.pbf_reader import PolygonCandidate
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso

LOGGER = logging.getLogger(__name__)


def _parse_geom(geom_json: str) -> dict[str, object] | None:
    try:
        parsed: object = json.loads(geom_json)
    except json.JSONDecodeError as e:
        LOGGER.debug("Skipping element with invalid GeoJSON: %s", e)
        return None
    return parsed if isinstance(parsed, dict) else None


def _compute_geom(geom_json: str) -> tuple[PolygonGeometry, dict[str, object]] | None:
    geom = _parse_geom(geom_json)
    if geom is None:
        return None
    try:
        pg = compute_polygon_geometry(geom)
    except GeometryError as e:
        LOGGER.debug("Skipping element with invalid geometry: %s", e)
        return None
    return pg, geom


def _row_dict(polygon: Polygon) -> dict[str, Any]:
    return dict(polygon.__dict__)


def candidate_to_polygon(
    candidate: PolygonCandidate,
    *,
    source_pbf_stem: str,
    region: str,
    source_pbf: str,
    extracted_at: str | None = None,
) -> Polygon | None:
    """Convert one osmium polygon candidate to a :class:`Polygon`.

    Returns ``None`` if geometry cannot be computed. Tags, primary
    tag, bbox, area, and bucket are all derived here. Wikipedia
    coverage fields are left at their defaults and filled in by the
    enrichment step later.
    """
    osm_type, osm_id, tags, geom_json = candidate
    computed = _compute_geom(geom_json)
    if computed is None:
        return None
    pg, geom = computed

    wikidata = tags.get("wikidata", "").strip()
    if not wikidata:
        # Defensive: the reader already filters, but the extractor
        # should also enforce this in case it is reused.
        return None

    cleaned_tags = {k: v for k, v in tags.items() if k != "wikidata"}
    name = cleaned_tags.get("name", "")
    bbox = bbox_from_geom(geom)

    return Polygon.make(
        source_pbf_stem=source_pbf_stem,
        region=region,
        source_pbf=source_pbf,
        osm_type=osm_type,
        osm_id=osm_id,
        wikidata=wikidata,
        name=name,
        tags=json_dumps(cleaned_tags),
        tag_keys=json_dumps(sorted(cleaned_tags.keys())),
        tag_count=len(cleaned_tags),
        osm_primary_tag=osm_primary_tag(cleaned_tags),
        centroid=centroid_geojson(pg.lon, pg.lat),
        lat=pg.lat,
        lon=pg.lon,
        bbox=json_dumps(bbox),
        geometry=json_dumps(geom),
        area_m2=pg.area_m2,
        area_km2=pg.area_m2 / 1_000_000.0,
        area_bucket=area_bucket(pg.area_m2),
        has_name=bool(name),
        has_wikidata=True,
        extraction_version=__version__,
        extracted_at=extracted_at or utc_now_iso(),
    )


def polygon_to_dict(p: Polygon) -> dict[str, Any]:
    """Convert a :class:`Polygon` to the row dict used by :mod:`io.parquet`."""
    return _row_dict(p)


__all__ = ["candidate_to_polygon", "polygon_to_dict"]
