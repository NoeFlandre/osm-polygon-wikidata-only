"""V2 polygon discovery with an explicit Wikipedia-tag opt-in.

V1 extraction remains Wikidata-only.  This module owns the additional
discovery rule used by V2: keep a polygon when its ``wikidata=*`` value
contains valid QIDs or its Wikipedia tags contain at least one valid
reference.  It produces plain row dictionaries so the V2 nullable
Wikidata contract never leaks into the V1 domain model.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only import __version__
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.analysis import area_bucket, bbox_from_geom, osm_primary_tag
from osm_polygon_wikidata_only.domain.geometry import (
    GeometryError,
    centroid_geojson,
    compute_polygon_geometry,
)
from osm_polygon_wikidata_only.domain.ids import polygon_id
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.io.pbf_reader import PBFReader, PolygonCandidate
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso
from osm_polygon_wikidata_only.v2.wikipedia_tags import parse_wikipedia_tags

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class V2PbfStem:
    """A source PBF path and its stable V2 stem."""

    path: Path
    stem: str
    region: str

    @classmethod
    def from_path(cls, path: Path) -> V2PbfStem:
        name = path.name
        stem = name.removesuffix(".osm.pbf")
        region = stem.removesuffix("-latest")
        return cls(path, stem, region)


@dataclass(frozen=True, slots=True)
class V2ExtractedPbf:
    """Immutable V2 polygon rows produced from one PBF."""

    stem: V2PbfStem
    polygons: tuple[dict[str, Any], ...]
    extraction_duration_s: float


def _geometry(geom_json: str) -> tuple[dict[str, Any], Any] | None:
    try:
        raw = json.loads(geom_json)
        if not isinstance(raw, dict):
            return None
        computed = compute_polygon_geometry(raw)
    except (GeometryError, json.JSONDecodeError):
        return None
    return raw, computed


def candidate_to_v2_row(
    candidate: PolygonCandidate,
    *,
    source_pbf_stem: str,
    region: str,
    source_pbf: str,
    extracted_at: str | None = None,
) -> dict[str, Any] | None:
    """Convert one retained candidate to a V2 polygon row."""
    osm_type, osm_id, tags, geom_json = candidate
    qids = qids_from_osm_tag(tags.get("wikidata", ""))
    refs, rejections = parse_wikipedia_tags(tags)
    if not qids and not refs:
        return None
    parsed = _geometry(geom_json)
    if parsed is None:
        return None
    geom, computed = parsed
    lat, lon = computed.lat, computed.lon
    cleaned_tags = {key: value for key, value in tags.items() if key != "wikidata"}
    name = cleaned_tags.get("name", "")
    bbox = bbox_from_geom(geom)
    row: dict[str, Any] = {
        "polygon_id": polygon_id(source_pbf_stem, osm_type, osm_id),
        "region": region,
        "source_pbf": source_pbf,
        "osm_type": osm_type,
        "osm_id": osm_id,
        "wikidata": ";".join(qids) if qids else None,
        "name": name,
        "tags": json_dumps(cleaned_tags),
        "tag_keys": json_dumps(sorted(cleaned_tags)),
        "tag_count": len(cleaned_tags),
        "osm_primary_tag": osm_primary_tag(cleaned_tags),
        "centroid": centroid_geojson(lon, lat),
        "lat": lat,
        "lon": lon,
        "bbox": json_dumps(bbox),
        "geometry": json_dumps(geom),
        "area_m2": computed.area_m2,
        "area_km2": computed.area_m2 / 1_000_000.0,
        "area_bucket": area_bucket(computed.area_m2),
        "has_name": bool(name),
        "has_wikidata": bool(qids),
        "has_wikipedia": False,
        "wikipedia_language_count": 0,
        "wikipedia_languages": "[]",
        "wikipedia_article_count": 0,
        "has_english_wikipedia": False,
        "has_french_wikipedia": False,
        "text_available": False,
        "best_language": "",
        "extraction_version": __version__,
        "extracted_at": extracted_at or utc_now_iso(),
        "wikipedia_tag_refs": json_dumps(
            [
                {
                    "language": ref.language,
                    "title": ref.title,
                    "raw_key": ref.raw_key,
                    "raw_value": ref.raw_value,
                }
                for ref in refs
            ]
        ),
        "wikipedia_tag_rejections": json_dumps(
            [
                {
                    "raw_key": rejection.raw_key,
                    "raw_value": rejection.raw_value,
                    "reason": rejection.reason,
                }
                for rejection in rejections
            ]
        ),
        "discovery_sources": json_dumps(
            sorted(
                source
                for source, present in (
                    ("wikidata", bool(qids)),
                    ("wikipedia_tag", bool(refs)),
                )
                if present
            )
        ),
    }
    return row


def extract_v2_pbf(pbf_path: Path, *, settings: Settings) -> V2ExtractedPbf:
    """Stream one PBF while retaining Wikidata and Wikipedia-tag candidates."""
    stem = V2PbfStem.from_path(pbf_path)
    started = time.perf_counter()
    extracted_at = utc_now_iso()
    rows: list[dict[str, Any]] = []
    reader = PBFReader(pbf_path, include_wikipedia_tagged=True)

    def add(candidate: PolygonCandidate) -> None:
        if settings.limit is not None and len(rows) >= settings.limit:
            return
        row = candidate_to_v2_row(
            candidate,
            source_pbf_stem=stem.stem,
            region=stem.region,
            source_pbf=pbf_path.name,
            extracted_at=extracted_at,
        )
        if row is not None:
            rows.append(row)

    stream = getattr(reader, "iter_polygon_candidates", None)
    if callable(stream):
        typed = cast(Callable[[Callable[[PolygonCandidate], None]], None], stream)
        typed(add)
    else:
        for candidate in reader.collect_polygon_candidates():
            add(candidate)
    LOGGER.info("V2 extracted %d polygons from %s", len(rows), pbf_path.name)
    return V2ExtractedPbf(stem, tuple(rows), time.perf_counter() - started)


__all__ = [
    "V2ExtractedPbf",
    "V2PbfStem",
    "candidate_to_v2_row",
    "extract_v2_pbf",
]
