"""Polygon extraction: PBF -> :class:`Polygon` rows.

This module is the bridge between :mod:`io.pbf_reader` (which yields
raw polygon candidates from osmium) and the :class:`Polygon` domain
model. It computes geometry, analysis metadata, and stable IDs but
does not perform any HTTP work.

It also owns :func:`extract_pbf` -- the PBF-streaming helper that
reads the source file, parses the filename into a :class:`PbfStem`,
filters out malformed candidates, respects ``settings.limit``, and
returns an immutable :class:`ExtractedPbf`. The processor facade
re-exports both for callers that already import from
``pipeline.processor``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.geometry import (
    GeometryError,
    PolygonGeometry,
    compute_polygon_geometry,
)
from osm_polygon_wikidata_only.domain.models import (
    Polygon,
    PolygonIdentity,
    PolygonOsmFields,
    PolygonShape,
)
from osm_polygon_wikidata_only.io.pbf_reader import PolygonCandidate
from osm_polygon_wikidata_only.pipeline.polygon_fields import build_common_polygon_fields
from osm_polygon_wikidata_only.utils.time import utc_now_iso

LOGGER = logging.getLogger(__name__)
# Lifecycle log messages ("Processing X (region=Y)" and
# "Extracted N polygons from X") previously emitted under
# :mod:`pipeline.processor`. Keep emitting them under that same
# legacy logger name so existing log filters continue to fire.
PROCESSOR_LOGGER = logging.getLogger("osm_polygon_wikidata_only.pipeline.processor")


@dataclass(frozen=True, slots=True)
class PbfStem:
    """A parsed PBF filename.

    ``stem`` is the part of the filename before ``.osm.pbf``
    (e.g. ``monaco-latest``). ``region`` is the part before
    ``-latest.osm.pbf`` (e.g. ``monaco``). The remote parquet paths
    use ``stem`` so the layout is stable across reruns.
    """

    path: Path
    stem: str
    region: str

    @classmethod
    def from_path(cls, path: Path) -> PbfStem:
        name = path.name
        stem = name[: -len(".osm.pbf")] if name.endswith(".osm.pbf") else path.stem
        region = stem[: -len("-latest")] if stem.endswith("-latest") else stem
        return cls(path=path, stem=stem, region=region)


@dataclass(frozen=True, slots=True)
class ExtractedPbf:
    """A PBF whose polygon rows are ready for enrichment.

    Carries the parsed :class:`PbfStem`, the immutable polygon
    tuple, and the wall-clock time spent streaming and converting
    candidates.
    """

    stem: PbfStem
    polygons: tuple[Polygon, ...]
    extraction_duration_s: float


def _parse_geom(geom_json: str) -> dict[str, object] | None:
    try:
        parsed: object = json.loads(geom_json)
    except json.JSONDecodeError as e:
        LOGGER.debug("Skipping element with invalid GeoJSON: %s", e)
        return None
    return cast(dict[str, object], parsed) if isinstance(parsed, dict) else None


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
    tag, bbox, area, and bucket use the shared field builder. Wikipedia
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
        return None
    fields = build_common_polygon_fields(
        osm_type=osm_type,
        osm_id=osm_id,
        tags=tags,
        geom=geom,
        computed=pg,
        region=region,
        source_pbf=source_pbf,
        extracted_at=extracted_at,
    )
    return Polygon.make(
        identity=PolygonIdentity(
            source_pbf_stem=source_pbf_stem,
            region=fields["region"],
            source_pbf=fields["source_pbf"],
            osm_type=fields["osm_type"],
            osm_id=fields["osm_id"],
        ),
        osm=PolygonOsmFields(
            wikidata=wikidata,
            name=fields["name"],
            tags=fields["tags"],
            tag_keys=fields["tag_keys"],
            tag_count=fields["tag_count"],
            osm_primary_tag=fields["osm_primary_tag"],
            has_name=fields["has_name"],
            has_wikidata=True,
        ),
        shape=PolygonShape(
            centroid=fields["centroid"],
            lat=fields["lat"],
            lon=fields["lon"],
            bbox=fields["bbox"],
            area_m2=fields["area_m2"],
            area_km2=fields["area_km2"],
            area_bucket=fields["area_bucket"],
            geometry=fields["geometry"],
        ),
        extraction_version=fields["extraction_version"],
        extracted_at=fields["extracted_at"],
    )


def polygon_to_dict(p: Polygon) -> dict[str, Any]:
    """Convert a :class:`Polygon` to the row dict used by :mod:`io.parquet`."""
    return dict(p.__dict__)


def extract_pbf(pbf_path: Path, *, settings: Settings) -> ExtractedPbf:
    """Stream *pbf_path* through the PBF reader and produce an
    :class:`ExtractedPbf`. No HTTP work, no durable writes.

    Reads candidates one at a time (``iter_polygon_candidates`` when
    available, otherwise the batch ``collect_polygon_candidates``
    fallback), filters out candidates missing a Wikidata tag or
    with invalid geometry, applies ``settings.limit`` as a hard
    cap, and records the elapsed time. The reader class is resolved
    lazily via the ``io.pbf_reader`` module so tests can monkeypatch
    ``PBFReader`` without importing it at extraction time.
    """
    stem = PbfStem.from_path(pbf_path)
    stage_started = time.perf_counter()
    extracted_at = utc_now_iso()
    PROCESSOR_LOGGER.info("Processing %s (region=%s)", pbf_path.name, stem.region)

    polygons: list[Polygon] = []
    # Keep this module import lazy so tests can replace the reader class at the
    # established ``io.pbf_reader.PBFReader`` seam before extraction begins.
    import osm_polygon_wikidata_only.io.pbf_reader as _pbf_reader_mod  # noqa: PLC0415

    reader = _pbf_reader_mod.PBFReader(pbf_path)

    def add_candidate(candidate: PolygonCandidate) -> None:
        if settings.limit is not None and len(polygons) >= settings.limit:
            return
        polygon = candidate_to_polygon(
            candidate,
            source_pbf_stem=stem.stem,
            region=stem.region,
            source_pbf=pbf_path.name,
            extracted_at=extracted_at,
        )
        if polygon is not None:
            polygons.append(polygon)

    stream_candidates = getattr(reader, "iter_polygon_candidates", None)
    if callable(stream_candidates):
        typed_stream = cast(Callable[[Callable[[PolygonCandidate], None]], None], stream_candidates)
        typed_stream(add_candidate)
    else:
        for candidate in reader.collect_polygon_candidates():
            add_candidate(candidate)
    PROCESSOR_LOGGER.info("Extracted %d polygons from %s", len(polygons), pbf_path.name)
    return ExtractedPbf(
        stem=stem,
        polygons=tuple(polygons),
        extraction_duration_s=time.perf_counter() - stage_started,
    )


__all__ = [
    "ExtractedPbf",
    "PbfStem",
    "candidate_to_polygon",
    "extract_pbf",
    "polygon_to_dict",
]
