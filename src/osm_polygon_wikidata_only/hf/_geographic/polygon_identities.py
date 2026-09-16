"""Deterministic global identity and centroid records for polygon reporting."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .models import CoverageMapError
from .parquet_reader import iter_required_columns

PolygonIdentity = tuple[str, int | str]


@dataclass(frozen=True, slots=True)
class PolygonRecord:
    """One deterministic representative row for a global polygon identity."""

    identity: PolygonIdentity
    polygon_id: str
    wikidata: str
    lon: float | None
    lat: float | None
    source_path: Path


@dataclass(frozen=True, slots=True)
class PolygonIndex:
    """Global identity records and the regional polygon-id lookup."""

    records: dict[PolygonIdentity, PolygonRecord]
    by_polygon_id: dict[str, PolygonIdentity]


def load_unique_polygon_records(paths: Iterable[Path]) -> PolygonIndex:
    """Read polygon files in sorted order and deduplicate by OSM identity.

    Production polygon tables carry ``osm_type`` and ``osm_id``. Older
    fixture-shaped tables without those columns use the historical
    ``<source>:<osm_type>:<osm_id>`` suffix, or a namespaced polygon-id key
    when no typed suffix exists. The first sorted row wins coordinates so
    overlapping regional extracts produce deterministic map points.
    """
    records: dict[PolygonIdentity, PolygonRecord] = {}
    by_polygon_id: dict[str, PolygonIdentity] = {}
    for path in sorted({Path(value) for value in paths}):
        _load_polygon_file(path, records, by_polygon_id)
    return PolygonIndex(records=records, by_polygon_id=by_polygon_id)


def _load_polygon_file(
    path: Path,
    records: dict[PolygonIdentity, PolygonRecord],
    by_polygon_id: dict[str, PolygonIdentity],
) -> None:
    names = set(pq.read_schema(path).names)
    if "polygon_id" not in names:
        raise CoverageMapError(f"polygons parquet {path} is missing required column: polygon_id")
    typed_identity = {"osm_type", "osm_id"}.issubset(names)
    columns = tuple(
        column
        for column in ("polygon_id", "osm_type", "osm_id", "wikidata", "lon", "lat")
        if column in names
    )
    for row_index, row in enumerate(iter_required_columns(path, columns, label="polygons")):
        _record_polygon_row(
            path,
            row_index,
            row,
            typed_identity=typed_identity,
            records=records,
            by_polygon_id=by_polygon_id,
        )


def _record_polygon_row(
    path: Path,
    row_index: int,
    row: dict[str, Any],
    *,
    typed_identity: bool,
    records: dict[PolygonIdentity, PolygonRecord],
    by_polygon_id: dict[str, PolygonIdentity],
) -> None:
    polygon_id = _required_polygon_id(path, row_index, row.get("polygon_id"))
    identity = _polygon_identity(row, polygon_id, typed_identity=typed_identity)
    if identity is None:
        raise CoverageMapError(
            f"polygons parquet {path} row {row_index} (polygon_id={polygon_id}) "
            "has an invalid (osm_type, osm_id) identity"
        )
    previous = by_polygon_id.get(polygon_id)
    if previous is not None and previous != identity:
        raise CoverageMapError(
            f"polygons parquet {path} has conflicting identities for polygon_id={polygon_id}"
        )
    by_polygon_id[polygon_id] = identity
    records.setdefault(identity, _polygon_record(path, polygon_id, identity, row))


def _required_polygon_id(path: Path, row_index: int, value: Any) -> str:
    if value in (None, ""):
        raise CoverageMapError(f"polygons parquet {path} row {row_index} is missing polygon_id")
    return str(value)


def _polygon_identity(
    row: dict[str, Any], polygon_id: str, *, typed_identity: bool
) -> PolygonIdentity | None:
    if typed_identity:
        osm_type = row.get("osm_type")
        osm_id = row.get("osm_id")
        if osm_type in (None, "") or osm_id in (None, ""):
            return None
        try:
            return str(osm_type), int(osm_id)
        except (TypeError, ValueError):
            return None
    return _legacy_polygon_identity(polygon_id)


def _legacy_polygon_identity(polygon_id: str) -> PolygonIdentity:
    parts = polygon_id.rsplit(":", 2)
    if len(parts) == 3 and parts[1] and parts[2]:
        try:
            return parts[1], int(parts[2])
        except ValueError:
            pass
    return "legacy", polygon_id


def _polygon_record(
    path: Path,
    polygon_id: str,
    identity: PolygonIdentity,
    row: dict[str, Any],
) -> PolygonRecord:
    return PolygonRecord(
        identity=identity,
        polygon_id=polygon_id,
        wikidata=_text_value(row.get("wikidata")),
        lon=_coordinate(path, polygon_id, row.get("lon"), "lon"),
        lat=_coordinate(path, polygon_id, row.get("lat"), "lat"),
        source_path=path,
    )


def _text_value(value: Any) -> str:
    return "" if value in (None, "") else str(value)


def _coordinate(path: Path, polygon_id: str, value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as error:
        raise CoverageMapError(
            f"polygons parquet {path} has invalid {name} for {polygon_id}"
        ) from error
    if not math.isfinite(coordinate):
        return None
    return coordinate


__all__ = ["PolygonIdentity", "PolygonIndex", "PolygonRecord", "load_unique_polygon_records"]
