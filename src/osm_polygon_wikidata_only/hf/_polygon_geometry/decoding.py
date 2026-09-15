"""Per-row GeoJSON and bbox decoding for the geometry statistics.

Both decoders are pure, total functions: an unreadable value yields
``None`` instead of raising, and the caller counts it. That keeps one
malformed row from failing a whole dataset-wide scan while still making
the malformed rows visible in the published statistics.

Vertex counting follows the ring convention used by the extraction
code: a closed ring repeats its first coordinate as its last, and that
repeat is not counted as a second vertex.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GeometrySample:
    """Ring structure of one readable Polygon or MultiPolygon row."""

    is_multipolygon: bool
    components: int
    rings: int
    holes: int
    vertices: int


@dataclass(frozen=True, slots=True)
class BboxSample:
    """One readable ``[min_lon, min_lat, max_lon, max_lat]`` bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def width_deg(self) -> float:
        """Longitude span in degrees."""
        return self.max_lon - self.min_lon

    @property
    def height_deg(self) -> float:
        """Latitude span in degrees."""
        return self.max_lat - self.min_lat

    @property
    def mean_lat(self) -> float:
        """Latitude the equirectangular metre conversion is evaluated at."""
        return (self.min_lat + self.max_lat) / 2.0


def decode_geometry(raw: object) -> GeometrySample | None:
    """Return the ring structure of ``raw``, or ``None`` when unreadable."""
    payload = _decode_json_object(raw)
    if payload is None:
        return None
    parts = _polygon_parts(payload)
    if parts is None:
        return None
    return _sample_from_parts(payload.get("type") == "MultiPolygon", parts)


def decode_bbox(raw: object) -> BboxSample | None:
    """Return the bounding box in ``raw``, or ``None`` when unreadable."""
    values = _bbox_numbers(raw)
    if values is None:
        return None
    min_lon, min_lat, max_lon, max_lat = values
    if max_lon < min_lon or max_lat < min_lat:
        return None
    return BboxSample(min_lon, min_lat, max_lon, max_lat)


def _decode_json_object(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, (str, bytes)) or not raw:
        return None
    try:
        payload: Any = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _polygon_parts(payload: dict[str, Any]) -> list[Any] | None:
    """Return the row's rings grouped per component, or ``None``."""
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None
    return _parts_for_type(str(payload.get("type")), coordinates)


def _parts_for_type(geometry_type: str, coordinates: list[Any]) -> list[Any] | None:
    """Group one geometry's coordinates per component by GeoJSON type."""
    if geometry_type == "Polygon":
        return [coordinates]
    if geometry_type != "MultiPolygon":
        return None
    return coordinates if _all_lists(coordinates) else None


def _all_lists(values: list[Any]) -> bool:
    return all(isinstance(value, list) for value in values)


def _sample_from_parts(is_multipolygon: bool, parts: list[Any]) -> GeometrySample | None:
    rings = 0
    holes = 0
    vertices = 0
    for part in parts:
        if not isinstance(part, list) or not part:
            return None
        counted = _part_vertices(part)
        if counted is None:
            return None
        rings += len(part)
        holes += len(part) - 1
        vertices += counted
    return GeometrySample(is_multipolygon, len(parts), rings, holes, vertices)


def _part_vertices(part: list[Any]) -> int | None:
    """Return one component's vertex count, or ``None`` when a ring is malformed."""
    total = 0
    for ring in part:
        counted = _ring_vertices(ring)
        if counted is None:
            return None
        total += counted
    return total


def _ring_vertices(ring: Any) -> int | None:
    """Return the distinct vertex count of one ring, or ``None`` when malformed.

    A ring must be a non-empty list of coordinate pairs. A scalar, a
    string, an empty ring, or an entry that is not a pair of finite
    numbers makes the whole row unreadable rather than a polygon with
    silently dropped vertices.
    """
    if not _is_ring(ring):
        return None
    return len(ring) - 1 if _is_closed(ring) else len(ring)


def _is_ring(ring: Any) -> bool:
    """Return ``True`` for a non-empty list of GeoJSON positions."""
    if not isinstance(ring, list) or not ring:
        return False
    return all(_is_coordinate(position) for position in ring)


def _is_coordinate(position: Any) -> bool:
    """Return ``True`` for a GeoJSON position.

    A position is a list of at least two ordinates, and every ordinate
    it carries -- a third altitude entry included -- must be a finite
    number. A row with a null, textual, or infinite ordinate anywhere in
    a position is unreadable rather than a polygon whose extra values
    were quietly ignored.
    """
    return (
        isinstance(position, list)
        and len(position) >= 2
        and all(_is_finite_number(value) for value in position)
    )


def _is_closed(ring: list[Any]) -> bool:
    return len(ring) > 1 and ring[0] == ring[-1]


def _bbox_numbers(raw: object) -> tuple[float, float, float, float] | None:
    payload = _decode_json_list(raw)
    if payload is None or len(payload) != 4:
        return None
    numbers: list[float] = []
    for value in payload:
        converted = _finite_float(value)
        if converted is None:
            return None
        numbers.append(converted)
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def _is_finite_number(value: object) -> bool:
    """Return ``True`` for a coordinate this scan can represent."""
    return _finite_float(value) is not None


def _finite_float(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one.

    Booleans are not numbers here, and a JSON integer too large for a
    float is not a coordinate: converting it raises, and one malformed
    row must count as unreadable rather than abort the whole scan.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _decode_json_list(raw: object) -> list[Any] | None:
    if not isinstance(raw, (str, bytes)) or not raw:
        return None
    try:
        payload: Any = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, list) else None


__all__ = ["BboxSample", "GeometrySample", "decode_bbox", "decode_geometry"]
