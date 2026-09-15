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
    if payload.get("type") == "Polygon":
        return [coordinates]
    if payload.get("type") != "MultiPolygon":
        return None
    return coordinates if all(isinstance(part, list) for part in coordinates) else None


def _sample_from_parts(is_multipolygon: bool, parts: list[Any]) -> GeometrySample | None:
    rings = 0
    holes = 0
    vertices = 0
    for part in parts:
        if not isinstance(part, list) or not part:
            return None
        rings += len(part)
        holes += len(part) - 1
        vertices += sum(_ring_vertices(ring) for ring in part)
    return GeometrySample(is_multipolygon, len(parts), rings, holes, vertices)


def _ring_vertices(ring: Any) -> int:
    """Return the distinct vertex count of one ring."""
    if not isinstance(ring, list) or not ring:
        return 0
    return len(ring) - 1 if _is_closed(ring) else len(ring)


def _is_closed(ring: list[Any]) -> bool:
    return len(ring) > 1 and ring[0] == ring[-1]


def _bbox_numbers(raw: object) -> tuple[float, float, float, float] | None:
    payload = _decode_json_list(raw)
    if payload is None or len(payload) != 4:
        return None
    if not all(_is_finite_number(value) for value in payload):
        return None
    return (float(payload[0]), float(payload[1]), float(payload[2]), float(payload[3]))


def _is_finite_number(value: object) -> bool:
    """Return ``True`` for a real JSON number; booleans are not numbers here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _decode_json_list(raw: object) -> list[Any] | None:
    if not isinstance(raw, (str, bytes)) or not raw:
        return None
    try:
        payload: Any = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, list) else None


__all__ = ["BboxSample", "GeometrySample", "decode_bbox", "decode_geometry"]
