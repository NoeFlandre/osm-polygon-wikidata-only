"""Polygon geometry helpers: centroid, geodesic area, serialization.

Pure functions operating on GeoJSON geometry dicts (the format produced
by :class:`osmium.geom.GeoJSONFactory`). No side effects, no I/O.

Area is computed with an equirectangular projection around the polygon's
own centroid. This is the standard "small polygon" approximation:
accurate to ~0.1% for polygons smaller than ~100 km wide, which covers
the vast majority of OSM polygons.

For multi-polygons, holes (inner rings) subtract from the outer-ring
sum, so the result is the signed planar area.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# WGS84 equatorial radius (meters). Sufficient for sub-kilometer accuracy
# at non-polar latitudes.
EARTH_RADIUS_M = 6_378_137.0


class GeometryError(ValueError):
    """Raised when a geometry cannot be processed."""


@dataclass(frozen=True)
class PolygonGeometry:
    """Computed centroid + area for a polygon or multi-polygon."""

    lon: float  # centroid longitude
    lat: float  # centroid latitude
    area_m2: float  # signed area in m^2 (always >= 0 for valid polygons)


def _rings(geom: dict[str, Any]) -> list[list[list[float]]]:
    """Extract rings from a GeoJSON Polygon or MultiPolygon.

    For a Polygon, returns [outer, *holes]. For a MultiPolygon, returns
    each part's rings flattened into one list (outer first, then its
    holes).
    """
    t = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        raise GeometryError("Geometry has no coordinates.")
    if t == "Polygon":
        return list(coords)
    if t == "MultiPolygon":
        out: list[list[list[float]]] = []
        for part in coords:
            out.extend(part)
        return out
    raise GeometryError(f"Unsupported geometry type for polygon: {t!r}")


def _ring_signed_area_and_centroid(
    ring: list[list[float]], lat0_rad: float, cos_lat0: float
) -> tuple[float, float, float, float]:
    """Compute signed area and projected centroid accumulators for one ring.

    Returns ``(cross_sum, sx, sy, n)``:

    * ``cross_sum`` = ``sum(x_i * y_{i+1} - x_{i+1} * y_i)`` (twice the
      signed planar area in m^2 once the caller multiplies by 0.5).
    * ``sx`` and ``sy`` are the unscaled projected-centroid accumulators
      whose value is ``6 * A * C`` (per GeoJSON ring orientation: outer
      rings are CCW, holes are CW per RFC 7946).
    * ``n`` = number of distinct vertices.

    Coordinates are *absolute* in projected meters (origin at the prime
    meridian / equator), so the caller applies the reference lon/lat
    offset once at the end.
    """
    if len(ring) < 4:
        # A closed ring needs at least 4 entries: 3 distinct vertices + repeat.
        raise GeometryError(f"Ring has only {len(ring)} vertices.")

    cross_sum = 0.0
    sx = 0.0
    sy = 0.0

    # GeoJSON rings are closed: first == last. Iterate i in [0, n) and
    # pair each vertex with the next one (i+1 modulo n).
    n = len(ring) - 1
    for i in range(n):
        p = ring[i]
        q = ring[(i + 1) % n]
        lon_i, lat_i = p[0], p[1]
        lon_j, lat_j = q[0], q[1]

        xi = EARTH_RADIUS_M * math.radians(lon_i) * cos_lat0
        yi = EARTH_RADIUS_M * math.radians(lat_i)
        xj = EARTH_RADIUS_M * math.radians(lon_j) * cos_lat0
        yj = EARTH_RADIUS_M * math.radians(lat_j)

        cross = xi * yj - xj * yi
        cross_sum += cross
        sx += (xi + xj) * cross
        sy += (yi + yj) * cross

    return cross_sum, sx, sy, float(n)


def _area_and_centroid_for_geom(
    rings: list[list[list[float]]],
) -> tuple[float, float, float]:
    """Compute signed area (m^2) and geographic centroid for a list of rings.

    Holes (CW rings per RFC 7946) subtract from outer rings. The result
    is the absolute area in m^2 and a centroid in (lon, lat) degrees.
    """
    if not rings:
        raise GeometryError("Geometry has no rings.")

    # Reference latitude for the projection: unweighted vertex mean of
    # the first (outer) ring. This gives an O(dx) accurate centroid.
    ref_lat = sum(v[1] for v in rings[0][:-1]) / max(1, len(rings[0]) - 1)
    lat0_rad = math.radians(ref_lat)
    cos_lat0 = math.cos(lat0_rad)
    if abs(cos_lat0) < 1e-12:
        # Polar area: fall back to a degenerate value rather than /0.
        cos_lat0 = 1e-12

    total_cross = 0.0
    total_sx = 0.0
    total_sy = 0.0

    for ring in rings:
        c, sx, sy, _ = _ring_signed_area_and_centroid(ring, lat0_rad, cos_lat0)
        total_cross += c
        total_sx += sx
        total_sy += sy

    # ``total_cross`` is 2 * signed_area_in_m2.
    signed_area_m2 = 0.5 * total_cross
    if signed_area_m2 == 0.0:
        # Empty or fully degenerated shape: fall back to the reference
        # point + 0 area so the caller still gets a reasonable centroid.
        ref_lon = sum(v[0] for v in rings[0][:-1]) / max(1, len(rings[0]) - 1)
        return 0.0, ref_lon, ref_lat

    # Centroid in projected absolute meters (origin at prime meridian / equator).
    cx_proj = total_sx / (6.0 * signed_area_m2)
    cy_proj = total_sy / (6.0 * signed_area_m2)

    area_m2 = abs(signed_area_m2)
    # Convert projected centroid back to (lon, lat). Because the
    # projected coordinates were computed in *absolute* radians-into-
    # meters, ``cx_proj`` already equals ``R * cos(lat0) * centroid_lon_rad``
    # — no extra reference addition is required.
    lon_rad = cx_proj / (EARTH_RADIUS_M * cos_lat0)
    lat_rad = cy_proj / EARTH_RADIUS_M
    return area_m2, math.degrees(lon_rad), math.degrees(lat_rad)


def compute_polygon_geometry(geom: dict[str, Any]) -> PolygonGeometry:
    """Compute centroid (lon, lat) and area (m^2) for a GeoJSON Polygon/MultiPolygon.

    Raises
    ------
    GeometryError
        If the geometry is empty, missing coordinates, or not a Polygon/MultiPolygon.
    """
    rings = _rings(geom)
    area_m2, lon, lat = _area_and_centroid_for_geom(rings)
    return PolygonGeometry(lon=lon, lat=lat, area_m2=area_m2)


def area_km2(area_m2: float) -> float:
    """Convert square meters to square kilometers."""
    return area_m2 / 1_000_000.0


def centroid_geojson(lon: float, lat: float) -> str:
    """Serialize a centroid as a GeoJSON Point string."""
    return json.dumps(
        {"type": "Point", "coordinates": [lon, lat]},
        ensure_ascii=False,
        sort_keys=True,
    )


def centroid_wkt(lon: float, lat: float) -> str:
    """Serialize a centroid as a WKT string (GeoJSON axis order: lon, lat)."""
    return f"POINT({lon:.10f} {lat:.10f})"


def merge_multi_polygon(geometries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate several GeoJSON Polygon/MultiPolygon geometries into one MultiPolygon."""
    parts: list[list[list[list[float]]]] = []
    for geom in geometries:
        t = geom.get("type")
        coords = geom.get("coordinates")
        if t == "Polygon":
            assert isinstance(coords, list)
            parts.append(coords)
        elif t == "MultiPolygon":
            assert isinstance(coords, list)
            parts.extend(coords)
        else:
            raise GeometryError(f"Cannot merge geometry of type {t!r}.")
    return {"type": "MultiPolygon", "coordinates": parts}
