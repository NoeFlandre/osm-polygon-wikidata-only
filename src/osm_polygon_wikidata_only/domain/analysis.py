"""Pure analysis helpers: area buckets, primary tag, bbox from rings.

All functions are deterministic and side-effect free so they can be
unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any

# Buckets chosen to cover tiny features (a building) to country-scale
# polygons, in roughly logarithmic steps.
AREA_BUCKETS: tuple[tuple[float, str], ...] = (
    (100.0, "<100m2"),
    (1_000.0, "100m2-1k_m2"),
    (10_000.0, "1k_m2-10k_m2"),
    (100_000.0, "10k_m2-100k_m2"),
    (1_000_000.0, "0.1-1km2"),
    (10_000_000.0, "1-10km2"),
    (100_000_000.0, "10-100km2"),
)
# Anything above the last threshold is "Larger than 100 km^2".
INF_BUCKET = ">100km2"


def area_bucket(area_m2: float) -> str:
    """Map a square-meter area to a human-readable bucket string.

    The bucket boundaries are inclusive on the lower end.
    """
    if area_m2 < 0:
        return "<100m2"  # degenerate; report the smallest bucket
    for upper, label in AREA_BUCKETS:
        if area_m2 < upper:
            return label
    return INF_BUCKET


# Tag keys we use to pick a single ``osm_primary_tag`` for analysis.
# Order matters: the first present key wins. We bias toward the most
# semantically meaningful OSM top-level tag.
PRIMARY_TAG_PREFERENCE: tuple[str, ...] = (
    "boundary",
    "landuse",
    "natural",
    "place",
    "building",
    "leisure",
    "amenity",
    "waterway",
    "highway",
    "aeroway",
    "railway",
    "man_made",
    "historic",
    "tourism",
    "shop",
    "office",
    "craft",
    "public_transport",
)


def osm_primary_tag(tags: dict[str, str]) -> str:
    """Pick the most informative OSM primary tag for the polygon.

    Returns a string of the form ``key=value`` (e.g. ``landuse=forest``)
    or an empty string if none of the preferred keys are present.
    """
    for key in PRIMARY_TAG_PREFERENCE:
        if tags.get(key):
            return f"{key}={tags[key]}"
    return ""


def bbox_from_geom(geom: dict[str, Any]) -> list[float]:
    """Compute ``[min_lon, min_lat, max_lon, max_lat]`` from a GeoJSON dict.

    Supports both Polygon and MultiPolygon. Returns an empty list if
    the geometry has no usable coordinates.
    """
    geometry_type = geom.get("type")
    coords = geom.get("coordinates")
    if not coords or geometry_type not in {"Polygon", "MultiPolygon"}:
        return []
    bounds = _geometry_bounds(coords, geometry_type)
    if bounds is None:
        return []
    return list(bounds)


def _geometry_bounds(
    coords: Any,
    geometry_type: str,
) -> tuple[float, float, float, float] | None:
    points = _geometry_points(coords, geometry_type)
    values: list[tuple[float, float]] = []
    for point in points:
        parsed = _coordinate_pair(point)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None
    longitudes, latitudes = zip(*values, strict=True)
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def _geometry_points(coords: Any, geometry_type: str) -> list[Any]:
    if geometry_type == "Polygon":
        return _polygon_points(coords)
    return _multipolygon_points(coords)


def _polygon_points(coords: Any) -> list[Any]:
    return [point for ring in coords for point in ring]


def _multipolygon_points(coords: Any) -> list[Any]:
    return [point for polygon in coords if polygon for ring in polygon for point in ring]


def _coordinate_pair(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
