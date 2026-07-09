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
    t = geom.get("type")
    coords = geom.get("coordinates")
    if not coords or t not in {"Polygon", "MultiPolygon"}:
        return []

    min_lon = float("inf")
    min_lat = float("inf")
    max_lon = float("-inf")
    max_lat = float("-inf")

    def _walk(points: list[Any]) -> None:
        nonlocal min_lon, min_lat, max_lon, max_lat
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                lon, lat = float(pt[0]), float(pt[1])
            except (TypeError, ValueError):
                continue
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat

    if t == "Polygon":
        for ring in coords:
            _walk(ring)
    else:  # MultiPolygon
        for poly in coords:
            if not poly:
                continue
            for ring in poly:
                _walk(ring)

    if min_lon == float("inf"):
        return []
    return [min_lon, min_lat, max_lon, max_lat]
