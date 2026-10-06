"""Natural Earth basemap loading and world-axis setup.

This module owns the Natural Earth 110m landmass loading, cache
fallback, and the shared matplotlib axis initialization. Visualization-
specific styling (colormaps, alpha, captions) is owned by
:mod:`.coverage` and :mod:`.polygon_count`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from matplotlib.collections import PathCollection
from matplotlib.path import Path as MplPath

LOGGER = logging.getLogger(__name__)


# Shared world-extent visual constants used by every visualization.
_OCEAN_COLOR = "#cfe2f3"
_LAND_COLOR = "#e8e0d0"
_LAND_EDGE = "#b8aa90"

# Shared figure layout constants.
_FIGSIZE = (16, 8)
_DPI = 100
FIGSIZE = _FIGSIZE
DPI = _DPI


def load_land_basemap(cache_dir: Path) -> list[Any] | None:
    """Load the cached Natural Earth 110m landmass GeoJSON, if available.

    Returns the parsed ``features`` list, or ``None`` if the cache is
    missing. We intentionally do not download anything here; rendering
    without landmasses is acceptable and the surrounding module performs
    no HTTP requests of its own (it relies on ``coverage_map.ensure_world_land``
    to manage the cache when a landmass overlay is requested).
    """
    candidate = cache_dir / "ne_110m_land.geojson"
    if not candidate.is_file() or candidate.stat().st_size == 0:
        return None
    return load_land_features(candidate)


def load_land_features(geojson_path: Path) -> list[Any] | None:
    """Load and cache a GeoJSON file's feature list by canonical path and mtime."""
    try:
        candidate = geojson_path.resolve(strict=True)
        stat = candidate.stat()
    except OSError as error:
        LOGGER.warning("Could not read cached land GeoJSON: %s", error)
        return None
    if stat.st_size == 0:
        return None
    data = _read_land_geojson(candidate.as_posix(), stat.st_mtime_ns)
    if not isinstance(data, dict):
        return None
    features = data.get("features")
    return features if isinstance(features, list) else []


@lru_cache(maxsize=16)
def _read_land_geojson(path: str, _mtime_ns: int) -> dict[str, Any] | None:
    """Parse each canonical GeoJSON path once per modification time."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.warning("Could not read cached land GeoJSON: %s", error)
        return None
    return data if isinstance(data, dict) else None


def draw_landmasses(
    ax: Any,
    features: Sequence[Any],
    *,
    facecolor: str = _LAND_COLOR,
    edgecolor: str = _LAND_EDGE,
    linewidth: float = 0.2,
    zorder: float = 1,
) -> None:
    """Draw all polygon geometry in one collection while preserving interior holes."""
    paths: list[MplPath] = []
    for feature in features:
        paths.extend(_feature_paths(feature))
    if not paths:
        return
    collection = PathCollection(
        paths,
        facecolors=[facecolor],
        edgecolors=[edgecolor],
        linewidths=linewidth,
        zorder=zorder,
    )
    ax.add_collection(collection, autolim=False)


def _feature_paths(feature: Any) -> list[MplPath]:
    geometry = _feature_geometry(feature)
    if geometry is None:
        return []
    paths: list[MplPath] = []
    for polygon in _geometry_polygons(geometry):
        path = _polygon_path(polygon)
        if path is not None:
            paths.append(path)
    return paths


def _geometry_coordinates(geometry: dict[str, Any]) -> list[Any] | tuple[Any, ...] | None:
    coordinates = geometry.get("coordinates")
    if isinstance(coordinates, (list, tuple)) and coordinates:
        return coordinates
    return None


def _geometry_polygons(geometry: dict[str, Any]) -> list[Any]:
    """Return coordinate-ring groups for Polygon and MultiPolygon geometry."""
    coordinates = _geometry_coordinates(geometry)
    if not coordinates:
        return []
    return _polygon_groups(geometry.get("type"), coordinates)


def _polygon_groups(geometry_type: Any, coordinates: list[Any] | tuple[Any, ...]) -> list[Any]:
    if geometry_type == "Polygon":
        return [coordinates]
    if geometry_type != "MultiPolygon":
        return []
    return list(filter(_is_coordinate_group, coordinates))


def _is_coordinate_group(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _polygon_path(rings: Any) -> MplPath | None:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring_index, ring in enumerate(rings):
        points = _closed_oriented_ring(ring, exterior=ring_index == 0)
        if points is None:
            continue
        vertices.extend(points)
        codes.append(int(MplPath.MOVETO))
        codes.extend(int(MplPath.LINETO) for _ in points[1:-1])
        codes.append(int(MplPath.CLOSEPOLY))
    if not vertices:
        return None
    return MplPath(vertices, codes)


def _closed_oriented_ring(ring: Any, *, exterior: bool) -> list[tuple[float, float]] | None:
    points = _ring_points(ring)
    if points is None:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    if _ring_needs_reversal(_ring_signed_area(points), exterior=exterior):
        points.reverse()
    return points


def _ring_points(ring: Any) -> list[tuple[float, float]] | None:
    if not _is_ring_candidate(ring):
        return None
    return _convert_ring(ring)


def _is_ring_candidate(ring: Any) -> bool:
    return isinstance(ring, (list, tuple)) and len(ring) >= 3


def _convert_ring(ring: Sequence[Any]) -> list[tuple[float, float]] | None:
    try:
        points = [(float(point[0]), float(point[1])) for point in ring]
    except (IndexError, TypeError, ValueError):
        return None
    return points if len(points) >= 3 else None


def _ring_signed_area(points: Sequence[tuple[float, float]]) -> float:
    return sum((x1 * y2) - (x2 * y1) for (x1, y1), (x2, y2) in pairwise(points))


def _ring_needs_reversal(signed_area: float, *, exterior: bool) -> bool:
    return signed_area < 0 if exterior else signed_area > 0


def _feature_geometry(feature: Any) -> dict[str, Any] | None:
    """Return a feature's geometry mapping when it has one."""
    if not isinstance(feature, dict):
        return None
    geom = feature.get("geometry")
    return geom if isinstance(geom, dict) else None


def init_axes(ax: Any, *, equal_aspect: bool = True) -> None:
    """Apply world-extent styling, optionally deferring the aspect policy."""
    ax.set_facecolor(_OCEAN_COLOR)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(True, color="#ffffff", linewidth=0.3, alpha=0.4)
    ax.tick_params(colors="#666666", labelsize=7)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


# Re-export the shared visual constants so coverage/polygon_count can
# reach them without re-declaring. They are private to this package.
__all__ = [
    "_DPI",
    "_FIGSIZE",
    "_LAND_COLOR",
    "_LAND_EDGE",
    "_OCEAN_COLOR",
    "draw_landmasses",
    "init_axes",
    "load_land_basemap",
    "load_land_features",
]
