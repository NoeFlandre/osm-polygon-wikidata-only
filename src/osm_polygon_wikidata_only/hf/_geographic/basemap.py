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
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches

LOGGER = logging.getLogger(__name__)


# Shared world-extent visual constants used by every visualization.
_OCEAN_COLOR = "#cfe2f3"
_LAND_COLOR = "#e8e0d0"
_LAND_EDGE = "#b8aa90"

# Shared figure layout constants.
_FIGSIZE = (16, 8)
_DPI = 100


def load_land_basemap(cache_dir: Path) -> list[Any] | None:
    """Load the cached Natural Earth 110m landmass GeoJSON, if available.

    Returns the parsed ``features`` list, or ``None`` if the cache is
    missing. We intentionally do not download anything here; rendering
    without landmasses is acceptable and the surrounding module performs
    no HTTP requests of its own (it relies on ``coverage_map.ensure_world_land``
    to manage the cache when a landmass overlay is requested).
    """
    candidate = cache_dir / "ne_110m_land.geojson"
    if not candidate.exists() or candidate.stat().st_size == 0:
        return None
    data = _read_land_geojson(candidate)
    return data.get("features") or [] if data is not None else None


def _read_land_geojson(candidate: Path) -> dict[str, Any] | None:
    """Read one cached GeoJSON object, logging malformed cache content."""
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.warning("Could not read cached land GeoJSON: %s", error)
        return None
    if not isinstance(data, dict):
        return None
    return data


def draw_landmasses(ax: Any, features: Sequence[Any]) -> None:
    """Draw Natural Earth landmasses on ``ax``."""
    for feature in features:
        for ring in _feature_rings(feature):
            _draw_land_ring(ax, ring)


def _feature_rings(feature: Any) -> list[Sequence[Sequence[float]]]:
    """Return exterior rings from one Natural Earth feature."""
    geometry = _feature_geometry(feature)
    return _geometry_rings(geometry) if geometry is not None else []


def _feature_geometry(feature: Any) -> dict[str, Any] | None:
    """Return a feature's geometry mapping when it has one."""
    if not isinstance(feature, dict):
        return None
    geom = feature.get("geometry")
    return geom if isinstance(geom, dict) else None


def _geometry_rings(geom: dict[str, Any]) -> list[Sequence[Sequence[float]]]:
    """Return exterior rings for a Polygon or MultiPolygon geometry."""
    gtype = geom.get("type")
    if gtype == "Polygon":
        return _polygon_rings(geom.get("coordinates"))
    if gtype == "MultiPolygon":
        return _multipolygon_rings(geom.get("coordinates"))
    return []


def _polygon_rings(coords: Any) -> list[Sequence[Sequence[float]]]:
    """Return the exterior ring of a non-empty Polygon."""
    return [coords[0]] if coords else []


def _multipolygon_rings(coords: Any) -> list[Sequence[Sequence[float]]]:
    """Return exterior rings of all non-empty MultiPolygon members."""
    if not coords:
        return []
    return [polygon[0] for polygon in coords if polygon]


def _draw_land_ring(ax: Any, ring: Sequence[Sequence[float]]) -> None:
    if not ring or len(ring) < 3:
        return
    patch = mpatches.Polygon(
        [(float(lon), float(lat)) for lon, lat in ring],
        closed=True,
        facecolor=_LAND_COLOR,
        edgecolor=_LAND_EDGE,
        linewidth=0.2,
        zorder=1,
    )
    ax.add_patch(patch)


def init_axes(ax: Any) -> None:
    """Apply the shared world-extent styling used by every visualization."""
    ax.set_facecolor(_OCEAN_COLOR)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(True, color="#ffffff", linewidth=0.3, alpha=0.4)
    ax.tick_params(colors="#666666", labelsize=7)
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
]
