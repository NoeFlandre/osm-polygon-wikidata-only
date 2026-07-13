"""H3 cell assignment, coordinate validation, and antimeridian geometry.

This module owns the coordinate → H3 mapping and the cell → ring
geometry helpers shared by both the coverage and the polygon-count
visualizations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import h3

from .models import CoverageMapError

if TYPE_CHECKING:
    from .models import CoverageCell, PolygonCountCell

LOGGER = logging.getLogger(__name__)


# Defaults and tunables --------------------------------------------------

DEFAULT_H3_RESOLUTION: int = 3
DEFAULT_MIN_POLYGONS_PER_CELL: int = 20


def assign_h3_cell(lat: float, lon: float, *, resolution: int = DEFAULT_H3_RESOLUTION) -> str:
    """Map a centroid to its H3 cell id at the requested resolution.

    Raises :class:`CoverageMapError` on null, NaN, out-of-range, or
    non-finite coordinates or on an invalid resolution.
    """
    if lat is None or lon is None:
        raise CoverageMapError("Latitude and longitude must not be null.")
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError) as error:
        raise CoverageMapError(
            f"Latitude and longitude must be numeric; got lat={lat!r}, lon={lon!r}."
        ) from error
    if not (math.isfinite(lat_value) and math.isfinite(lon_value)):
        raise CoverageMapError(
            f"Latitude and longitude must be finite; got lat={lat_value!r}, lon={lon_value!r}."
        )
    if not (-90.0 <= lat_value <= 90.0):
        raise CoverageMapError(f"Latitude {lat_value} is outside the [-90, 90] range.")
    if not (-180.0 <= lon_value <= 180.0):
        raise CoverageMapError(f"Longitude {lon_value} is outside the [-180, 180] range.")
    if not isinstance(resolution, int) or not (0 <= resolution <= 15):
        raise CoverageMapError(f"H3 resolution must be an int in [0, 15]; got {resolution!r}.")
    try:
        return str(h3.latlng_to_cell(lat_value, lon_value, resolution))
    except (ValueError, h3.H3ValueError) as error:
        raise CoverageMapError(
            f"Could not assign H3 cell for ({lat_value}, {lon_value}) "
            f"at resolution {resolution}: {error}"
        ) from error


def split_antimeridian(points: Sequence[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Split a polygon ring at longitudinal jumps larger than 180 degrees."""
    if len(points) < 4:
        return [list(points)]
    from itertools import pairwise

    rings: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    for prev, curr in pairwise(points):
        if abs(curr[0] - prev[0]) > 180.0:
            rings.append(current)
            current = [curr]
        else:
            current.append(curr)
    if len(current) >= 3:
        rings.append(current)
    elif current:
        # Carry a degenerate tail onto the previous ring so we never
        # produce zero-area patches.
        if rings:
            rings[-1].extend(current)
        else:
            rings.append(current)
    return rings


def cell_rings(cell: CoverageCell | PolygonCountCell) -> list[list[tuple[float, float]]]:
    """Return the antimeridian-split ``(lon, lat)`` rings for ``cell``."""
    try:
        boundary = h3.cell_to_boundary(cell.h3_cell)
    except (ValueError, h3.H3ValueError):
        LOGGER.warning("Could not fetch boundary for %s", cell.h3_cell)
        return []
    if not boundary:
        return []
    boundary_pairs = list(boundary)
    raw_points: list[tuple[float, float]] = []
    for pair in boundary_pairs:
        if len(pair) >= 2:
            raw_points.append((float(pair[0]), float(pair[1])))
    points: list[tuple[float, float]] = [(lon, lat) for lat, lon in raw_points]
    return [ring for ring in split_antimeridian(points) if len(ring) >= 3]
