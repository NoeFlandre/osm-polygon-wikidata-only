"""H3 cell assignment, coordinate validation, and antimeridian geometry.

This module owns the coordinate → H3 mapping and the cell → ring
geometry helpers shared by both the coverage and the polygon-count
visualizations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

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
    lat_value, lon_value = _coerce_coordinates(lat, lon)
    _validate_coordinate_ranges(lat_value, lon_value)
    _validate_resolution(resolution)
    try:
        return str(h3.latlng_to_cell(lat_value, lon_value, resolution))
    except (ValueError, h3.H3ValueError) as error:
        raise CoverageMapError(
            f"Could not assign H3 cell for ({lat_value}, {lon_value}) "
            f"at resolution {resolution}: {error}"
        ) from error


def _coerce_coordinates(lat: float, lon: float) -> tuple[float, float]:
    if lat is None or lon is None:
        raise CoverageMapError("Latitude and longitude must not be null.")
    values = _coerce_numeric_coordinates(lat, lon)
    _ensure_finite_coordinates(values)
    return values


def _coerce_numeric_coordinates(lat: object, lon: object) -> tuple[float, float]:
    try:
        return float(cast(Any, lat)), float(cast(Any, lon))
    except (TypeError, ValueError) as error:
        raise CoverageMapError(
            f"Latitude and longitude must be numeric; got lat={lat!r}, lon={lon!r}."
        ) from error


def _ensure_finite_coordinates(values: tuple[float, float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise CoverageMapError(
            f"Latitude and longitude must be finite; got lat={values[0]!r}, lon={values[1]!r}."
        )


def _validate_coordinate_ranges(lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise CoverageMapError(f"Latitude {lat} is outside the [-90, 90] range.")
    if not (-180.0 <= lon <= 180.0):
        raise CoverageMapError(f"Longitude {lon} is outside the [-180, 180] range.")


def _validate_resolution(resolution: object) -> None:
    if not isinstance(resolution, int) or not (0 <= resolution <= 15):
        raise CoverageMapError(f"H3 resolution must be an int in [0, 15]; got {resolution!r}.")


def split_antimeridian(points: Sequence[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Clip an antimeridian-crossing polygon into closed local rings.

    Merely splitting at a longitude jump leaves open fragments. A plotting
    library then closes those fragments with a world-spanning segment. We
    instead unwrap the polygon, clip it against each 360-degree world slab,
    and shift the resulting closed polygons back into ``[-180, 180]``.
    """
    if len(points) < 3 or not _crosses_antimeridian(points):
        return [list(points)]

    unwrapped = _unwrap_points(points)
    min_slab, max_slab = _slab_bounds(unwrapped)
    return _clip_slabs(unwrapped, min_slab, max_slab)


def _crosses_antimeridian(points: Sequence[tuple[float, float]]) -> bool:
    return not all(
        abs(points[index][0] - points[index - 1][0]) <= 180.0 for index in range(len(points))
    )


def _unwrap_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unwrapped = [points[0]]
    for lon, lat in points[1:]:
        previous_lon = unwrapped[-1][0]
        while lon - previous_lon > 180.0:
            lon -= 360.0
        while lon - previous_lon < -180.0:
            lon += 360.0
        unwrapped.append((lon, lat))
    return unwrapped


def _slab_bounds(points: Sequence[tuple[float, float]]) -> tuple[int, int]:
    minimum = math.floor((min(lon for lon, _ in points) + 180.0) / 360.0)
    maximum = math.floor((max(lon for lon, _ in points) + 180.0) / 360.0)
    return minimum, maximum


def _clip_slabs(
    points: Sequence[tuple[float, float]],
    minimum: int,
    maximum: int,
) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    for slab in range(minimum, maximum + 1):
        left = -180.0 + 360.0 * slab
        right = 180.0 + 360.0 * slab
        clipped = _clip_longitude(
            _clip_longitude(points, left, keep_greater=True), right, keep_greater=False
        )
        if len(clipped) >= 3:
            rings.append([(lon - 360.0 * slab, lat) for lon, lat in clipped])
    return rings


def _clip_longitude(
    points: Sequence[tuple[float, float]],
    boundary: float,
    *,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Clip ``points`` against one vertical longitude boundary."""
    if not points:
        return []

    inside = _inside_boundary(boundary, keep_greater)
    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        _append_clipped_segment(
            output,
            previous,
            current,
            previous_inside,
            current_inside,
            boundary,
        )
        previous = current
        previous_inside = current_inside
    return output


def _inside_boundary(boundary: float, keep_greater: bool):
    return lambda point: point[0] >= boundary if keep_greater else point[0] <= boundary


def _intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    boundary: float,
) -> tuple[float, float]:
    delta = end[0] - start[0]
    if delta == 0.0:
        return (boundary, start[1])
    ratio = (boundary - start[0]) / delta
    return (boundary, start[1] + ratio * (end[1] - start[1]))


def _append_clipped_segment(
    output: list[tuple[float, float]],
    previous: tuple[float, float],
    current: tuple[float, float],
    previous_inside: bool,
    current_inside: bool,
    boundary: float,
) -> None:
    if current_inside:
        if not previous_inside:
            output.append(_intersection(previous, current, boundary))
        output.append(current)
    elif previous_inside:
        output.append(_intersection(previous, current, boundary))


def cell_rings(cell: CoverageCell | PolygonCountCell) -> list[list[tuple[float, float]]]:
    """Return the antimeridian-split ``(lon, lat)`` rings for ``cell``."""
    try:
        boundary = h3.cell_to_boundary(cell.h3_cell)
    except (ValueError, h3.H3ValueError):
        LOGGER.warning("Could not fetch boundary for %s", cell.h3_cell)
        return []
    points = _boundary_points(boundary)
    return [ring for ring in split_antimeridian(points) if len(ring) >= 3]


def _boundary_points(boundary: object) -> list[tuple[float, float]]:
    if not boundary:
        return []
    pairs = cast(Sequence[Sequence[float]], boundary)
    points: list[tuple[float, float]] = []
    for pair in pairs:
        if len(pair) >= 2:
            points.append((float(pair[1]), float(pair[0])))
    return points
