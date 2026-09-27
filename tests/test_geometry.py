"""Behavioral tests for pure polygon geometry helpers.

These tests cover normal polygons, holes, multipolygons, degenerate input,
serialization, and validation failures without touching the filesystem or
network.
"""

from __future__ import annotations

import random

import pytest

from osm_polygon_wikidata_only.domain.geometry import (
    GeometryError,
    _projection_for_rings,
    _ring_signed_area_and_centroid,
    area_km2,
    centroid_geojson,
    compute_polygon_geometry,
)


def _square(x0: float, y0: float, size: float = 1.0) -> list[list[float]]:
    return [
        [x0, y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0, y0 + size],
        [x0, y0],
    ]


def test_polygon_geometry_returns_centroid_and_positive_area() -> None:
    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [_square(0, 0)]})

    assert result.lon == pytest.approx(0.5)
    assert result.lat == pytest.approx(0.5)
    assert result.area_m2 > 0
    assert area_km2(result.area_m2) == pytest.approx(result.area_m2 / 1_000_000)


def test_polygon_geometry_translation_and_scale_invariants() -> None:
    """A deterministic property-style sweep checks geometric invariants."""
    rng = random.Random(20260819)
    for _ in range(20):
        x0 = rng.uniform(-170, 170)
        y0 = rng.uniform(-70, 70)
        size = rng.uniform(0.01, 2.0)
        result = compute_polygon_geometry(
            {"type": "Polygon", "coordinates": [_square(x0, y0, size)]}
        )

        assert result.lon == pytest.approx(x0 + size / 2)
        assert result.lat == pytest.approx(y0 + size / 2)
        assert result.area_m2 > 0


def test_tiny_polygon_keeps_centroid_accumulators_zero_initialized() -> None:
    size = 1e-6
    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [_square(0.0, 0.0, size)]})

    assert result.lon == pytest.approx(size / 2, abs=1e-12)
    assert result.lat == pytest.approx(size / 2, abs=1e-12)
    assert result.area_m2 > 0


def test_ring_moments_preserve_geojson_orientation_sign() -> None:
    ring = _square(0.0, 0.0)

    counterclockwise = _ring_signed_area_and_centroid(ring, cos_lat0=1.0)
    clockwise = _ring_signed_area_and_centroid(list(reversed(ring)), cos_lat0=1.0)

    assert counterclockwise[0] > 0
    assert clockwise[0] < 0


def test_reversed_ring_has_same_geometry() -> None:
    ring = _square(10, 20)
    forward = compute_polygon_geometry({"type": "Polygon", "coordinates": [ring]})
    reversed_result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [list(reversed(ring))]}
    )

    assert reversed_result.lon == pytest.approx(forward.lon)
    assert reversed_result.lat == pytest.approx(forward.lat)
    assert reversed_result.area_m2 == pytest.approx(forward.area_m2)


def test_polygon_hole_reduces_area() -> None:
    outer = _square(0, 0)
    # Clockwise inner ring, as required for a GeoJSON hole.
    hole = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25]]
    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [outer, hole]})
    full = compute_polygon_geometry({"type": "Polygon", "coordinates": [outer]})

    assert result.area_m2 == pytest.approx(full.area_m2 * 0.75)
    assert result.lon == pytest.approx(0.5)
    assert result.lat == pytest.approx(0.5)


def test_multipolygon_geometry_uses_area_weighted_centroid() -> None:
    result = compute_polygon_geometry(
        {
            "type": "MultiPolygon",
            "coordinates": [[_square(0, 0)], [_square(2, 0)]],
        }
    )

    assert result.lon == pytest.approx(1.5)
    assert result.lat == pytest.approx(0.5)


def test_degenerate_ring_returns_reference_point_and_zero_area() -> None:
    result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [2, 0], [0, 0]]]}
    )

    assert result.lon == 1.0
    assert result.lat == 0.0
    assert result.area_m2 == 0.0


def test_polar_geometry_uses_safe_projection_fallback() -> None:
    result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [[[0, 90], [1, 90], [2, 90], [0, 90]]]}
    )

    assert result.area_m2 == 0.0


@pytest.mark.parametrize(
    "geometry, expected_message",
    [
        (
            {"type": "LineString", "coordinates": []},
            "Unsupported geometry type for polygon: 'LineString'",
        ),
        ({"type": "Polygon"}, "Geometry has no coordinates."),
        ({"type": "Polygon", "coordinates": []}, "Geometry has no rings."),
        (
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]},
            "Ring has only 3 vertices.",
        ),
    ],
)
def test_invalid_geometry_fails_with_actionable_error(
    geometry: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(GeometryError) as exc_info:
        compute_polygon_geometry(geometry)

    assert str(exc_info.value) == expected_message


def test_projection_uses_exact_cosine_floor_at_pole() -> None:
    ring = [[0.0, 90.0], [1.0, 90.0], [2.0, 90.0], [0.0, 90.0]]
    _, _, cos_lat0 = _projection_for_rings([ring])

    assert cos_lat0 == 1e-12


def test_centroid_serializers_preserve_longitude_then_latitude() -> None:
    assert centroid_geojson(2.5, 48.1) == '{"coordinates": [2.5, 48.1], "type": "Point"}'


_DEG = 6_378_137.0 * 3.141592653589793 / 180.0


def test_polygon_area_matches_equirectangular_formula_exactly() -> None:
    import math

    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [_square(0, 60, 2.0)]})

    # Reference latitude is the vertex mean (61 deg); area uses its cosine.
    expected = (2.0 * _DEG) ** 2 * math.cos(math.radians(61.0))
    assert result.area_m2 == pytest.approx(expected, rel=1e-12)
    assert result.lon == pytest.approx(1.0, rel=1e-12)
    assert result.lat == pytest.approx(61.0, rel=1e-12)


def test_rectangle_centroid_uses_both_axes_independently() -> None:
    ring = [[10.0, 40.0], [16.0, 40.0], [16.0, 42.0], [10.0, 42.0], [10.0, 40.0]]

    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [ring]})

    assert result.lon == pytest.approx(13.0, rel=1e-12)
    assert result.lat == pytest.approx(41.0, rel=1e-12)


def test_triangle_centroid_is_vertex_mean_not_ring_mean() -> None:
    ring = [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]]

    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [ring]})

    assert result.lon == pytest.approx(1.0, rel=1e-9)
    assert result.lat == pytest.approx(1.0, rel=1e-9)


def test_off_centre_hole_shifts_centroid_away_from_hole() -> None:
    outer = _square(0, 0, 2.0)
    hole = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]

    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [outer, hole]})
    full = compute_polygon_geometry({"type": "Polygon", "coordinates": [outer]})

    assert result.area_m2 == pytest.approx(full.area_m2 * 0.75, rel=1e-12)
    # L-shape made of three unit squares: centroid (7/6, 7/6).
    assert result.lon == pytest.approx(7 / 6, rel=1e-9)
    assert result.lat == pytest.approx(7 / 6, rel=1e-9)


def test_multipolygon_area_sums_parts_and_weights_centroid() -> None:
    small = _square(0, 0, 1.0)
    large = _square(3, 0, 2.0)

    result = compute_polygon_geometry({"type": "MultiPolygon", "coordinates": [[small], [large]]})
    small_only = compute_polygon_geometry({"type": "Polygon", "coordinates": [small]})

    # Projection is fixed by the first part (reference latitude 0.5).
    assert result.area_m2 == pytest.approx(small_only.area_m2 * 5, rel=1e-12)
    assert result.lon == pytest.approx((0.5 * 1 + 4.0 * 4) / 5, rel=1e-9)
    assert result.lat == pytest.approx((0.5 * 1 + 1.0 * 4) / 5, rel=1e-9)


def test_multipolygon_parts_keep_their_holes() -> None:
    hole = [[3.5, 0.5], [3.5, 1.5], [4.5, 1.5], [4.5, 0.5], [3.5, 0.5]]

    result = compute_polygon_geometry(
        {"type": "MultiPolygon", "coordinates": [[_square(0, 0)], [_square(3, 0, 2.0), hole]]}
    )
    unit = compute_polygon_geometry({"type": "Polygon", "coordinates": [_square(0, 0)]})

    assert result.area_m2 == pytest.approx(unit.area_m2 * 4, rel=1e-12)


def test_degenerate_fallback_uses_distinct_vertex_means() -> None:
    result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [[[0, 3], [2, 3], [7, 3], [0, 3]]]}
    )

    assert result == type(result)(lon=3.0, lat=3.0, area_m2=0.0)


def test_polar_degenerate_ring_keeps_reference_point() -> None:
    result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [[[0, 90], [1, 90], [5, 90], [0, 90]]]}
    )

    assert result.lon == 2.0
    assert result.lat == 90.0


def test_near_polar_polygon_uses_clamped_cosine_without_dividing_by_zero() -> None:
    import math

    ring = [[0.0, 89.0], [1.0, 89.0], [1.0, 90.0], [0.0, 90.0], [0.0, 89.0]]

    result = compute_polygon_geometry({"type": "Polygon", "coordinates": [ring]})

    ref = (89.0 + 89.0 + 90.0 + 90.0) / 4
    assert result.area_m2 == pytest.approx(_DEG**2 * math.cos(math.radians(ref)), rel=1e-9)
    assert result.lon == pytest.approx(0.5, rel=1e-9)


def test_four_entry_ring_is_the_minimum_accepted() -> None:
    result = compute_polygon_geometry(
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]}
    )

    assert result.area_m2 > 0


def test_area_km2_divides_by_one_million() -> None:
    assert area_km2(2_500_000.0) == 2.5


def test_centroid_geojson_keeps_non_ascii_safe_and_sorted() -> None:
    assert centroid_geojson(-0.5, 1.25) == '{"coordinates": [-0.5, 1.25], "type": "Point"}'
