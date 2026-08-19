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
    area_km2,
    centroid_geojson,
    centroid_wkt,
    compute_polygon_geometry,
    merge_multi_polygon,
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
    "geometry, message",
    [
        ({"type": "LineString", "coordinates": []}, "Unsupported geometry type"),
        ({"type": "Polygon"}, "no coordinates"),
        ({"type": "Polygon", "coordinates": []}, "no rings"),
        ({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}, "only 3 vertices"),
    ],
)
def test_invalid_geometry_fails_with_actionable_error(
    geometry: dict[str, object], message: str
) -> None:
    with pytest.raises(GeometryError, match=message):
        compute_polygon_geometry(geometry)


def test_centroid_serializers_preserve_longitude_then_latitude() -> None:
    assert centroid_geojson(2.5, 48.1) == '{"coordinates": [2.5, 48.1], "type": "Point"}'
    assert centroid_wkt(2.5, 48.1) == "POINT(2.5000000000 48.1000000000)"


def test_merge_multi_polygon_flattens_polygon_and_multipolygon_parts() -> None:
    merged = merge_multi_polygon(
        [
            {"type": "Polygon", "coordinates": [_square(0, 0)]},
            {"type": "MultiPolygon", "coordinates": [[_square(2, 0)], [_square(4, 0)]]},
        ]
    )

    assert merged["type"] == "MultiPolygon"
    assert len(merged["coordinates"]) == 3


def test_merge_multi_polygon_rejects_non_polygon_geometry() -> None:
    with pytest.raises(GeometryError, match="Cannot merge geometry"):
        merge_multi_polygon([{"type": "Point", "coordinates": [0, 0]}])
