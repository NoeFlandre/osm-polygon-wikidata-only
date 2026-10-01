"""Property-based invariants for :mod:`osm_polygon_wikidata_only.domain.geometry`."""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from osm_polygon_wikidata_only.domain.geometry import (
    area_km2,
    centroid_geojson,
    compute_polygon_geometry,
)

Ring = list[list[float]]

_REL = 1e-7


@st.composite
def _convex_ring(draw: st.DrawFn) -> Ring:
    """A closed, counter-clockwise convex ring of a small polygon (< ~50 km)."""

    center_lon = draw(st.floats(min_value=-170.0, max_value=170.0))
    center_lat = draw(st.floats(min_value=-70.0, max_value=70.0))
    radius = draw(st.floats(min_value=0.001, max_value=0.4))
    count = draw(st.integers(min_value=3, max_value=12))
    gaps = draw(st.lists(st.floats(min_value=0.2, max_value=1.0), min_size=count, max_size=count))
    total = sum(gaps)
    angle = draw(st.floats(min_value=0.0, max_value=2 * math.pi))
    points: Ring = []
    for gap in gaps:
        points.append(
            [center_lon + radius * math.cos(angle), center_lat + radius * math.sin(angle)]
        )
        angle += 2 * math.pi * gap / total
    return [*points, points[0]]


def _polygon(*rings: Ring) -> dict[str, object]:
    return {"type": "Polygon", "coordinates": [list(ring) for ring in rings]}


def _reversed(ring: Ring) -> Ring:
    return list(reversed(ring))


def _rotated(ring: Ring, shift: int) -> Ring:
    open_ring = ring[:-1]
    shift %= len(open_ring)
    rotated = open_ring[shift:] + open_ring[:shift]
    return [*rotated, rotated[0]]


def _hole_inside(ring: Ring, scale: float) -> Ring:
    """A clockwise ring shrunk towards the vertex mean, strictly inside a convex ring."""

    open_ring = ring[:-1]
    mean_lon = sum(p[0] for p in open_ring) / len(open_ring)
    mean_lat = sum(p[1] for p in open_ring) / len(open_ring)
    shrunk = [
        [mean_lon + scale * (p[0] - mean_lon), mean_lat + scale * (p[1] - mean_lat)]
        for p in open_ring
    ]
    return _reversed([*shrunk, shrunk[0]])


@given(ring=_convex_ring())
def test_area_is_positive_for_convex_ring(ring: Ring) -> None:
    assert compute_polygon_geometry(_polygon(ring)).area_m2 > 0


@given(ring=_convex_ring(), shift=st.integers(min_value=0, max_value=20))
def test_area_ignores_orientation_and_start_vertex(ring: Ring, shift: int) -> None:
    base = compute_polygon_geometry(_polygon(ring)).area_m2
    assert compute_polygon_geometry(_polygon(_reversed(ring))).area_m2 == pytest.approx(
        base, rel=_REL
    )
    assert compute_polygon_geometry(_polygon(_rotated(ring, shift))).area_m2 == pytest.approx(
        base, rel=_REL
    )


@example(
    ring=[
        [105.001, 62.0],
        [104.99906030737921, 62.00034202014333],
        [105.00076604444313, 61.999357212390315],
        [105.001, 62.0],
    ],
    offset=1.0,
)
@given(ring=_convex_ring(), offset=st.floats(min_value=-5.0, max_value=5.0))
def test_area_is_invariant_under_longitude_translation(ring: Ring, offset: float) -> None:
    moved = [[lon + offset, lat] for lon, lat in ring]
    assert compute_polygon_geometry(_polygon(moved)).area_m2 == pytest.approx(
        compute_polygon_geometry(_polygon(ring)).area_m2, rel=1e-6
    )


@given(ring=_convex_ring(), scale=st.floats(min_value=0.1, max_value=0.9))
@example(
    ring=[
        [150.251, 45.5],
        [150.25030901699438, 45.50095105651629],
        [150.24919098300563, 45.49941221474771],
        [150.251, 45.5],
    ],
    scale=0.75,
)
def test_hole_reduces_polygon_area(ring: Ring, scale: float) -> None:
    outer = compute_polygon_geometry(_polygon(ring)).area_m2
    hole = _hole_inside(ring, scale)
    with_hole = compute_polygon_geometry(_polygon(ring, hole)).area_m2
    assert 0 <= with_hole < outer


@given(ring=_convex_ring())
def test_convex_centroid_lies_inside_bbox(ring: Ring) -> None:
    geometry = compute_polygon_geometry(_polygon(ring))
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    eps = 1e-9
    assert min(lons) - eps <= geometry.lon <= max(lons) + eps
    assert min(lats) - eps <= geometry.lat <= max(lats) + eps


@given(ring=_convex_ring())
def test_multipolygon_of_one_part_matches_polygon(ring: Ring) -> None:
    single = compute_polygon_geometry(_polygon(ring))
    multi = compute_polygon_geometry({"type": "MultiPolygon", "coordinates": [[ring]]})
    assert multi == single


@given(area=st.floats(min_value=0.0, max_value=1e15, allow_nan=False))
def test_area_km2_scales_by_one_million(area: float) -> None:
    assert area_km2(area) == area / 1e6


@given(
    lon=st.floats(min_value=-180.0, max_value=180.0),
    lat=st.floats(min_value=-90.0, max_value=90.0),
)
def test_centroid_geojson_round_trips(lon: float, lat: float) -> None:
    assert json.loads(centroid_geojson(lon, lat)) == {"type": "Point", "coordinates": [lon, lat]}
