"""Tests for osm_polygon_wikidata_only.domain.analysis."""

from __future__ import annotations

from osm_polygon_wikidata_only.domain.analysis import (
    area_bucket,
    bbox_from_geom,
    osm_primary_tag,
)


def test_area_bucket_tiny() -> None:
    assert area_bucket(50) == "<100m2"


def test_area_bucket_small() -> None:
    assert area_bucket(500) == "100m2-1k_m2"


def test_area_bucket_medium() -> None:
    assert area_bucket(50_000) == "10k_m2-100k_m2"


def test_area_bucket_one_km2() -> None:
    # 0.5 km^2 is in 0.1-1km2, 5 km^2 is in 1-10km2. The boundary
    # itself (1 km^2) is placed in the next bucket by convention.
    assert area_bucket(500_000) == "0.1-1km2"
    assert area_bucket(5_000_000) == "1-10km2"


def test_area_bucket_country_scale() -> None:
    assert area_bucket(200_000_000) == ">100km2"


def test_area_bucket_negative_clamps_to_smallest() -> None:
    assert area_bucket(-1) == "<100m2"


def test_osm_primary_tag_picks_boundary_first() -> None:
    assert (
        osm_primary_tag({"boundary": "administrative", "landuse": "forest"})
        == "boundary=administrative"
    )


def test_osm_primary_tag_falls_back_to_landuse() -> None:
    assert osm_primary_tag({"landuse": "forest"}) == "landuse=forest"


def test_osm_primary_tag_natural() -> None:
    assert osm_primary_tag({"natural": "water"}) == "natural=water"


def test_osm_primary_tag_unknown() -> None:
    assert osm_primary_tag({"random": "thing"}) == ""


def test_osm_primary_tag_skips_empty_value() -> None:
    assert osm_primary_tag({"boundary": ""}) == ""


def test_bbox_from_polygon() -> None:
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        ],
    }
    assert bbox_from_geom(geom) == [0.0, 0.0, 2.0, 2.0]


def test_bbox_from_multipolygon() -> None:
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]],
        ],
    }
    assert bbox_from_geom(geom) == [0.0, 0.0, 6.0, 6.0]


def test_bbox_from_empty_geom() -> None:
    assert bbox_from_geom({"type": "Polygon", "coordinates": []}) == []


def test_bbox_from_unsupported_type() -> None:
    assert bbox_from_geom({"type": "Point", "coordinates": [0, 0]}) == []


def test_bbox_handles_polygon_with_hole() -> None:
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]],
        ],
    }
    # The hole is included in the bbox span (correct, since the hole
    # is still part of the polygon's footprint).
    assert bbox_from_geom(geom) == [0.0, 0.0, 10.0, 10.0]
