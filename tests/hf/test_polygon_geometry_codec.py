"""Exact payload tests for the polygon geometry statistics codec."""

from __future__ import annotations

import json

from osm_polygon_wikidata_only.hf._polygon_geometry.codec import stats_payload
from osm_polygon_wikidata_only.hf._polygon_geometry.models import (
    AreaHistogramBucket,
    AreaSummary,
    Distribution,
    ExtentSummary,
    PolygonGeometryStats,
    ShapeSummary,
    SourceAreaSummary,
)


def _dist(base: float) -> Distribution:
    return Distribution(
        minimum=base + 0.1,
        maximum=base + 0.2,
        mean=base + 0.3,
        median=base + 0.4,
        p95=base + 0.5,
        p99=base + 0.6,
    )


def _dist_json(base: float) -> dict[str, float]:
    return {
        "minimum": base + 0.1,
        "maximum": base + 0.2,
        "mean": base + 0.3,
        "median": base + 0.4,
        "p95": base + 0.5,
        "p99": base + 0.6,
    }


def _stats() -> PolygonGeometryStats:
    return PolygonGeometryStats(
        contract_version="v-test",
        file_count=2,
        polygon_count=3,
        area=AreaSummary(
            total_m2=1.0,
            minimum_m2=2.0,
            maximum_m2=3.0,
            mean_m2=4.0,
            median_m2=5.0,
            p1_m2=6.0,
            p5_m2=7.0,
            p25_m2=8.0,
            p75_m2=9.0,
            p95_m2=10.0,
            p99_m2=11.0,
            non_positive_count=12,
            below_one_m2_count=13,
            null_count=14,
        ),
        area_histogram=(
            AreaHistogramBucket(label="<1", lower_m2=None, upper_m2=1.0, count=4),
            AreaHistogramBucket(label=">=1", lower_m2=1.0, upper_m2=None, count=5),
        ),
        shape=ShapeSummary(
            polygon_count=21,
            multipolygon_count=22,
            unreadable_count=23,
            with_holes_count=24,
            total_rings=25,
            total_holes=26,
            total_vertices=27,
            vertices=_dist(100),
            rings=_dist(200),
            components=_dist(300),
        ),
        extent=ExtentSummary(
            dataset_min_lon=-1.5,
            dataset_min_lat=-2.5,
            dataset_max_lon=3.5,
            dataset_max_lat=4.5,
            width_deg=_dist(400),
            height_deg=_dist(500),
            width_m=_dist(600),
            height_m=_dist(700),
            wider_than_180_deg_count=31,
            pole_touching_count=32,
            unreadable_count=33,
        ),
        per_source=(
            SourceAreaSummary(
                source_pbf="monaco-latest",
                polygon_count=41,
                total_area_m2=42.0,
                median_area_m2=43.0,
                maximum_area_m2=44.0,
            ),
        ),
    )


EXPECTED = {
    "contract_version": "v-test",
    "source": {
        "table": "polygons",
        "column_scope": ["source_pbf", "area_m2", "bbox", "geometry"],
        "file_count": 2,
        "polygon_count": 3,
    },
    "area_m2": {
        "total": 1.0,
        "minimum": 2.0,
        "maximum": 3.0,
        "mean": 4.0,
        "median": 5.0,
        "p1": 6.0,
        "p5": 7.0,
        "p25": 8.0,
        "p75": 9.0,
        "p95": 10.0,
        "p99": 11.0,
        "non_positive_count": 12,
        "below_one_m2_count": 13,
        "null_count": 14,
    },
    "area_histogram": [
        {"label": "<1", "lower_m2": None, "upper_m2": 1.0, "count": 4},
        {"label": ">=1", "lower_m2": 1.0, "upper_m2": None, "count": 5},
    ],
    "geometry": {
        "polygon_count": 21,
        "multipolygon_count": 22,
        "unreadable_count": 23,
        "with_holes_count": 24,
        "total_rings": 25,
        "total_holes": 26,
        "total_vertices": 27,
        "vertices": _dist_json(100),
        "rings": _dist_json(200),
        "components": _dist_json(300),
    },
    "extent": {
        "dataset_bbox": [-1.5, -2.5, 3.5, 4.5],
        "width_deg": _dist_json(400),
        "height_deg": _dist_json(500),
        "width_m": _dist_json(600),
        "height_m": _dist_json(700),
        "wider_than_180_deg_count": 31,
        "pole_touching_count": 32,
        "unreadable_count": 33,
    },
    "per_source_pbf": [
        {
            "source_pbf": "monaco-latest",
            "polygon_count": 41,
            "total_area_m2": 42.0,
            "median_area_m2": 43.0,
            "maximum_area_m2": 44.0,
        }
    ],
}


def test_stats_payload_maps_every_field_exactly() -> None:
    assert stats_payload(_stats()) == EXPECTED


def test_stats_payload_is_deterministic_json() -> None:
    first = json.dumps(stats_payload(_stats()), sort_keys=True)
    second = json.dumps(stats_payload(_stats()), sort_keys=True)

    assert first == second
    assert json.loads(first) == EXPECTED


def test_empty_stats_payload_has_empty_collections() -> None:
    payload = stats_payload(PolygonGeometryStats())

    assert payload["area_histogram"] == []
    assert payload["per_source_pbf"] == []
    assert payload["extent"]["dataset_bbox"] == [0.0, 0.0, 0.0, 0.0]
