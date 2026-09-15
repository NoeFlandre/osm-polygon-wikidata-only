"""Deterministic JSON payload for the published statistics report.

The payload is a plain, sorted-key JSON object written through
:func:`osm_polygon_wikidata_only.io.atomic.atomic_write_json`, so an
unchanged dataset produces a byte-identical ``stats.json``. Every key
documented in the dataset card and in ``docs/architecture.md`` is
produced here, and nothing else.
"""

from __future__ import annotations

from typing import Any

from .models import (
    AreaHistogramBucket,
    AreaSummary,
    Distribution,
    ExtentSummary,
    PolygonGeometryStats,
    ShapeSummary,
    SourceAreaSummary,
)


def stats_payload(stats: PolygonGeometryStats) -> dict[str, Any]:
    """Return the machine-readable statistics report as a JSON object."""
    return {
        "contract_version": stats.contract_version,
        "source": {
            "table": "polygons",
            "column_scope": ["source_pbf", "area_m2", "bbox", "geometry"],
            "file_count": stats.file_count,
            "polygon_count": stats.polygon_count,
        },
        "area_m2": _area_payload(stats.area),
        "area_histogram": [_bucket_payload(bucket) for bucket in stats.area_histogram],
        "geometry": _shape_payload(stats.shape),
        "extent": _extent_payload(stats.extent),
        "per_source_pbf": [_source_payload(source) for source in stats.per_source],
    }


def _area_payload(area: AreaSummary) -> dict[str, Any]:
    return {
        "total": area.total_m2,
        "minimum": area.minimum_m2,
        "maximum": area.maximum_m2,
        "mean": area.mean_m2,
        "median": area.median_m2,
        "p1": area.p1_m2,
        "p5": area.p5_m2,
        "p25": area.p25_m2,
        "p75": area.p75_m2,
        "p95": area.p95_m2,
        "p99": area.p99_m2,
        "non_positive_count": area.non_positive_count,
        "below_one_m2_count": area.below_one_m2_count,
        "null_count": area.null_count,
    }


def _bucket_payload(bucket: AreaHistogramBucket) -> dict[str, Any]:
    return {
        "label": bucket.label,
        "lower_m2": bucket.lower_m2,
        "upper_m2": bucket.upper_m2,
        "count": bucket.count,
    }


def _shape_payload(shape: ShapeSummary) -> dict[str, Any]:
    return {
        "polygon_count": shape.polygon_count,
        "multipolygon_count": shape.multipolygon_count,
        "unreadable_count": shape.unreadable_count,
        "with_holes_count": shape.with_holes_count,
        "total_rings": shape.total_rings,
        "total_holes": shape.total_holes,
        "total_vertices": shape.total_vertices,
        "vertices": _distribution_payload(shape.vertices),
        "rings": _distribution_payload(shape.rings),
        "components": _distribution_payload(shape.components),
    }


def _extent_payload(extent: ExtentSummary) -> dict[str, Any]:
    return {
        "dataset_bbox": [
            extent.dataset_min_lon,
            extent.dataset_min_lat,
            extent.dataset_max_lon,
            extent.dataset_max_lat,
        ],
        "width_deg": _distribution_payload(extent.width_deg),
        "height_deg": _distribution_payload(extent.height_deg),
        "width_m": _distribution_payload(extent.width_m),
        "height_m": _distribution_payload(extent.height_m),
        "wider_than_180_deg_count": extent.wider_than_180_deg_count,
        "pole_touching_count": extent.pole_touching_count,
        "unreadable_count": extent.unreadable_count,
    }


def _distribution_payload(distribution: Distribution) -> dict[str, float]:
    return {
        "minimum": distribution.minimum,
        "maximum": distribution.maximum,
        "mean": distribution.mean,
        "median": distribution.median,
        "p95": distribution.p95,
        "p99": distribution.p99,
    }


def _source_payload(source: SourceAreaSummary) -> dict[str, Any]:
    return {
        "source_pbf": source.source_pbf,
        "polygon_count": source.polygon_count,
        "total_area_m2": source.total_area_m2,
        "median_area_m2": source.median_area_m2,
        "maximum_area_m2": source.maximum_area_m2,
    }


__all__ = ["stats_payload"]
