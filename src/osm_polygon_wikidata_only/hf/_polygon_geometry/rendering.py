"""Markdown rendering of the polygon surface and geometry statistics.

The block is appended to the dataset card. It is side-effect free and
byte-stable: every number comes from the supplied
:class:`~.models.PolygonGeometryStats`, every float is already rounded
by :mod:`.summaries`, and the formatters below use fixed precision.

The card carries the headline figures only. The complete report,
including the per-``source_pbf`` breakdown, is published next to the
card as ``stats.json``.
"""

from __future__ import annotations

from .models import Distribution, PolygonGeometryStats

_SQUARE_METRES_PER_SQUARE_KILOMETRE = 1_000_000.0


def render_polygon_geometry_stats(stats: PolygonGeometryStats) -> str:
    """Render the dataset-card block for one statistics snapshot."""
    parts = [
        "## Polygon surface and geometry\n",
        _render_preamble(stats),
    ]
    if stats.polygon_count == 0:
        return "\n".join([*parts, "No polygon rows are published yet.\n"])
    parts.extend(
        [
            _render_surface_table(stats),
            _render_shape_table(stats),
            _render_extent_table(stats),
            _render_histogram_table(stats),
            _render_definitions(),
        ]
    )
    return "\n".join(parts)


def _render_preamble(stats: PolygonGeometryStats) -> str:
    return (
        "Recomputed from every row of the published `polygons/*.parquet` table before each "
        "dataset-card publication: all "
        f"{_plural(stats.polygon_count, 'row')} across "
        f"{_plural(stats.file_count, 'regional file')}, with no "
        "sampling and no other input. Areas are the table's own `area_m2` values (WGS84 "
        "equirectangular approximation, holes subtracted); shape and extent figures are decoded "
        "from the `geometry` and `bbox` columns. The complete machine-readable report, including "
        "the per-source breakdown, is published as "
        "[`stats.json`](stats.json).\n"
    )


def _render_surface_table(stats: PolygonGeometryStats) -> str:
    area = stats.area
    rows = [
        ("Total surface", _area(area.total_m2)),
        ("Smallest", _area(area.minimum_m2)),
        ("1st percentile", _area(area.p1_m2)),
        ("5th percentile", _area(area.p5_m2)),
        ("25th percentile", _area(area.p25_m2)),
        ("Median", _area(area.median_m2)),
        ("Mean", _area(area.mean_m2)),
        ("75th percentile", _area(area.p75_m2)),
        ("95th percentile", _area(area.p95_m2)),
        ("99th percentile", _area(area.p99_m2)),
        ("Largest", _area(area.maximum_m2)),
        ("Degenerate (`area_m2` <= 0)", _plural(area.non_positive_count, "polygon")),
        ("Below 1 m2", _plural(area.below_one_m2_count, "polygon")),
        ("Missing `area_m2`", _plural(area.null_count, "polygon")),
    ]
    return _table("Surface (`area_m2`)", "Value", rows)


def _render_shape_table(stats: PolygonGeometryStats) -> str:
    shape = stats.shape
    rows = [
        ("`Polygon` rows", f"{shape.polygon_count:,}"),
        ("`MultiPolygon` rows", f"{shape.multipolygon_count:,}"),
        ("Rows with holes", f"{shape.with_holes_count:,}"),
        ("Undecodable `geometry`", f"{shape.unreadable_count:,}"),
        ("Total rings", f"{shape.total_rings:,}"),
        ("Total holes", f"{shape.total_holes:,}"),
        ("Total vertices", f"{shape.total_vertices:,}"),
        ("Vertices per polygon (median / p95 / max)", _counts(shape.vertices)),
        ("Rings per polygon (median / p95 / max)", _counts(shape.rings)),
        ("Components per polygon (median / p95 / max)", _counts(shape.components)),
    ]
    return _table("Geometry shape", "Value", rows)


def _render_extent_table(stats: PolygonGeometryStats) -> str:
    extent = stats.extent
    bbox = (
        f"`[{extent.dataset_min_lon:.6f}, {extent.dataset_min_lat:.6f}, "
        f"{extent.dataset_max_lon:.6f}, {extent.dataset_max_lat:.6f}]`"
    )
    rows = [
        ("Dataset bounding box", bbox),
        ("Bbox width in degrees (median / p95 / max)", _degrees(extent.width_deg)),
        ("Bbox height in degrees (median / p95 / max)", _degrees(extent.height_deg)),
        ("Bbox width in metres (median / p95 / max)", _metres(extent.width_m)),
        ("Bbox height in metres (median / p95 / max)", _metres(extent.height_m)),
        ("Bbox wider than 180 degrees", _plural(extent.wider_than_180_deg_count, "polygon")),
        ("Bbox touching a pole", _plural(extent.pole_touching_count, "polygon")),
        ("Undecodable `bbox`", _plural(extent.unreadable_count, "polygon")),
    ]
    return _table("Extent", "Value", rows)


def _render_histogram_table(stats: PolygonGeometryStats) -> str:
    rows = [(f"`{bucket.label}`", f"{bucket.count:,}") for bucket in stats.area_histogram]
    return _table("Area bucket (log scale)", "Polygons", rows)


def _render_definitions() -> str:
    return (
        "**Field definitions:**\n\n"
        "- `Total surface`: sum of `area_m2` over every published polygon row. Overlapping "
        "polygons are summed independently, so this is the total polygon surface, not the "
        "distinct land area covered.\n"
        "- Percentiles and medians interpolate linearly between the two closest ranks.\n"
        "- `Degenerate`: rows whose recorded `area_m2` is zero or negative. `Below 1 m2` counts "
        "rows with a positive area smaller than one square metre.\n"
        "- A vertex is one coordinate pair; each ring's repeated closing coordinate is counted "
        "once. A hole is an inner ring, and `Components` is the number of `MultiPolygon` "
        "members, which is `1` for every `Polygon` row.\n"
        "- Metre spans convert the degree spans with the same equirectangular rule used for "
        "`area_m2`, evaluated at each bounding box's mean latitude.\n"
        "- `Bbox wider than 180 degrees` is the antimeridian signature: the box spans more than "
        "half the globe because the polygon crosses longitude 180.\n"
        "- Area buckets are half-open decade intervals `[lower, upper)` in square metres; the "
        "bucket list is identical in every report so two reports stay comparable.\n"
    )


def _plural(count: int, noun: str) -> str:
    """Return ``count`` and ``noun``, pluralised with a trailing ``s``."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _table(header: str, value_header: str, rows: list[tuple[str, str]]) -> str:
    lines = [f"| {header} | {value_header} |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines) + "\n"


def _area(value: float) -> str:
    square_kilometres = value / _SQUARE_METRES_PER_SQUARE_KILOMETRE
    return f"{value:,.2f} m2 ({square_kilometres:,.4f} km2)"


def _counts(distribution: Distribution) -> str:
    return f"{distribution.median:,.1f} / {distribution.p95:,.1f} / {distribution.maximum:,.0f}"


def _degrees(distribution: Distribution) -> str:
    return f"{distribution.median:.6f} / {distribution.p95:.6f} / {distribution.maximum:.6f}"


def _metres(distribution: Distribution) -> str:
    return f"{distribution.median:,.2f} / {distribution.p95:,.2f} / {distribution.maximum:,.2f}"


__all__ = ["render_polygon_geometry_stats"]
