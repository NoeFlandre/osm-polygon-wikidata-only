"""Polygon surface and geometry statistics: compute, render, publish.

This module is a thin compatibility facade. The implementation lives in
:mod:`osm_polygon_wikidata_only.hf._polygon_geometry` and the public
names below are re-exported unchanged.

The snapshot is computed from the published polygon table only --
every valid row of every ``polygons/<stem>.parquet`` file, with no
sampling, no truncation, and no external lookup -- and the processed
manifest is cross-checked so a stale or foreign input is refused
rather than silently summarised. :func:`render_polygon_geometry_stats`
turns the snapshot into the dataset-card block, and
:func:`write_polygon_stats_report` writes the machine-readable
``stats.json`` payload atomically.

Repeated calls for one unchanged polygon directory reuse the computed
snapshot: the card block and the JSON report are therefore always the
same numbers, produced by a single scan.
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.io.atomic import atomic_write_json

from ._polygon_geometry.aggregation import compute_polygon_geometry_stats
from ._polygon_geometry.codec import stats_payload
from ._polygon_geometry.models import PolygonGeometryStats
from ._polygon_geometry.rendering import render_polygon_geometry_stats
from ._polygon_geometry.validation import PolygonStatsInputError

# One-entry memo of the last scan, keyed by the polygon files' identity
# (name, size, mtime). A publication renders the card and writes the
# report from one scan, and a second publication over an unchanged
# polygon directory does no Parquet reads at all. Any change to any
# polygon file changes the key and forces a fresh scan.
_MEMO: dict[str, tuple[tuple[tuple[str, int, int], ...], PolygonGeometryStats]] = {}


def polygon_directory_fingerprint(processed_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Return the identity of the polygon files under ``processed_dir``."""
    polygons_dir = processed_dir / "polygons"
    if not polygons_dir.is_dir():
        return ()
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(polygons_dir.glob("*.parquet"))
    )


def load_polygon_geometry_stats(processed_dir: Path) -> PolygonGeometryStats:
    """Return the statistics snapshot, reusing the memo when it is current."""
    key = str(processed_dir)
    fingerprint = polygon_directory_fingerprint(processed_dir)
    cached = _MEMO.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    stats = compute_polygon_geometry_stats(processed_dir)
    _MEMO.clear()
    _MEMO[key] = (fingerprint, stats)
    return stats


def render_polygon_stats_section(processed_dir: Path) -> str:
    """Render the dataset-card block for the polygon table."""
    return render_polygon_geometry_stats(load_polygon_geometry_stats(processed_dir))


def write_polygon_stats_report(processed_dir: Path, destination: Path) -> Path:
    """Write the machine-readable ``stats.json`` report atomically."""
    atomic_write_json(destination, stats_payload(load_polygon_geometry_stats(processed_dir)))
    return destination


__all__ = [
    "PolygonGeometryStats",
    "PolygonStatsInputError",
    "compute_polygon_geometry_stats",
    "load_polygon_geometry_stats",
    "polygon_directory_fingerprint",
    "render_polygon_geometry_stats",
    "render_polygon_stats_section",
    "stats_payload",
    "write_polygon_stats_report",
]
