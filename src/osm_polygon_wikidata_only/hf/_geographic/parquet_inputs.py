"""Parquet I/O helpers for the geographic visualization.

This module owns the schema validation, batched reads, and column-
pruned I/O used by the aggregation step. No rendering or aggregation
logic lives here; only the file-loading primitives.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq  # noqa: F401 (preserved private compatibility surface)

from .h3_geometry import assign_h3_cell
from .models import CoverageMapError
from .parquet_reader import (
    _metadata_columns,  # noqa: F401 (preserved private compatibility surface)
    _missing_columns_error,  # noqa: F401 (preserved private compatibility surface)
    iter_required_columns,
    read_required_columns,  # noqa: F401 (preserved public compatibility surface)
    sorted_parquets,
)


def require_directory(path: Path, *, label: str) -> Path:
    """Return ``path`` after asserting it exists and is a directory."""
    if not path.exists() or not path.is_dir():
        raise CoverageMapError(
            f"Required {label} directory does not exist: {path}. "
            f"Run a complete PBF processing pass first."
        )
    return path


def load_qualifying_article_ids(articles_dir: Path) -> set[str]:
    """Return the set of article IDs whose ``full_text`` is non-empty and non-whitespace."""
    qualifying: set[str] = set()
    for parquet_path in sorted_parquets(articles_dir):
        qualifying.update(_qualifying_ids_from_file(parquet_path))
    return qualifying


def _qualifying_ids_from_file(parquet_path: Path) -> set[str]:
    qualifying: set[str] = set()
    for row in iter_required_columns(parquet_path, ("article_id", "full_text"), label="articles"):
        text = row.get("full_text")
        article_id = row.get("article_id")
        if isinstance(text, str) and text.strip() and article_id:
            qualifying.add(str(article_id))
    return qualifying


def load_covered_polygon_ids(
    links_dir: Path,
    qualifying_article_ids: set[str],
) -> set[str]:
    """Return the set of polygon IDs linked to at least one qualifying article."""
    covered: set[str] = set()
    for parquet_path in sorted_parquets(links_dir):
        covered.update(_covered_ids_from_file(parquet_path, qualifying_article_ids))
    return covered


def _covered_ids_from_file(path: Path, qualifying_article_ids: set[str]) -> set[str]:
    covered: set[str] = set()
    for row in iter_required_columns(path, ("polygon_id", "article_id"), label="polygon_articles"):
        article_id = row.get("article_id")
        polygon_id = row.get("polygon_id")
        if article_id is not None and str(article_id) in qualifying_article_ids and polygon_id:
            covered.add(str(polygon_id))
    return covered


def load_polygon_cells(
    polygons_dir: Path,
    *,
    h3_resolution: int,
) -> list[tuple[str, str]]:
    """Return a sorted list of ``(polygon_id, h3_cell)`` tuples.

    Every polygon row in ``polygons/*.parquet`` must contribute to the
    denominator, so missing, null, non-finite, or out-of-range
    coordinates are never silently skipped. Invalid rows raise
    :class:`CoverageMapError` identifying the source parquet path and
    the offending polygon id so the operator can fix the data.
    """
    parquet_paths = sorted_parquets(polygons_dir)
    return _load_unique_polygon_cells(parquet_paths, h3_resolution)


def _load_unique_polygon_cells(paths: list[Path], h3_resolution: int) -> list[tuple[str, str]]:
    from .polygon_identities import load_unique_polygon_records

    index = load_unique_polygon_records(paths)
    rows: list[tuple[str, str]] = []
    for _identity, record in sorted(
        index.records.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        if record.lat is None or record.lon is None:
            raise CoverageMapError(
                f"polygons parquet {record.source_path} has invalid lat/lon coordinates "
                f"for {record.polygon_id}"
            )
        try:
            cell = assign_h3_cell(record.lat, record.lon, resolution=h3_resolution)
        except CoverageMapError as error:
            raise CoverageMapError(
                f"polygons parquet (polygon_id={record.polygon_id}) has invalid coordinates "
                f"(lat={record.lat}, lon={record.lon}): {error}"
            ) from error
        rows.append((record.polygon_id, cell))
    return rows
