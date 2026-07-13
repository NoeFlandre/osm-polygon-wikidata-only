"""Parquet I/O helpers for the geographic visualization.

This module owns the schema validation, batched reads, and column-
pruned I/O used by the aggregation step. No rendering or aggregation
logic lives here; only the file-loading primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .h3_geometry import assign_h3_cell
from .models import CoverageMapError

# PyArrow metadata columns that are not part of the user schema.
PYARROW_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {"__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
)


def sorted_parquets(directory: Path) -> list[Path]:
    """Return the deterministic sorted list of parquet files in ``directory``.

    Returns an empty list if the directory does not exist.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def require_directory(path: Path, *, label: str) -> Path:
    """Return ``path`` after asserting it exists and is a directory."""
    if not path.exists() or not path.is_dir():
        raise CoverageMapError(
            f"Required {label} directory does not exist: {path}. "
            f"Run a complete PBF processing pass first."
        )
    return path


def read_required_columns(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Read only ``columns`` from ``parquet_path`` as a list of dicts.

    Raises :class:`CoverageMapError` when the parquet file is missing
    required columns or is unreadable. The error message identifies
    the source file and the offending columns.
    """
    import pyarrow as pa

    actual: set[str] = set()
    try:
        metadata = pq.read_metadata(parquet_path)  # type: ignore[no-untyped-call]
        actual = set(metadata.schema.names) - PYARROW_INTERNAL_COLUMNS
    # ``except Exception`` retained: PyArrow's metadata API raises
    # across several unstable exception types depending on the
    # corruption mode. When the metadata read fails, we fall through
    # with an empty ``actual`` column-name set and let the
    # column-pruned ``pq.read_table`` call determine the outcome:
    # a valid parquet with the requested columns still loads; missing
    # columns are translated into ``CoverageMapError``. See
    # ``tests/hf/test_geographic_text_coverage.py`` for the focused
    # schema-introspection tests.
    except Exception:
        actual = set()
    try:
        table = pq.read_table(parquet_path, columns=list(columns))  # type: ignore[no-untyped-call]
    except pa.ArrowInvalid as error:
        missing = sorted(set(columns) - actual)
        raise CoverageMapError(
            f"{label} parquet {parquet_path} is missing required columns: {missing}"
        ) from error
    except KeyError as error:
        missing = sorted(set(columns) - actual)
        raise CoverageMapError(
            f"{label} parquet {parquet_path} is missing required columns: {missing}"
        ) from error
    except OSError as error:
        raise CoverageMapError(f"Could not read {label} parquet {parquet_path}: {error}") from error
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def load_qualifying_article_ids(articles_dir: Path) -> set[str]:
    """Return the set of article IDs whose ``full_text`` is non-empty and non-whitespace."""
    qualifying: set[str] = set()
    for parquet_path in sorted_parquets(articles_dir):
        for row in read_required_columns(
            parquet_path, ("article_id", "full_text"), label="articles"
        ):
            text = row.get("full_text")
            if text is None:
                continue
            if not isinstance(text, str):
                continue
            if not text.strip():
                continue
            article_id = row.get("article_id")
            if article_id:
                qualifying.add(str(article_id))
    return qualifying


def load_covered_polygon_ids(
    links_dir: Path,
    qualifying_article_ids: set[str],
) -> set[str]:
    """Return the set of polygon IDs linked to at least one qualifying article."""
    covered: set[str] = set()
    for parquet_path in sorted_parquets(links_dir):
        for row in read_required_columns(
            parquet_path, ("polygon_id", "article_id"), label="polygon_articles"
        ):
            article_id = row.get("article_id")
            if article_id is None:
                continue
            if str(article_id) not in qualifying_article_ids:
                continue
            polygon_id = row.get("polygon_id")
            if polygon_id:
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
    rows: list[tuple[str, str]] = []
    for parquet_path in sorted_parquets(polygons_dir):
        table_rows = read_required_columns(
            parquet_path, ("polygon_id", "lat", "lon"), label="polygons"
        )
        for row_index, row in enumerate(table_rows):
            polygon_id = row.get("polygon_id")
            lat = row.get("lat")
            lon = row.get("lon")
            if not polygon_id:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} is missing "
                    f"polygon_id; cannot include it in the visualization denominator."
                )
            if lat is None or lon is None:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} (polygon_id="
                    f"{polygon_id}) has null lat or lon; cannot include it in the "
                    f"visualization denominator."
                )
            try:
                cell = assign_h3_cell(lat, lon, resolution=h3_resolution)
            except CoverageMapError as error:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} (polygon_id="
                    f"{polygon_id}) has invalid coordinates (lat={lat}, lon={lon}): "
                    f"{error}"
                ) from error
            rows.append((str(polygon_id), cell))
    rows.sort(key=lambda pair: pair[0])
    return rows
