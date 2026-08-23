"""Parquet I/O helpers for the geographic visualization.

This module owns the schema validation, batched reads, and column-
pruned I/O used by the aggregation step. No rendering or aggregation
logic lives here; only the file-loading primitives.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .h3_geometry import assign_h3_cell
from .models import CoverageMapError

# PyArrow metadata columns that are not part of the user schema.
PYARROW_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {"__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
)
_PARQUET_BATCH_SIZE = 65_536


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


def iter_required_columns(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
    batch_size: int = _PARQUET_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Stream only ``columns`` from ``parquet_path`` as dicts.

    Raises :class:`CoverageMapError` when the parquet file is missing
    required columns or is unreadable. The error message identifies
    the source file and the offending columns.
    """
    import pyarrow as pa

    actual: set[str] = set()
    metadata_read = False
    try:
        metadata = pq.read_metadata(parquet_path)  # type: ignore[no-untyped-call]
        actual = set(metadata.schema.names) - PYARROW_INTERNAL_COLUMNS
        metadata_read = True
    # ``except Exception`` retained: PyArrow's metadata API raises
    # across several unstable exception types depending on the
    # corruption mode. When the metadata read fails, we fall through
    # with an empty ``actual`` column-name set and let the
    # ``ParquetFile`` schema determine the outcome. A valid parquet
    # with the requested columns still streams; missing columns are
    # translated into ``CoverageMapError``. See
    # ``tests/hf/test_geographic_text_coverage.py`` for the focused
    # schema-introspection tests.
    except Exception:
        actual = set()
    try:
        with pq.ParquetFile(parquet_path) as parquet_file:  # type: ignore[no-untyped-call]
            if not metadata_read:
                actual = set(parquet_file.schema.names) - PYARROW_INTERNAL_COLUMNS
            missing = sorted(set(columns) - actual)
            if missing:
                raise CoverageMapError(
                    f"{label} parquet {parquet_path} is missing required columns: {missing}"
                )
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=list(columns),
            ):
                yield from batch.to_pylist()
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


def read_required_columns(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Read only ``columns`` from ``parquet_path`` as a list of dicts.

    This compatibility wrapper retains the list-returning API. Production
    readers should use :func:`iter_required_columns` to bound peak memory.
    """
    return list(iter_required_columns(parquet_path, columns, label=label))


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
    rows: list[tuple[str, str]] = []
    for parquet_path in sorted_parquets(polygons_dir):
        rows.extend(_polygon_cells_from_file(parquet_path, h3_resolution))
    rows.sort(key=lambda pair: pair[0])
    return rows


def _polygon_cells_from_file(path: Path, h3_resolution: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row_index, row in enumerate(
        iter_required_columns(path, ("polygon_id", "lat", "lon"), label="polygons")
    ):
        rows.append(_polygon_cell(path, row_index, row, h3_resolution))
    return rows


def _polygon_cell(
    path: Path, row_index: int, row: dict[str, Any], h3_resolution: int
) -> tuple[str, str]:
    polygon_id = row.get("polygon_id")
    lat = row.get("lat")
    lon = row.get("lon")
    if not polygon_id:
        raise CoverageMapError(
            f"polygons parquet {path} row {row_index} is missing polygon_id; "
            "cannot include it in the visualization denominator."
        )
    if lat is None or lon is None:
        raise CoverageMapError(
            f"polygons parquet {path} row {row_index} (polygon_id={polygon_id}) "
            "has null lat or lon; cannot include it in the visualization denominator."
        )
    try:
        cell = assign_h3_cell(lat, lon, resolution=h3_resolution)
    except CoverageMapError as error:
        raise CoverageMapError(
            f"polygons parquet {path} row {row_index} (polygon_id={polygon_id}) "
            f"has invalid coordinates (lat={lat}, lon={lon}): {error}"
        ) from error
    return str(polygon_id), cell
