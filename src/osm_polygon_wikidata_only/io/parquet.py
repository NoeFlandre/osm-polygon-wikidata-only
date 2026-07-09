"""Parquet I/O for the three dataset tables.

Each table is written as a single ``.parquet`` file using ``pyarrow``.
The schema is fixed and lives in :mod:`domain.schema`, so the writer
and the reader can both be schema-driven.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    article_schema,
    empty_row,
    polygon_article_schema,
    polygon_schema,
)

LOGGER = logging.getLogger(__name__)


def _fill_columns(
    rows: Iterable[dict[str, object]], columns: tuple[str, ...]
) -> list[dict[str, object]]:
    """Materialize rows into a list, ensuring every column is present."""
    out: list[dict[str, object]] = []
    for row in rows:
        # Make sure every column is present (None / default if missing).
        normalized = {col: row.get(col) for col in columns}
        out.append(normalized)
    return out


def write_table(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: tuple[str, ...],
    schema: pa.Schema,
) -> int:
    """Write ``rows`` to ``path`` using ``schema`` and return the row count.

    Empty input is allowed: we write a single placeholder row with
    default values, then immediately drop it so the resulting parquet
    file has zero data rows but the correct schema. This way
    ``datasets.load_dataset`` and friends always see a typed schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = _fill_columns(rows, columns)
    if not materialized:
        placeholder = empty_row(columns)
        table = pa.Table.from_pylist([placeholder], schema=schema)
        table = table.slice(0, 0)
    else:
        table = pa.Table.from_pylist(list(materialized), schema=schema)
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]
    LOGGER.info("Wrote %d rows to %s", len(materialized), path)
    return len(materialized)


def read_table(path: Path) -> pa.Table:
    """Read a parquet file at ``path`` and return a :class:`pa.Table`."""
    result: pa.Table = pq.read_table(path)  # type: ignore[no-untyped-call]
    return result


def write_polygons(path: Path, rows: Iterable[dict[str, object]]) -> int:
    return write_table(path, rows, columns=POLYGON_COLUMNS, schema=polygon_schema())


def write_articles(path: Path, rows: Iterable[dict[str, object]]) -> int:
    return write_table(path, rows, columns=ARTICLE_COLUMNS, schema=article_schema())


def write_polygon_articles(path: Path, rows: Iterable[dict[str, object]]) -> int:
    return write_table(path, rows, columns=POLYGON_ARTICLE_COLUMNS, schema=polygon_article_schema())


__all__ = [
    "read_table",
    "write_articles",
    "write_polygon_articles",
    "write_polygons",
    "write_table",
]
