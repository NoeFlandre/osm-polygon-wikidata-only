"""Parquet scanning and column-pruned reads.

Owns sorted file enumeration, the warning logger used when an input
file is malformed or unreadable, and the PyArrow read primitives
called by the aggregation step.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")


def sorted_parquets(directory: Path) -> list[Path]:
    """Return the deterministic sorted list of parquet files in ``directory``.

    Returns an empty list when ``directory`` does not exist.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def safe_table(
    parquet_path: Path,
    columns: Iterable[str],
) -> pa.Table | None:
    """Read ``columns`` from ``parquet_path`` with the documented skip-on-error policy.

    Returns the table on success, or ``None`` after emitting the
    ``Skipping {path}: {error}`` warning when PyArrow raises
    :class:`OSError` or :class:`KeyError`. Any other exception
    propagates unchanged.
    """
    try:
        return pq.read_table(  # type: ignore[no-untyped-call]
            parquet_path,
            columns=list(columns),
        )
    except (OSError, KeyError) as e:
        LOGGER.warning("Skipping %s: %s", parquet_path, e)
        return None


def safe_metadata_row_count(parquet_path: Path) -> int | None:
    """Read the row count from parquet metadata with the same skip policy."""
    try:
        rows = pq.read_metadata(parquet_path).num_rows  # type: ignore[no-untyped-call]
    except (OSError, KeyError) as e:
        LOGGER.warning("Skipping %s: %s", parquet_path, e)
        return None
    return int(rows)
