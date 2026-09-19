"""Bounded, schema-aware Parquet readers shared by geographic inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet

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

    actual, metadata_read = _metadata_columns(parquet_path)
    try:
        yield from _iter_required_rows(
            parquet_path,
            columns,
            label=label,
            batch_size=batch_size,
            actual=actual,
            metadata_read=metadata_read,
        )
    except pa.ArrowInvalid as error:
        missing = sorted(set(columns) - actual)
        raise _missing_columns_error(label, parquet_path, missing) from error
    except KeyError as error:
        missing = sorted(set(columns) - actual)
        raise _missing_columns_error(label, parquet_path, missing) from error
    except OSError as error:
        raise CoverageMapError(f"Could not read {label} parquet {parquet_path}: {error}") from error


def _metadata_columns(parquet_path: Path) -> tuple[set[str], bool]:
    """Return user columns and whether metadata inspection succeeded."""
    try:
        metadata = pq.read_metadata(parquet_path)
    # ``except Exception`` retained: PyArrow's metadata API raises
    # across several unstable exception types depending on the
    # corruption mode. The ParquetFile schema remains the fallback.
    except Exception:
        return set(), False
    return set(metadata.schema.names) - PYARROW_INTERNAL_COLUMNS, True


def _missing_columns_error(label: str, parquet_path: Path, missing: list[str]) -> CoverageMapError:
    return CoverageMapError(
        f"{label} parquet {parquet_path} is missing required columns: {missing}"
    )


def _iter_required_rows(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
    batch_size: int,
    actual: set[str],
    metadata_read: bool,
) -> Iterator[dict[str, Any]]:
    with open_parquet(parquet_path) as parquet_file:
        if not metadata_read:
            actual.update(set(parquet_file.schema.names) - PYARROW_INTERNAL_COLUMNS)
        missing = sorted(set(columns) - actual)
        if missing:
            raise _missing_columns_error(label, parquet_path, missing)
        for batch in iter_record_batches(
            parquet_file,
            batch_size=batch_size,
            columns=list(columns),
        ):
            yield from batch.to_pylist()


def read_required_columns(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Read only ``columns`` from ``parquet_path`` as dicts.

    This compatibility wrapper retains the list-returning API. Production
    readers should use :func:`iter_required_columns` to bound peak memory.
    """
    return list(iter_required_columns(parquet_path, columns, label=label))


__all__ = [
    "PYARROW_INTERNAL_COLUMNS",
    "iter_required_columns",
    "read_required_columns",
    "sorted_parquets",
]
