"""Shared Parquet reader settings for the batch-scanning code paths.

Every scanner in this project reads one file at a time, front to back, on
the calling thread. PyArrow's default reader does not: ``ParquetFile``
pre-buffers row groups through Arrow's IO thread pool, so each
``iter_batches`` step waits on a future fulfilled by a pool thread.

Under the long sequential scans the publication paths perform -- thousands
of files opened and closed in turn -- that hand-off deadlocks. A V2 release
was observed parked in ``arrow::FutureImpl::Wait`` for over an hour with
every pool thread idle, while a second process read the same files in
under a second.

Opening with ``pre_buffer=False`` and iterating with ``use_threads=False``
removes the hand-off: the read happens synchronously on the thread that
asked for it. A sequential scan of one file gains nothing from the pool
anyway, so this costs no real throughput.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BATCH_SIZE = 65_536


def open_parquet(path: Path | str) -> pq.ParquetFile:
    """Open ``path`` without Arrow's pre-buffering thread hand-off."""
    return pq.ParquetFile(path, pre_buffer=False)


def iter_record_batches(
    parquet_file: pq.ParquetFile,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    columns: Sequence[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Yield record batches read synchronously on the calling thread."""
    kwargs: dict[str, Any] = {"batch_size": batch_size, "use_threads": False}
    if columns is not None:
        kwargs["columns"] = list(columns)
    return parquet_file.iter_batches(**kwargs)


__all__ = ["DEFAULT_BATCH_SIZE", "iter_record_batches", "open_parquet"]
