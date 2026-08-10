"""Atomic local file publication helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    """Write a Snappy-compressed Parquet table through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_tmp)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="snappy")  # type: ignore[no-untyped-call]
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["atomic_write_parquet", "atomic_write_text"]
