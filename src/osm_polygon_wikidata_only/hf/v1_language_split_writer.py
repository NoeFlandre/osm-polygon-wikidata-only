"""Buffered Parquet writers for V1 language-partition artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_PARQUET_COMPRESSION = "snappy"
_SHARD_FLUSH_BYTES = 16 * 1024 * 1024


class V1LanguageSplitError(ValueError):
    """Raised when a V1 partition release cannot be completed safely."""


@dataclass(slots=True)
class BufferedWriter:
    """Buffer language slices until a row group is worth writing.

    Source batches fan out across languages. Buffering those slices avoids
    interleaving tiny writes across many open partition files.
    """

    writer: Any
    pending: list[pa.RecordBatch] = field(default_factory=list)
    pending_bytes: int = 0

    def flush(self) -> None:
        """Write buffered batches as one row group."""
        if not self.pending:
            return
        self.writer.write_table(pa.Table.from_batches(self.pending))
        self.pending = []
        self.pending_bytes = 0

    def close(self) -> None:
        """Flush pending batches, then close the Parquet writer."""
        self.flush()
        self.writer.close()


def buffer_rows(writer: BufferedWriter, rows: pa.RecordBatch) -> None:
    """Buffer one row slice and flush it when the byte threshold is reached."""
    writer.pending.append(rows)
    writer.pending_bytes += rows.nbytes
    if writer.pending_bytes >= _SHARD_FLUSH_BYTES:
        writer.flush()


def new_parquet_writer(path: Path, schema: pa.Schema) -> Any:
    """Open a deterministic compressed V1 language partition writer."""
    return pq.ParquetWriter(
        str(path),
        schema,
        compression=_PARQUET_COMPRESSION,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=True,
        write_statistics=True,
    )


def close_writers(writers: dict[str, BufferedWriter]) -> None:
    """Close every open language writer in stable language order."""
    for language in sorted(writers):
        writers[language].close()


def validate_staged_schema(
    path: Path,
    schema: pa.Schema,
    table: str,
    language: str,
) -> None:
    """Ensure a staged partition retains the source schema and metadata."""
    actual = _read_staged_schema(path)
    if not actual.equals(schema, check_metadata=True):
        raise V1LanguageSplitError(
            f"generated {table} {language} artifact has a schema mismatch: {path}"
        )


def _read_staged_schema(path: Path) -> pa.Schema:
    try:
        return pq.read_schema(path)
    except Exception as error:
        raise V1LanguageSplitError(
            f"Could not validate generated artifact {path}: {error}"
        ) from error
