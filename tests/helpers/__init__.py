"""Shared helpers for the test suite."""

from __future__ import annotations

import hashlib
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.config.paths import DataRoot


def http_error(
    code: int,
    *,
    retry_after: str | None = None,
    url: str = "https://example.test",
    msg: str = "error",
) -> urllib.error.HTTPError:
    """Build an ``HTTPError`` with an optional ``Retry-After`` header."""
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, msg, headers, None)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensured_data_root(root: Path) -> DataRoot:
    """Return a :class:`DataRoot` at ``root`` with its directory layout created."""
    data_root = DataRoot(root)
    data_root.ensure()
    return data_root


def write_rows(
    path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema,
    **write_options: Any,
) -> None:
    """Write ``rows`` as a Parquet table at ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, **write_options)


def write_single_text_row(path: Path, schema: pa.Schema, text: str = "First.") -> None:
    """Write one all-null row of ``schema``, filling ``text`` when the column exists."""
    row: dict[str, object] = {field.name: None for field in schema}
    if "text" in schema.names:
        row["text"] = text
    write_rows(path, [row], schema)


__all__ = [
    "ensured_data_root",
    "http_error",
    "sha256_file",
    "write_rows",
    "write_single_text_row",
]
