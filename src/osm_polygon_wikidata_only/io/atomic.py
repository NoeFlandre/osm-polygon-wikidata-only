"""Atomic local file publication.

Responsibility:
    Own the single durable-publication ritual used across the package:
    create a hidden temporary sibling inside the destination directory,
    let a caller fill it, and install it over the target with one
    :func:`os.replace`. A reader of the target path therefore sees either
    the previous content or the complete new content, never a partial
    write, and a failure never leaves the temporary sibling behind.

    :func:`atomic_replacement` is the mechanism; the writers below are
    thin payload-specific spellings of it. Callers should not name
    :mod:`tempfile`, :func:`os.replace`, or cleanup themselves.

Out of scope (intentionally retained elsewhere):
    * Multi-file transactions whose members must roll back together (see
      :mod:`pipeline.persistence` and :mod:`v2.storage`).
    * Whole-directory publication, which stages a temporary directory and
      replaces a tree (see :mod:`augmentation.checkpoints` and
      :mod:`pipeline._wikidata_recovery.checkpoints`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.utils.json import dumps as json_dumps


@contextmanager
def atomic_replacement(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling that becomes ``path`` when the block ends.

    The sibling is created inside ``path.parent`` -- creating that
    directory first when it is missing -- so the final rename stays on one
    filesystem and is therefore atomic. Its name is hidden and suffixed
    ``.tmp`` so an interrupted run leaves nothing a directory scan would
    mistake for a published artifact.

    Any exception propagates with ``path`` untouched and the sibling
    removed. The cleanup deliberately catches :class:`BaseException` so a
    :class:`KeyboardInterrupt` or :class:`SystemExit` mid-write is cleaned
    up too, and tolerates a body that already consumed the sibling itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        yield temporary
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Publish ``text`` at ``path``, flushed to disk before it is installed."""
    with atomic_replacement(path) as temporary, temporary.open("w", encoding=encoding) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_json(path: Path, value: Any) -> None:
    """Publish ``value`` in the pipeline's deterministic JSON encoding.

    The encoding is :func:`utils.json.dumps` plus a trailing newline, which
    is byte-stable across runs and platforms and is what every checkpoint,
    ledger, and manifest reader in the package expects.
    """
    atomic_write_text(path, json_dumps(value) + "\n")


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    """Publish ``table`` at ``path`` as a Snappy-compressed Parquet file."""
    with atomic_replacement(path) as temporary:
        pq.write_table(table, temporary, compression="snappy")  # type: ignore[no-untyped-call]


def atomic_copy_file(source: Path, target: Path) -> None:
    """Copy ``source`` onto ``target`` without ever exposing a partial file."""
    with (
        atomic_replacement(target) as temporary,
        source.open("rb") as reader,
        temporary.open("wb") as writer,
    ):
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


__all__ = [
    "atomic_copy_file",
    "atomic_replacement",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
]
