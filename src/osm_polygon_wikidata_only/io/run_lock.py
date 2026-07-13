"""Non-blocking process ownership for unified dataset synchronization."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


class RunLockError(RuntimeError):
    pass


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock for one sync process without waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream: IO[str] = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunLockError(f"Unified sync is already running ({path})") from error
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


__all__ = ["RunLockError", "exclusive_run_lock"]
