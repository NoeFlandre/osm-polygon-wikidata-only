"""Streaming content-hash helpers for local artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

_SHA256_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["sha256_file"]
