"""Shared helpers for the test suite."""

from __future__ import annotations

import hashlib
import urllib.error
from email.message import Message
from pathlib import Path


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


__all__ = ["http_error", "sha256_file"]
