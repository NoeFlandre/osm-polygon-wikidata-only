"""Read JSON documents from disk under an explicit error policy."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def read_json(
    path: Path,
    *,
    on_malformed: Callable[[json.JSONDecodeError], Exception] | None = None,
) -> object:
    """Parse the UTF-8 JSON document stored at ``path``.

    Without ``on_malformed`` the read is lenient: a file that cannot be read,
    is not valid UTF-8, or does not parse yields ``None``. That includes an
    integer past Python's digit limit, which the parser reports as a plain
    ``ValueError`` rather than a ``JSONDecodeError``. With ``on_malformed``,
    invalid JSON raises ``on_malformed(error)`` chained to the parser error,
    while an unreadable file and the digit-limit ``ValueError`` propagate
    unchanged. The result is not checked for shape; callers that need a JSON
    object check it themselves.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        if on_malformed is None:
            return None
        raise
    try:
        return json.loads(text)
    except ValueError as error:
        _reject_unparsed(error, on_malformed)
        return None


def _reject_unparsed(
    error: ValueError,
    on_malformed: Callable[[json.JSONDecodeError], Exception] | None,
) -> None:
    """Apply the error policy to text the parser rejected."""
    if on_malformed is None:
        return
    if isinstance(error, json.JSONDecodeError):
        raise on_malformed(error) from error
    raise error


__all__ = ["read_json"]
