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
    is not valid UTF-8, or is not valid JSON yields ``None``. With it, invalid
    JSON raises ``on_malformed(error)`` chained to the parser error, while an
    unreadable file propagates its own exception. The result is not checked
    for shape; callers that need a JSON object check it themselves.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        if on_malformed is None:
            return None
        raise
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        if on_malformed is None:
            return None
        raise on_malformed(error) from error


__all__ = ["read_json"]
