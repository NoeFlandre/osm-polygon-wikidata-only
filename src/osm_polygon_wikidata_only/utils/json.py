"""Deterministic JSON serialization helpers.

Centralizing these here means the whole pipeline produces byte-stable
strings for ``tags``, ``tag_keys``, ``bbox``, ``wikipedia_languages``,
``wikidata_aliases``, ``categories``, etc. — which is required for
content hashing and reproducible builds.
"""

from __future__ import annotations

import json
from typing import Any


def dumps(value: Any) -> str:
    """Deterministic JSON: sorted keys, UTF-8 preserved, no trailing spaces.

    The output is byte-stable for the same input across runs and
    platforms. ``ensure_ascii=False`` keeps non-ASCII characters as-is,
    which is more compact and more readable; we explicitly do *not*
    escape ``/`` because that is allowed in JSON.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(s: str | bytes) -> Any:
    """Tiny wrapper to keep callers symmetric with :func:`dumps`."""
    return json.loads(s)


def dumps_compact_list(values: list[str]) -> str:
    """Sort + dedup + JSON-encode a list of strings."""
    return dumps(sorted({v for v in values if v}))
