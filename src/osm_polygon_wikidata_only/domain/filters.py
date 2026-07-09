"""Pure predicates for filtering OSM elements.

Kept as pure functions so they are easy to unit-test without any I/O.
"""

from __future__ import annotations

from collections.abc import Mapping


def has_wikidata(tags: Mapping[str, str]) -> bool:
    """True if the element has a non-empty ``wikidata`` tag.

    Whitespace-only values are treated as missing.
    """
    return bool(tags.get("wikidata", "").strip())


def is_polygon_relation(tags: Mapping[str, str]) -> bool:
    """True for relations of type ``multipolygon``."""
    return tags.get("type", "").strip() == "multipolygon"
