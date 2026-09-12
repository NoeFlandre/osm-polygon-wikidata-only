"""Pure Wikidata QID validation and OSM-tag parsing."""

from __future__ import annotations

import re

_QID_PATTERN = re.compile(r"^Q[1-9]\d*$")


def is_valid_qid(qid: str) -> bool:
    """Return whether *qid* is a positive Wikidata entity identifier."""
    return bool(_QID_PATTERN.fullmatch(qid))


def qids_from_osm_tag(value: str) -> tuple[str, ...]:
    """Return the distinct QIDs in one OSM ``wikidata=*`` value.

    OSM uses semicolons for the uncommon case where a tag contains
    multiple values. Preserve their source order while trimming the
    whitespace permitted around separators. An empty or malformed
    component makes the complete value invalid rather than silently
    dropping source data.
    """
    components = tuple(component.strip() for component in value.split(";"))
    if any(not is_valid_qid(component) for component in components):
        return ()
    return tuple(dict.fromkeys(components))


__all__ = ["is_valid_qid", "qids_from_osm_tag"]
