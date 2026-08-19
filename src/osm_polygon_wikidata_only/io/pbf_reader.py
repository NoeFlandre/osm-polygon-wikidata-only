"""Streaming reader for OSM PBF files.

The reader is wrapped around an osmium SimpleHandler. It invokes a
caller-supplied callback for every way/relation that:

* carries a non-empty ``wikidata`` tag, **and**
* is a polygonal candidate (closed way, or relation of ``type=multipolygon``).

Callers building the V2 Wikipedia-tag dataset may opt in to retaining
polygonal elements that have no Wikidata tag but do have a Wikipedia tag.
The default remains the V1 Wikidata-only filter.

Nodes and non-polygonal elements never reach the callback, so memory
stays bounded even for planet-sized files.

This module deliberately does not import the geometry-computation code;
the callback receives the raw osmium element plus a precomputed
GeoJSON geometry string and decides what to do with it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import osmium
import osmium.geom
import osmium.osm

_REGION_RE = re.compile(r"^(?P<region>.+)-latest\.osm\.pbf$")


# The callback receives the (osm_type, osm_id, tags_dict, geom_json) tuple
# for every retained polygonal element. ``geom_json`` is a GeoJSON string.
PolygonCandidate = tuple[str, int, dict[str, str], str]
"""A polygon candidate yielded by :class:`PBFReader`."""

Callback = Callable[[PolygonCandidate], None]
"""Callback signature for retained polygonal elements."""


class PBFReadError(RuntimeError):
    """Raised when a PBF file cannot be read (corrupt, missing, unsupported)."""


def region_from_filename(pbf_path: str | Path) -> str:
    """Extract the region slug from a Geofabrik-style filename."""
    name = Path(pbf_path).name
    match = _REGION_RE.match(name)
    if not match:
        raise ValueError(
            f"Filename {name!r} does not match the Geofabrik pattern '<region>-latest.osm.pbf'"
        )
    return match.group("region")


class PBFReader:
    """Streaming polygonal-element reader backed by osmium."""

    def __init__(
        self,
        pbf_path: str | Path,
        *,
        include_wikipedia_tagged: bool = False,
    ) -> None:
        self.pbf_path = Path(pbf_path)
        self.include_wikipedia_tagged = include_wikipedia_tagged
        if not self.pbf_path.exists():
            raise PBFReadError(f"PBF file does not exist: {self.pbf_path}")
        if not self.pbf_path.is_file():
            raise PBFReadError(f"PBF path is not a file: {self.pbf_path}")

    @property
    def region_name(self) -> str:
        return region_from_filename(self.pbf_path)

    def iter_polygon_candidates(self, callback: Callback) -> None:
        """Stream polygon candidates; deliver each one to ``callback``.

        ``callback`` is invoked synchronously from inside the osmium
        handler, so it must not block on heavy I/O. The handler does
        not retain any state between calls: memory is bounded.
        """
        try:
            handler = _PolygonHandler(
                callback,
                include_wikipedia_tagged=self.include_wikipedia_tagged,
            )
            # ``locations=True`` attaches the NodeLocationsForWays indexer
            # so the Areas assembler can resolve way node coordinates.
            handler.apply_file(str(self.pbf_path), locations=True)
        except (OSError, RuntimeError, ValueError) as e:
            # ``osmium`` raises a mix of OSError / RuntimeError /
            # ValueError depending on the underlying libosmium error.
            # Any of these indicates the file is unreadable.
            raise PBFReadError(f"Failed to read PBF {self.pbf_path}: {e}") from e

    def collect_polygon_candidates(self) -> list[PolygonCandidate]:
        """Convenience wrapper for tests / one-shot runs."""
        out: list[PolygonCandidate] = []
        self.iter_polygon_candidates(out.append)
        return out


class _PolygonHandler(osmium.SimpleHandler):
    """Internal: streams polygonal area candidates built by osmium.

    We use the ``area`` callback: osmium's Areas indexer assembles a
    polygon for every closed way (treated as a polygon) AND every
    multipolygon relation, then calls this hook for each one. The
    GeoJSONFactory receives the pre-built :class:`osmium.osm.Area`
    object and produces a GeoJSON MultiPolygon string.

    The Areas indexer auto-installs because the handler defines
    ``area()``; node locations are auto-installed because ``create``
    needs them.
    """

    def __init__(self, callback: Callback, *, include_wikipedia_tagged: bool = False) -> None:
        super().__init__()
        self._callback = callback
        self._include_wikipedia_tagged = include_wikipedia_tagged
        self._factory = osmium.geom.GeoJSONFactory()

    @staticmethod
    def _tags(tags: Any) -> dict[str, str]:
        return {tag.k: tag.v for tag in tags}

    @staticmethod
    def _has_relevant_tag(tags: Any, *, include_wikipedia_tagged: bool) -> bool:
        """Return whether the final values contain a retained OSM reference.

        ``osmium`` normally exposes unique keys, but keeping the last value
        matches the existing ``dict`` conversion if a malformed input
        contains duplicate keys.
        """
        references = _reference_tags(tags, include_wikipedia_tagged=include_wikipedia_tagged)
        if references.get("wikidata", "").strip():
            return True
        return _has_wikipedia_reference(
            references,
            include_wikipedia_tagged=include_wikipedia_tagged,
        )

    def area(self, a: osmium.osm.Area) -> None:
        tags = _candidate_tags(
            a.tags,
            include_wikipedia_tagged=self._include_wikipedia_tagged,
        )
        if tags is None:
            return
        osm_type = _area_osm_type(a)
        if osm_type is None:
            return
        geom_json = _geometry_json(self._factory, a)
        if geom_json is None:
            return
        self._callback((osm_type, a.id, tags, geom_json))


def _candidate_tags(tags: Any, *, include_wikipedia_tagged: bool) -> dict[str, str] | None:
    """Return normalized tags for a retained area, or ``None``."""
    if not _PolygonHandler._has_relevant_tag(
        tags,
        include_wikipedia_tagged=include_wikipedia_tagged,
    ):
        return None
    return _PolygonHandler._tags(tags)


def _reference_tags(tags: Any, *, include_wikipedia_tagged: bool) -> dict[str, str]:
    """Collect only retained reference keys, preserving the last value."""
    references: dict[str, str] = {}
    for tag in tags:
        if tag.k == "wikidata" or (include_wikipedia_tagged and _is_wikipedia_key(tag.k)):
            references[tag.k] = tag.v
    return references


def _has_wikipedia_reference(values: dict[str, str], *, include_wikipedia_tagged: bool) -> bool:
    """Return whether normalized tags contain an opted-in Wikipedia reference."""
    if not include_wikipedia_tagged:
        return False
    return any(_is_wikipedia_key(key) and value.strip() for key, value in values.items())


def _is_wikipedia_key(key: str) -> bool:
    """Return whether an OSM key stores a Wikipedia reference."""
    return key == "wikipedia" or key.startswith("wikipedia:")


def _area_osm_type(area: Any) -> str | None:
    """Return the OSM identity represented by an assembled area."""
    if area.is_multipolygon():
        return "relation"
    if area.from_way():
        return "way"
    return None


def _geometry_json(factory: Any, area: Any) -> str | None:
    """Build GeoJSON while treating malformed geometry as non-candidate."""
    try:
        return str(factory.create_multipolygon(area))
    except (RuntimeError, ValueError):
        return None
