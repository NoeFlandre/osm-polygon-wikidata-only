"""Streaming reader for OSM PBF files.

The reader is wrapped around an osmium SimpleHandler. It invokes a
caller-supplied callback for every way/relation that:

* carries a non-empty ``wikidata`` tag, **and**
* is a polygonal candidate (closed way, or relation of ``type=multipolygon``).

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

    def __init__(self, pbf_path: str | Path) -> None:
        self.pbf_path = Path(pbf_path)
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
            handler = _PolygonHandler(callback)
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

    def __init__(self, callback: Callback) -> None:
        super().__init__()
        self._callback = callback
        self._factory = osmium.geom.GeoJSONFactory()

    @staticmethod
    def _tags(tags: Any) -> dict[str, str]:
        return {tag.k: tag.v for tag in tags}

    def area(self, a: osmium.osm.Area) -> None:
        tags = self._tags(a.tags)
        wd = tags.get("wikidata", "").strip()
        if not wd:
            return
        if a.is_multipolygon():
            osm_type = "relation"
        elif a.from_way():
            osm_type = "way"
        else:
            return
        try:
            geom_json: str = self._factory.create_multipolygon(a)
        except (RuntimeError, ValueError):
            return
        self._callback((osm_type, a.id, tags, geom_json))
