"""Shared file-stat fingerprints used by V2 restart and cache contracts.

The class now lives in :mod:`osm_polygon_wikidata_only.io.fingerprints` so
that the I/O layer does not depend on V2. This module keeps the historical
import path working.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.io.fingerprints import FileStatFingerprint

__all__ = ["FileStatFingerprint"]
