"""Deterministic ID generation.

Stable IDs are required so the same PBF processed twice produces the
same row keys — which keeps the manifest and remote parquet paths
idempotent.
"""

from __future__ import annotations

import hashlib


def polygon_id(source_pbf_stem: str, osm_type: str, osm_id: int) -> str:
    """Stable polygon identifier: ``<stem>:<osm_type>:<osm_id>``.

    ``source_pbf_stem`` is the PBF filename without the
    ``-latest.osm.pbf`` suffix (e.g. ``monaco-latest``). This matches
    the remote parquet path pattern.
    """
    return f"{source_pbf_stem}:{osm_type}:{osm_id}"


def article_id(wikidata: str, language: str, page_id: int, revision_id: int) -> str:
    """Stable article identifier.

    Format: ``<wikidata>:<language>:<page_id>:<revision_id>``. Includes
    the revision_id so that the same article at different points in
    time produces a different row (useful for change tracking).
    """
    return f"{wikidata}:{language}:{page_id}:{revision_id}"


def content_hash(text: str) -> str:
    """Stable SHA-256 of article text, used for deduplication.

    Returns the lowercase hex digest of the UTF-8 encoding of ``text``.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
