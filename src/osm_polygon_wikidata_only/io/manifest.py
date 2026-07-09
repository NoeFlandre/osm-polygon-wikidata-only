"""Manifest of processed PBFs.

A single JSON file at ``<data-root>/processed/manifests/processed_pbfs.json``
records every PBF that has been processed. Reprocessing a PBF
overwrites the existing entry deterministically.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.domain.models import ManifestStats
from osm_polygon_wikidata_only.utils.json import dumps, loads
from osm_polygon_wikidata_only.utils.time import utc_now_iso

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "processed_pbfs.json"


def manifest_path(processed_manifests_dir: Path) -> Path:
    """Return the canonical path to the manifest JSON file."""
    return processed_manifests_dir / MANIFEST_FILENAME


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read the manifest. Returns ``{}`` if the file does not exist."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    parsed: object = loads(text)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def save_manifest(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write the manifest atomically.

    ``entries`` is keyed by ``source_pbf``. Order of keys is
    deterministic: the output is sorted by key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = dict(sorted(entries.items()))
    path.write_text(dumps(sorted_entries) + "\n", encoding="utf-8")
    LOGGER.info("Saved manifest with %d entries to %s", len(sorted_entries), path)


def make_entry(
    *,
    source_pbf: str,
    region: str,
    polygons_path: str,
    articles_path: str,
    polygon_articles_path: str,
    stats: ManifestStats,
    extraction_version: str,
    processed_at: str | None = None,
) -> dict[str, Any]:
    """Build a single manifest entry."""
    return {
        "source_pbf": source_pbf,
        "region": region,
        "polygons_path": polygons_path,
        "articles_path": articles_path,
        "polygon_articles_path": polygon_articles_path,
        "extraction_version": extraction_version,
        "processed_at": processed_at or utc_now_iso(),
        **stats.to_dict(),
    }


def upsert_entry(
    path: Path,
    *,
    source_pbf: str,
    region: str,
    polygons_path: str,
    articles_path: str,
    polygon_articles_path: str,
    stats: ManifestStats,
    extraction_version: str,
) -> dict[str, Any]:
    """Insert or update the entry for ``source_pbf`` in the manifest.

    Returns the entry that was written.
    """
    entries = load_manifest(path)
    entry = make_entry(
        source_pbf=source_pbf,
        region=region,
        polygons_path=polygons_path,
        articles_path=articles_path,
        polygon_articles_path=polygon_articles_path,
        stats=stats,
        extraction_version=extraction_version,
    )
    entries[source_pbf] = entry
    save_manifest(path, entries)
    return entry


def iter_entries(entries: dict[str, dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Iterate manifest entries in deterministic order."""
    for key in sorted(entries):
        yield key, entries[key]


__all__ = [
    "MANIFEST_FILENAME",
    "iter_entries",
    "load_manifest",
    "make_entry",
    "manifest_path",
    "save_manifest",
    "upsert_entry",
]
