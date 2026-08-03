"""Resumable V2 build coordinator.

This runner never calls the V1 processor.  It scans each source PBF with the
V2 discovery rule, reuses finalized V1 rows and sidecars, fetches only direct
Wikipedia pages absent from V1, and commits each completed region atomically.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.v2.card import write_v2_card
from osm_polygon_wikidata_only.v2.config import V2_CACHE_CONTRACT_VERSION, V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.extractor import extract_v2_pbf
from osm_polygon_wikidata_only.v2.publication import (
    metadata_publication_ops,
    region_publication_ops,
)
from osm_polygon_wikidata_only.v2.reuse import SectionClient, merge_v2_region
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest
from osm_polygon_wikidata_only.v2.v1_index import build_v1_reuse_index

LOGGER = logging.getLogger(__name__)


Upload = Callable[[list[PublicationOp], str], None]


def run_v2_sync(
    input_path: Path,
    *,
    data_root: DataRoot,
    settings: Settings,
    wikipedia_client: WikipediaClient,
    section_client: SectionClient | None = None,
    section_workers: int = 8,
    cache: JsonFileCache | None = None,
    push: bool = False,
    upload: Upload | None = None,
    remote_inventory: RemoteInventory | None = None,
) -> int:
    """Build V2 regions and optionally publish each completed region."""
    pbfs = collect_pbfs([input_path])
    if not pbfs:
        LOGGER.warning("No PBF inputs to process for V2")
        return 0
    data_root.processed_v2.mkdir(parents=True, exist_ok=True)
    index = build_v1_reuse_index(data_root.processed)
    v2_cache = cache or JsonFileCache(
        data_root.v2_cache / "wikipedia",
        contract_version=V2_CACHE_CONTRACT_VERSION,
    )
    manifest = load_v2_manifest(data_root.processed_v2)
    completed = 0
    for pbf in pbfs:
        stem = pbf.name.removesuffix(".osm.pbf")
        current = (
            settings.skip_existing
            and not settings.force
            and _region_is_current(data_root.processed_v2, stem, manifest)
        )
        if current:
            region_ops = region_publication_ops(data_root.processed_v2, stem)
            remote_complete = remote_inventory is not None and all(
                remote_inventory.contains(op.path_in_repo) for op in region_ops
            )
            if not push or remote_complete:
                LOGGER.info("Skipping V2 %s (already current)", stem)
                completed += 1
                continue
            if upload is None:
                raise RuntimeError("V2 publication requested without an upload callback")
            LOGGER.info("Publishing existing V2 %s (remote artifacts are incomplete)", stem)
            upload(region_ops, f"Repair V2 region {stem}")
            completed += 1
            continue
        LOGGER.info("Starting V2 region %s", stem)
        extracted = extract_v2_pbf(pbf, settings=settings)
        merge_v2_region(
            data_root,
            extracted,
            index=index,
            wikipedia_client=wikipedia_client,
            section_client=section_client,
            section_workers=section_workers,
            cache=v2_cache,
            fetch_full_text=settings.fetch_full_text,
        )
        manifest = load_v2_manifest(data_root.processed_v2)
        completed += 1
        if push:
            if upload is None:
                raise RuntimeError("V2 publication requested without an upload callback")
            upload(region_publication_ops(data_root.processed_v2, stem), f"Add V2 region {stem}")
        LOGGER.info("Completed V2 region %s (%d/%d)", stem, completed, len(pbfs))

    card = write_v2_card(data_root.processed_v2)
    if push:
        if upload is None:
            raise RuntimeError("V2 publication requested without an upload callback")
        upload(
            metadata_publication_ops(data_root.processed_v2), "Update V2 dataset card and manifest"
        )
    LOGGER.info("V2 sync complete: %d region(s); card=%s", completed, card)
    return 0


def _region_is_current(
    processed_v2: Path,
    stem: str,
    manifest: dict[str, dict[str, Any]],
) -> bool:
    entry = manifest.get(stem)
    if entry is None or entry.get("contract_version") != V2_CONTRACT_VERSION:
        return False
    expected = {
        "polygons_path": f"polygons/{stem}.parquet",
        "documents_path": f"wikipedia/documents/{stem}.parquet",
        "sections_path": f"wikipedia/sections/{stem}.parquet",
        "links_path": f"polygon_document_links/{stem}.parquet",
    }
    if any(entry.get(field) != value for field, value in expected.items()):
        return False
    hashes = entry.get("file_hashes")
    if not isinstance(hashes, dict):
        return False
    for relative in expected.values():
        path = processed_v2 / relative
        if not path.is_file() or hashes.get(relative) != _sha256(path):
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_v2_sync"]
