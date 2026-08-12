"""Resumable V2 build coordinator.

This runner never calls the V1 processor.  It scans each source PBF with the
V2 discovery rule, reuses finalized V1 rows and sidecars, fetches only direct
Wikipedia pages absent from V1, and commits each completed region atomically.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import LOCAL_DATASET_HERO_FILE
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.v2.card import V2CardStats, compute_v2_card_stats, write_v2_card
from osm_polygon_wikidata_only.v2.checkpoints import clear_v2_checkpoints
from osm_polygon_wikidata_only.v2.config import V2_CACHE_CONTRACT_VERSION, V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.extractor import extract_v2_pbf
from osm_polygon_wikidata_only.v2.maps import generate_v2_map_assets
from osm_polygon_wikidata_only.v2.publication import (
    _REGION_UPLOAD_BATCH_SIZE,
    Upload,
    metadata_publication_ops,
    remote_region_complete,
    upload_region_batches,
)
from osm_polygon_wikidata_only.v2.resume import V2FileHashCache
from osm_polygon_wikidata_only.v2.reuse import (
    SectionClient,
    merge_v2_region,
    reconcile_v2_region,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest
from osm_polygon_wikidata_only.v2.v1_index import start_v1_reuse_index

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RegionPlan:
    pbf: Path
    stem: str
    action: str


def _plan_regions(
    pbfs: Sequence[Path],
    *,
    data_root: DataRoot,
    settings: Settings,
    manifest: dict[str, dict[str, Any]],
    push: bool,
    remote_inventory: RemoteInventory | None,
    hash_cache: V2FileHashCache | None = None,
) -> tuple[_RegionPlan, ...]:
    plans: list[_RegionPlan] = []
    for pbf in pbfs:
        stem = pbf.name.removesuffix(".osm.pbf")
        local_artifacts_current = (
            settings.skip_existing
            and not settings.force
            and _region_artifacts_are_current(
                data_root.processed_v2,
                stem,
                manifest,
                hash_cache=hash_cache,
            )
        )
        if not local_artifacts_current:
            action = "extract"
        elif manifest[stem].get("v1_index_reconciled", True) is not True:
            action = "reconcile"
        elif not push or remote_region_complete(
            remote_inventory,
            data_root.processed_v2,
            stem,
        ):
            action = "skip"
        else:
            action = "publish"
        plans.append(_RegionPlan(pbf, stem, action))
    return tuple(plans)


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
    trackio_publish: Callable[[V2CardStats], None] | None = None,
) -> int:
    """Build V2 regions and optionally publish each completed region."""
    pbfs = collect_pbfs([input_path])
    if not pbfs:
        LOGGER.warning("No PBF inputs to process for V2")
        return 0
    data_root.processed_v2.mkdir(parents=True, exist_ok=True)
    index = start_v1_reuse_index(
        data_root.processed,
        cache_dir=data_root.v2_cache / "v1-index",
    )
    LOGGER.info(
        "V2 V1 reuse index started in background; extraction and indexed-page reuse can overlap"
    )
    v2_cache = cache or JsonFileCache(
        data_root.v2_cache / "wikipedia",
        contract_version=V2_CACHE_CONTRACT_VERSION,
    )
    hash_cache = V2FileHashCache(data_root.v2_cache / "resume-hashes.json")
    manifest = load_v2_manifest(data_root.processed_v2)
    plans = _plan_regions(
        pbfs,
        data_root=data_root,
        settings=settings,
        manifest=manifest,
        push=push,
        remote_inventory=remote_inventory,
        hash_cache=hash_cache,
    )
    completed = 0
    extraction_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="v2-extraction",
    )
    extraction_future: Future[Any] | None = None
    extraction_index: int | None = None
    provisional_stems: list[str] = []
    extracted_stems: list[str] = []
    pending_publish_stems: list[str] = []

    def flush_pending_publishes() -> None:
        if not pending_publish_stems:
            return
        if upload is None:
            raise RuntimeError("V2 publication requested without an upload callback")
        upload_region_batches(
            data_root.processed_v2,
            pending_publish_stems,
            upload=upload,
            repair=True,
        )
        pending_publish_stems.clear()

    def schedule_next(after_index: int) -> None:
        nonlocal extraction_future, extraction_index
        if extraction_future is not None:
            return
        for candidate_index in range(after_index + 1, len(plans)):
            candidate = plans[candidate_index]
            if candidate.action != "extract":
                continue
            LOGGER.info(
                "Starting V2 region %s (extraction scheduled; one-region prefetch)",
                candidate.stem,
            )
            extraction_future = extraction_executor.submit(
                extract_v2_pbf,
                candidate.pbf,
                settings=settings,
                checkpoint_dir=data_root.v2_cache / "checkpoints",
            )
            extraction_index = candidate_index
            return

    try:
        schedule_next(-1)
        for position, plan in enumerate(plans):
            stem = plan.stem
            if plan.action == "skip":
                LOGGER.info("Skipping V2 %s (already current)", stem)
                completed += 1
                continue
            if plan.action == "publish":
                LOGGER.info("Queueing existing V2 %s for batched publication", stem)
                pending_publish_stems.append(stem)
                if len(pending_publish_stems) >= _REGION_UPLOAD_BATCH_SIZE:
                    flush_pending_publishes()
                completed += 1
                continue
            if plan.action == "reconcile":
                LOGGER.info("Resuming V2 %s from persisted provisional artifacts", stem)
                provisional_stems.append(stem)
                extracted_stems.append(stem)
                completed += 1
                continue

            if extraction_future is None or extraction_index != position:
                raise RuntimeError(f"V2 extraction prefetch lost region {stem}")
            flush_pending_publishes()
            future = extraction_future
            extraction_future = None
            extraction_index = None
            extracted = future.result()
            schedule_next(position)
            merge_v2_region(
                data_root,
                extracted,
                index=index,
                wikipedia_client=wikipedia_client,
                section_client=section_client,
                section_workers=section_workers,
                cache=v2_cache,
                fetch_full_text=settings.fetch_full_text,
                direct_workers=settings.enrichment_site_workers,
                wait_for_index=False,
                checkpoint_dir=data_root.v2_cache / "checkpoints",
            )
            completed += 1
            extracted_stems.append(stem)
            if _has_wikipedia_refs(extracted):
                provisional_stems.append(stem)
                LOGGER.info(
                    "Prepared V2 region %s while the V1 reuse index continues; final reconciliation is deferred",
                    stem,
                )
            else:
                LOGGER.info("Prepared V2 region %s (%d/%d)", stem, completed, len(pbfs))
                clear_v2_checkpoints(data_root.v2_cache / "checkpoints", stem)

        # Finalize every speculative region only after every V1 shard has been
        # checked.  All extraction, direct page fetching, and section parsing
        # above can therefore advance while this shared index is building.
        LOGGER.info(
            "V2 prepared %d region(s) while the V1 reuse index was building; finalizing index and reconciling %d provisional region(s)",
            len(extracted_stems),
            len(provisional_stems),
        )
        flush_pending_publishes()
        index.wait_until_ready()
        for stem in provisional_stems:
            reconcile_v2_region(
                data_root,
                stem,
                index=index,
                wikipedia_client=wikipedia_client,
                cache=v2_cache,
                fetch_full_text=settings.fetch_full_text,
                section_client=section_client,
                section_workers=section_workers,
                checkpoint_dir=data_root.v2_cache / "checkpoints",
            )
            clear_v2_checkpoints(data_root.v2_cache / "checkpoints", stem)
        if push:
            if upload is None:
                raise RuntimeError("V2 publication requested without an upload callback")
            upload_region_batches(
                data_root.processed_v2,
                extracted_stems,
                upload=upload,
                repair=False,
            )
        for stem in extracted_stems:
            LOGGER.info("Completed V2 region %s", stem)
        land_path = data_root.cache / "ne_110m_land.geojson"
        generate_v2_map_assets(
            data_root.processed_v2,
            data_root.processed_v2 / "assets",
            land_geojson_path=land_path if land_path.is_file() else None,
            land_cache_dir=data_root.cache,
        )
        card_stats = compute_v2_card_stats(
            data_root.processed_v2,
            v1_processed=data_root.processed,
        )
        card = write_v2_card(
            data_root.processed_v2,
            v1_processed=data_root.processed,
            stats=card_stats,
        )
        if push and trackio_publish is not None:
            trackio_publish(card_stats)
        if push:
            if upload is None:
                raise RuntimeError("V2 publication requested without an upload callback")
            upload(
                metadata_publication_ops(
                    data_root.processed_v2,
                    hero_path=LOCAL_DATASET_HERO_FILE
                    if LOCAL_DATASET_HERO_FILE.is_file()
                    else None,
                ),
                "Update V2 dataset card and manifest",
            )
        LOGGER.info("V2 sync complete: %d region(s); card=%s", completed, card)
        return 0
    finally:
        if extraction_future is not None:
            extraction_future.cancel()
        extraction_executor.shutdown(wait=True, cancel_futures=True)
        try:
            hash_cache.flush()
        except OSError:
            LOGGER.warning("V2 resume hash cache could not be saved; next run will rehash files")
        index.close()


def _has_wikipedia_refs(extracted: Any) -> bool:
    """Return whether an extracted region needs final V1 title reconciliation."""
    for polygon in extracted.polygons:
        try:
            refs = json.loads(str(polygon.get("wikipedia_tag_refs", "[]")))
        except json.JSONDecodeError:
            continue
        if isinstance(refs, list) and refs:
            return True
    return False


def _region_artifacts_are_current(
    processed_v2: Path,
    stem: str,
    manifest: dict[str, dict[str, Any]],
    *,
    hash_cache: V2FileHashCache | None = None,
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
        if not path.is_file():
            return False
        current_hash = hash_cache.digest(path) if hash_cache is not None else _sha256(path)
        if hashes.get(relative) != current_hash:
            return False
    return True


def _region_is_current(
    processed_v2: Path,
    stem: str,
    manifest: dict[str, dict[str, Any]],
    *,
    hash_cache: V2FileHashCache | None = None,
) -> bool:
    """Return whether a region has current artifacts and final reconciliation."""
    return (
        _region_artifacts_are_current(
            processed_v2,
            stem,
            manifest,
            hash_cache=hash_cache,
        )
        and manifest[stem].get("v1_index_reconciled", True) is True
    )


def _sha256(path: Path) -> str:
    return sha256_file(path)


__all__ = ["run_v2_sync"]
