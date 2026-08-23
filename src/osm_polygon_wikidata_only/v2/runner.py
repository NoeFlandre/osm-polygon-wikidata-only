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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import LOCAL_V2_DATASET_HERO_FILE
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


@dataclass(slots=True)
class _V2ExecutionState:
    data_root: DataRoot
    settings: Settings
    wikipedia_client: WikipediaClient
    section_client: SectionClient | None
    section_workers: int
    cache: JsonFileCache
    hash_cache: V2FileHashCache
    index: Any
    upload: Upload | None
    push: bool
    extraction_executor: ThreadPoolExecutor
    extraction_future: Future[Any] | None = None
    extraction_index: int | None = None
    provisional_stems: list[str] = field(default_factory=list)
    extracted_stems: list[str] = field(default_factory=list)
    pending_publish_stems: list[str] = field(default_factory=list)


def _plan_region(
    pbf: Path,
    *,
    data_root: DataRoot,
    settings: Settings,
    manifest: dict[str, dict[str, Any]],
    push: bool,
    remote_inventory: RemoteInventory | None,
    hash_cache: V2FileHashCache | None,
) -> _RegionPlan:
    """Choose the durable action for one source PBF."""
    stem = pbf.name.removesuffix(".osm.pbf")
    action = _region_action(
        stem,
        data_root=data_root,
        settings=settings,
        manifest=manifest,
        push=push,
        remote_inventory=remote_inventory,
        hash_cache=hash_cache,
    )
    return _RegionPlan(pbf, stem, action)


def _region_action(
    stem: str,
    *,
    data_root: DataRoot,
    settings: Settings,
    manifest: dict[str, dict[str, Any]],
    push: bool,
    remote_inventory: RemoteInventory | None,
    hash_cache: V2FileHashCache | None,
) -> str:
    """Select extract, reconcile, publish, or skip for one region."""
    local_artifacts_current = _local_region_artifacts_current(
        stem,
        data_root=data_root,
        settings=settings,
        manifest=manifest,
        hash_cache=hash_cache,
    )
    if not local_artifacts_current:
        return "extract"
    if _needs_index_reconciliation(manifest, stem):
        return "reconcile"
    return _remote_region_action(
        data_root,
        stem,
        push=push,
        remote_inventory=remote_inventory,
    )


def _needs_index_reconciliation(manifest: dict[str, dict[str, Any]], stem: str) -> bool:
    return manifest[stem].get("v1_index_reconciled", True) is not True


def _remote_region_action(
    data_root: DataRoot,
    stem: str,
    *,
    push: bool,
    remote_inventory: RemoteInventory | None,
) -> str:
    if not push or remote_region_complete(remote_inventory, data_root.processed_v2, stem):
        return "skip"
    return "publish"


def _local_region_artifacts_current(
    stem: str,
    *,
    data_root: DataRoot,
    settings: Settings,
    manifest: dict[str, dict[str, Any]],
    hash_cache: V2FileHashCache | None,
) -> bool:
    """Check local skip settings and the durable artifact contract."""
    if not settings.skip_existing or settings.force:
        return False
    return _region_artifacts_are_current(
        data_root.processed_v2,
        stem,
        manifest,
        hash_cache=hash_cache,
    )


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
    return tuple(
        _plan_region(
            pbf,
            data_root=data_root,
            settings=settings,
            manifest=manifest,
            push=push,
            remote_inventory=remote_inventory,
            hash_cache=hash_cache,
        )
        for pbf in pbfs
    )


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
    state = _V2ExecutionState(
        data_root=data_root,
        settings=settings,
        wikipedia_client=wikipedia_client,
        section_client=section_client,
        section_workers=section_workers,
        cache=v2_cache,
        hash_cache=hash_cache,
        index=index,
        upload=upload,
        push=push,
        extraction_executor=ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="v2-extraction",
        ),
    )
    try:
        _schedule_next(state, plans, after_index=-1)
        completed = _process_regions(state, plans, total=len(pbfs))
        _finalize_regions(state, completed, trackio_publish=trackio_publish)
        return 0
    finally:
        _cleanup(state)


def _flush_pending_publishes(state: _V2ExecutionState) -> None:
    if not state.pending_publish_stems:
        return
    if state.upload is None:
        raise RuntimeError("V2 publication requested without an upload callback")
    upload_region_batches(
        state.data_root.processed_v2,
        state.pending_publish_stems,
        upload=state.upload,
        repair=True,
    )
    state.pending_publish_stems.clear()


def _schedule_next(
    state: _V2ExecutionState,
    plans: tuple[_RegionPlan, ...],
    *,
    after_index: int,
) -> None:
    if state.extraction_future is not None:
        return
    for candidate_index in range(after_index + 1, len(plans)):
        candidate = plans[candidate_index]
        if candidate.action != "extract":
            continue
        LOGGER.info(
            "Starting V2 region %s (extraction scheduled; one-region prefetch)",
            candidate.stem,
        )
        state.extraction_future = state.extraction_executor.submit(
            extract_v2_pbf,
            candidate.pbf,
            settings=state.settings,
            checkpoint_dir=state.data_root.v2_cache / "checkpoints",
        )
        state.extraction_index = candidate_index
        return


def _take_extraction(state: _V2ExecutionState, plan: _RegionPlan, position: int) -> Any:
    if state.extraction_future is None or state.extraction_index != position:
        raise RuntimeError(f"V2 extraction prefetch lost region {plan.stem}")
    future = state.extraction_future
    state.extraction_future = None
    state.extraction_index = None
    return future.result()


def _process_regions(
    state: _V2ExecutionState,
    plans: tuple[_RegionPlan, ...],
    *,
    total: int,
) -> int:
    completed = 0
    for position, plan in enumerate(plans):
        _process_plan(state, plan, position, plans, completed + 1, total)
        completed += 1
    return completed


def _process_plan(
    state: _V2ExecutionState,
    plan: _RegionPlan,
    position: int,
    plans: tuple[_RegionPlan, ...],
    completed: int,
    total: int,
) -> None:
    if plan.action == "skip":
        LOGGER.info("Skipping V2 %s (already current)", plan.stem)
        return
    if plan.action == "publish":
        LOGGER.info("Queueing existing V2 %s for batched publication", plan.stem)
        state.pending_publish_stems.append(plan.stem)
        if len(state.pending_publish_stems) >= _REGION_UPLOAD_BATCH_SIZE:
            _flush_pending_publishes(state)
        return
    if plan.action == "reconcile":
        LOGGER.info("Resuming V2 %s from persisted provisional artifacts", plan.stem)
        state.provisional_stems.append(plan.stem)
        state.extracted_stems.append(plan.stem)
        return
    _process_extraction(state, plan, position, plans, completed, total)


def _process_extraction(
    state: _V2ExecutionState,
    plan: _RegionPlan,
    position: int,
    plans: tuple[_RegionPlan, ...],
    completed: int,
    total: int,
) -> None:
    _flush_pending_publishes(state)
    extracted = _take_extraction(state, plan, position)
    _schedule_next(state, plans, after_index=position)
    merge_v2_region(
        state.data_root,
        extracted,
        index=state.index,
        wikipedia_client=state.wikipedia_client,
        section_client=state.section_client,
        section_workers=state.section_workers,
        cache=state.cache,
        fetch_full_text=state.settings.fetch_full_text,
        direct_workers=state.settings.enrichment_site_workers,
        wait_for_index=False,
        checkpoint_dir=state.data_root.v2_cache / "checkpoints",
    )
    state.extracted_stems.append(plan.stem)
    if _has_wikipedia_refs(extracted):
        state.provisional_stems.append(plan.stem)
        LOGGER.info(
            "Prepared V2 region %s while the V1 reuse index continues; final reconciliation is deferred",
            plan.stem,
        )
    else:
        LOGGER.info("Prepared V2 region %s (%d/%d)", plan.stem, completed, total)
        clear_v2_checkpoints(state.data_root.v2_cache / "checkpoints", plan.stem)


def _finalize_regions(
    state: _V2ExecutionState,
    completed: int,
    *,
    trackio_publish: Callable[[V2CardStats], None] | None,
) -> None:
    LOGGER.info(
        "V2 prepared %d region(s) while the V1 index was building; finalizing index and reconciling %d provisional region(s)",
        len(state.extracted_stems),
        len(state.provisional_stems),
    )
    _flush_pending_publishes(state)
    state.index.wait_until_ready()
    _reconcile_provisional_regions(state)
    _publish_regions(state)
    for stem in state.extracted_stems:
        LOGGER.info("Completed V2 region %s", stem)
    card = _write_v2_metadata(state, trackio_publish)
    LOGGER.info("V2 sync complete: %d region(s); card=%s", completed, card)


def _reconcile_provisional_regions(state: _V2ExecutionState) -> None:
    for stem in state.provisional_stems:
        reconcile_v2_region(
            state.data_root,
            stem,
            index=state.index,
            wikipedia_client=state.wikipedia_client,
            cache=state.cache,
            fetch_full_text=state.settings.fetch_full_text,
            section_client=state.section_client,
            section_workers=state.section_workers,
            checkpoint_dir=state.data_root.v2_cache / "checkpoints",
        )
        clear_v2_checkpoints(state.data_root.v2_cache / "checkpoints", stem)


def _publish_regions(state: _V2ExecutionState) -> None:
    if not state.push:
        return
    if state.upload is None:
        raise RuntimeError("V2 publication requested without an upload callback")
    upload_region_batches(
        state.data_root.processed_v2,
        state.extracted_stems,
        upload=state.upload,
        repair=False,
    )


def _write_v2_metadata(
    state: _V2ExecutionState,
    trackio_publish: Callable[[V2CardStats], None] | None,
) -> Path:
    land_path = state.data_root.cache / "ne_110m_land.geojson"
    generate_v2_map_assets(
        state.data_root.processed_v2,
        state.data_root.processed_v2 / "assets",
        land_geojson_path=land_path if land_path.is_file() else None,
        land_cache_dir=state.data_root.cache,
    )
    card_stats = compute_v2_card_stats(
        state.data_root.processed_v2,
        v1_processed=state.data_root.processed,
    )
    card = write_v2_card(
        state.data_root.processed_v2,
        v1_processed=state.data_root.processed,
        stats=card_stats,
    )
    if state.push and trackio_publish is not None:
        trackio_publish(card_stats)
    if state.push:
        _publish_v2_metadata(state)
    return card


def _publish_v2_metadata(state: _V2ExecutionState) -> None:
    if state.upload is None:
        raise RuntimeError("V2 publication requested without an upload callback")
    state.upload(
        metadata_publication_ops(
            state.data_root.processed_v2,
            hero_path=LOCAL_V2_DATASET_HERO_FILE if LOCAL_V2_DATASET_HERO_FILE.is_file() else None,
        ),
        "Update V2 dataset card and manifest",
    )


def _cleanup(state: _V2ExecutionState) -> None:
    if state.extraction_future is not None:
        state.extraction_future.cancel()
    state.extraction_executor.shutdown(wait=True, cancel_futures=True)
    try:
        state.hash_cache.flush()
    except OSError:
        LOGGER.warning("V2 resume hash cache could not be saved; next run will rehash files")
    state.index.close()


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
    expected = _expected_region_paths(stem)
    if not _manifest_paths_match(entry, expected):
        return False
    hashes = entry.get("file_hashes")
    if not isinstance(hashes, dict):
        return False
    return _region_files_match(processed_v2, expected, hashes, hash_cache=hash_cache)


def _expected_region_paths(stem: str) -> dict[str, str]:
    """Return the canonical V2 artifact paths for one region."""
    return {
        "polygons_path": f"polygons/{stem}.parquet",
        "documents_path": f"wikipedia/documents/{stem}.parquet",
        "sections_path": f"wikipedia/sections/{stem}.parquet",
        "links_path": f"polygon_document_links/{stem}.parquet",
    }


def _manifest_paths_match(entry: dict[str, Any], expected: dict[str, str]) -> bool:
    """Check that a manifest entry points at the canonical artifacts."""
    return all(entry.get(field) == value for field, value in expected.items())


def _region_files_match(
    processed_v2: Path,
    expected: dict[str, str],
    hashes: dict[str, Any],
    *,
    hash_cache: V2FileHashCache | None,
) -> bool:
    """Verify all V2 artifacts exist and match their manifest hashes."""
    for relative in expected.values():
        path = processed_v2 / relative
        if not path.is_file():
            return False
        current_hash = hash_cache.digest(path) if hash_cache is not None else sha256_file(path)
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


__all__ = ["run_v2_sync"]
