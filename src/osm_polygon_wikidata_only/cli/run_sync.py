"""CLI shell for ``sync-dir``.

Owns argparse, runtime construction, augmentation-client setup,
heartbeat wiring, the upload-queue lifecycle, the unified-plan
count log line, and the publication pipeline boundary. The
state-execution policy lives in :mod:`pipeline.sync_runner`;
this module only builds collaborators and calls
:func:`pipeline.sync_runner.run_sync`.

When ``--push`` is disabled, ``build_upload_files`` and
``submit_upload`` are both passed as ``None`` so the runner
never invokes publication assembly. When ``--push`` is enabled,
the existing :func:`cli.commands._sync_upload_files` helper is
pre-bound into ``build_upload_files`` until the Phase 7
publication refactor moves it into :mod:`hf.publication`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    augmentation_is_current,
)
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import StubHfHub, upload_files
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.pipeline.processor import ExtractedPbf
from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def execute(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    build_upload_files: Callable[..., list[tuple[Path, str]]] | None = None,
) -> int:
    """Run the ``sync-dir`` CLI command by wiring collaborators to
    :func:`pipeline.sync_runner.run_sync`.

    The CLI shell owns the unified-plan count log line and
    decides whether publication assembly runs. When
    ``--push`` is disabled, both ``build_upload_files`` and
    ``submit_upload`` are ``None`` and the runner never invokes
    the publication assembly.
    """
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    augmentation_client = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )
    pbfs = collect_pbfs([args.input])
    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    core_stems = {name.removesuffix(".osm.pbf") for name in entries}
    current_augmentation = {stem for stem in core_stems if augmentation_is_current(data_root, stem)}
    states = plan_sync_states(
        pbfs,
        core_stems=core_stems,
        augmentation_stems=current_augmentation,
        force=settings.force or not settings.skip_existing,
    )

    push_enabled = bool(getattr(args, "push", False))
    upload_queue = _build_upload_queue(
        push=push_enabled,
        dry_run=getattr(args, "dry_run", False),
        settings=settings,
        data_root=data_root,
        num_threads=getattr(args, "upload_threads", 2),
    )

    counts = {action: sum(state.action is action for state in states) for action in SyncAction}
    LOGGER.info(
        "Unified sync plan: %d augmentation backlog, %d core missing, %d complete",
        counts[SyncAction.AUGMENT],
        counts[SyncAction.PROCESS],
        counts[SyncAction.COMPLETE],
    )

    # Capture settings + clients once so the closures below don't
    # need to look them up at call time.
    wikidata_client = runtime.wikidata
    wikipedia_client = runtime.wikipedia
    runtime_cache = runtime.cache

    from osm_polygon_wikidata_only.pipeline.processor import (
        extract_pbf as _extract_pbf,
    )
    from osm_polygon_wikidata_only.pipeline.processor import (
        process_extracted_pbf as _process_extracted_pbf,
    )

    def _extract(pbf_path: Path) -> ExtractedPbf:
        return _extract_pbf(pbf_path, settings=settings)

    def _process(extracted: ExtractedPbf) -> Any:
        return _process_extracted_pbf(
            extracted,
            data_root=data_root,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            settings=replace(settings, skip_existing=False),
            cache=runtime_cache,
        )

    def _augment(state: RegionSyncState) -> Any:
        from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress

        progress = AugmentationProgress()
        LOGGER.info("Sync region %s: augmentation started", state.stem)
        from osm_polygon_wikidata_only.pipeline.sync_heartbeat import SyncHeartbeat

        actionable = [s for s in states if s.action is not SyncAction.COMPLETE]
        with SyncHeartbeat(
            region=state.stem,
            region_index=actionable.index(state) + 1 if state in actionable else 0,
            region_total=len(actionable) or len(states),
            augmentation_snapshot=progress.snapshot,
            scheduler_snapshot=runtime.scheduler.snapshot,
            auth_snapshot=runtime.session.auth_snapshot,
            log=LOGGER.info,
        ):
            augmentation_result = augment_region(
                data_root,
                state.stem,
                augmentation_client,
                progress=progress,
            )
        LOGGER.info(
            "Unified sync completed %s: %s",
            state.stem,
            augmentation_result.counts,
        )
        return augmentation_result

    def _submit_upload(files: list[tuple[Path, str]], message: str) -> None:
        if upload_queue is None:
            return
        upload_queue.submit(files, message)

    def _close_uploads() -> list[str]:
        if upload_queue is None:
            return []
        return upload_queue.close_and_wait()

    # Push-disabled: do not even hand the runner a publication
    # builder. The runner will skip _maybe_submit entirely.
    publish_builder: Callable[..., list[tuple[Path, str]]] | None = (
        build_upload_files if push_enabled else None
    )
    submit_callback: Callable[[list[tuple[Path, str]], str], None] | None = (
        _submit_upload if push_enabled else None
    )

    from osm_polygon_wikidata_only.pipeline import sync_runner as sync_runner_mod

    try:
        rc = sync_runner_mod.run_sync(
            states,
            extract_pbf=_extract,
            process_extracted_pbf=_process,
            augment_region=_augment,
            build_upload_files=publish_builder,
            commit_message=_commit_message(getattr(args, "commit_message", None)),
            submit_upload=submit_callback,
            close_uploads=_close_uploads,
        )
    except Exception as error:
        if upload_queue is not None:
            LOGGER.error("Unified sync aborted: %s", error)
        raise
    if rc != 0:
        LOGGER.error("Unified sync completed with failures (rc=%d)", rc)
    return rc


def _commit_message(
    override: str | None,
) -> Callable[[RegionSyncState], str]:
    if override:

        def _factory_override(_state: RegionSyncState) -> str:
            return override

        return _factory_override

    def _factory_default(state: RegionSyncState) -> str:
        return f"Sync complete region {state.stem}"

    return _factory_default


def _build_upload_queue(
    *,
    push: bool,
    dry_run: bool,
    settings: Settings,
    data_root: DataRoot,
    num_threads: int,
) -> BackgroundUploadQueue | None:
    """Open the documented ``BackgroundUploadQueue`` and resume pending jobs."""
    if not push:
        return None

    hub = StubHfHub() if dry_run else None

    def upload_job(files: list[tuple[Path, str]], message: str) -> None:
        upload_files(
            settings.repo_id,
            files,
            hub=hub,
            token=settings.hf_token,
            commit_message=message,
            num_threads=num_threads,
        )

    queue = BackgroundUploadQueue(
        upload=upload_job,
        max_pending=2,
        state_dir=data_root.cache / "sync_upload_jobs",
    )
    resumed = queue.resume_pending()
    if resumed:
        LOGGER.info("Resumed %d pending background upload(s)", resumed)
    return queue


__all__ = ["execute"]
