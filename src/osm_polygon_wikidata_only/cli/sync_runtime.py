"""Runtime collaborator assembly for the ``sync-dir`` application."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    load_existing_augmentation_result,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import prepare_local_retirement
from osm_polygon_wikidata_only.cli.dependencies import WikimediaRuntime
from osm_polygon_wikidata_only.cli.sync_application import (
    SyncApplication,
    SyncApplicationContext,
    SyncApplicationServices,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.audit import (
    record_region_recovery_receipt,
)
from osm_polygon_wikidata_only.pipeline.link_migration import (
    apply_link_migration,
    plan_link_migration,
)
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
    clear_metadata_refresh_marker,
    load_metadata_refresh_marker,
    set_metadata_refresh_marker,
)
from osm_polygon_wikidata_only.pipeline.processor import ExtractedPbf, ProcessResult
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState
from osm_polygon_wikidata_only.pipeline.sync_planning import (
    PreparedSyncPlan,
    ensure_recovery_audit_unblocked,
)
from osm_polygon_wikidata_only.pipeline.sync_reconciliation import log_remote_reconciliation_summary
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import (
    audit_wikidata_integrity,
    repair_wikidata_region,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def load_existing_core_for_publication(
    data_root: DataRoot,
    stem: str,
    core: object | None,
    *,
    required: bool,
) -> object | None:
    """Load finalized core artifacts when a repair changed or must republish them."""
    if core is not None or not required:
        return core
    # Recovery-only publication is loaded after the normal no-repair path is known.
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        load_existing_core_artifacts,
    )

    return load_existing_core_artifacts(data_root, stem)


def build_augmentation_client(
    data_root: DataRoot, runtime: WikimediaRuntime
) -> AugmentationWikimediaClient:
    return AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )


def run_sync_application(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    runtime: WikimediaRuntime,
    augmentation_client: AugmentationWikimediaClient,
    prepared: PreparedSyncPlan,
    push_enabled: bool,
    dry_run: bool,
    upload_queue: BackgroundUploadQueue | None,
    containment_enqueued: bool,
    publish_builder: Callable[..., list[PublicationOp]] | None,
) -> int:
    application = SyncApplication(
        context=SyncApplicationContext(
            data_root=data_root,
            settings=settings,
            runtime=runtime,
            augmentation_client=augmentation_client,
            states=prepared.states,
            push_enabled=push_enabled,
            dry_run=dry_run,
            pending_stems=prepared.all_pending_stems,
            stems_with_gaps=prepared.remote_state.stems_with_gaps,
            reconciliation_plan=prepared.remote_state.plan,
            upload_queue=upload_queue,
            publish_builder=publish_builder,
            core_will_be_repaired=prepared.core_will_be_repaired,
            core_repaired=prepared.remote_state.core_repaired,
            containment_enqueued=containment_enqueued,
        ),
        services=build_sync_services(
            args,
            data_root=data_root,
            settings=settings,
            runtime=runtime,
        ),
    )
    return application.run().return_code


def build_sync_services(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    runtime: WikimediaRuntime,
) -> SyncApplicationServices:
    """Build the injected sync collaborators from this module's bindings.

    Collaborators are resolved from this module at call time so tests can
    replace their owning bindings without coupling to the CLI shell.
    """
    wikidata_client = runtime.wikidata
    wikipedia_client = runtime.wikipedia
    runtime_cache = runtime.cache
    # Construct the heavy processing collaborators after the plan and runtime exist.
    from osm_polygon_wikidata_only.pipeline.processor import (  # noqa: PLC0415
        extract_pbf as _extract_pbf,
    )
    from osm_polygon_wikidata_only.pipeline.processor import (  # noqa: PLC0415
        process_extracted_pbf as _process_extracted_pbf,
    )

    def _extract(pbf_path: Path) -> ExtractedPbf:
        return _extract_pbf(pbf_path, settings=settings)

    def _process(extracted: ExtractedPbf) -> ProcessResult:
        return _process_extracted_pbf(
            extracted,
            data_root=data_root,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            settings=replace(settings, skip_existing=False),
            cache=runtime_cache,
        )

    # Keep the remaining execution collaborators lazy until the selected sync path is ready.
    from osm_polygon_wikidata_only.augmentation.progress import (  # noqa: PLC0415
        AugmentationProgress,
    )
    from osm_polygon_wikidata_only.hf.publication import (  # noqa: PLC0415
        assemble_metadata_only_upload,
        assemble_region_upload,
    )
    from osm_polygon_wikidata_only.pipeline import sync_runner as sync_runner_mod  # noqa: PLC0415
    from osm_polygon_wikidata_only.pipeline.sync_heartbeat import (  # noqa: PLC0415
        SyncHeartbeat,
    )

    return SyncApplicationServices(
        extract_pbf=_extract,
        process_extracted_pbf=_process,
        augment_region=augment_region,
        load_existing_augmentation=load_existing_augmentation_result,
        run_sync=sync_runner_mod.run_sync,
        plan_link_migration=plan_link_migration,
        apply_link_migration=apply_link_migration,
        audit_wikidata_integrity=audit_wikidata_integrity,
        ensure_recovery_audit_unblocked=ensure_recovery_audit_unblocked,
        repair_wikidata_region=repair_wikidata_region,
        prepare_local_retirement=prepare_local_retirement,
        add_pending_publications=add_pending_publications,
        record_region_recovery_receipt=record_region_recovery_receipt,
        assemble_region_upload=assemble_region_upload,
        assemble_metadata_only_upload=assemble_metadata_only_upload,
        load_existing_core_for_publication=load_existing_core_for_publication,
        commit_message=_commit_message(getattr(args, "commit_message", None)),
        log_remote_reconciliation_summary=log_remote_reconciliation_summary,
        load_metadata_refresh_marker=load_metadata_refresh_marker,
        set_metadata_refresh_marker=set_metadata_refresh_marker,
        clear_metadata_refresh_marker=clear_metadata_refresh_marker,
        augmentation_progress=AugmentationProgress,
        sync_heartbeat=SyncHeartbeat,
        logger=LOGGER,
    )


def _commit_message(override: str | None) -> Callable[[RegionSyncState], str]:
    """Build the stable commit message callback used by region publication."""
    if override:
        return lambda _state: override
    return lambda state: f"Sync complete region {state.stem}"
