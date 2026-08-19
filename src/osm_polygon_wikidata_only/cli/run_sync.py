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
the CLI shell builds the region-publication list through
:func:`hf.publication.assemble_region_upload` (a pure assembler
that performs NO upload) and submits the returned list through
the upload queue exactly once per region.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    augmentation_is_current,
    load_existing_augmentation_result,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    MigrationError,
    MigrationOperation,
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import (
    finalize_local_retirement,
    prepare_local_retirement,
)
from osm_polygon_wikidata_only.cli._sync.retirement import (
    paired_retirement_stems as _paired_retirement_stems,
)
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.cli.sync_application import (
    SyncApplication,
    SyncApplicationContext,
    SyncApplicationServices,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.audit import (
    record_region_recovery_receipt,
)
from osm_polygon_wikidata_only.pipeline.link_migration import (
    apply_link_migration,
    plan_link_migration,
)
from osm_polygon_wikidata_only.pipeline.local_validation import LocalValidationProgress
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
    clear_metadata_refresh_marker,
    load_metadata_refresh_marker,
    load_pending_publications,
    remove_pending_publications,
)
from osm_polygon_wikidata_only.pipeline.processor import ExtractedPbf
from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import (
    RecoveryAuditResult,
    audit_wikidata_integrity,
    repair_wikidata_region,
)

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


@dataclass(slots=True)
class _RemoteReconciliation:
    """Remote inputs computed once before sync-state planning."""

    inventory: RemoteInventory | None
    plan: Any | None
    augmentation_current: dict[str, bool]
    stems_with_gaps: set[str]
    containment_publications: dict[str, tuple[str, ...]]
    core_repaired: bool


def _recovery_audit_stems(
    *,
    input_stems: set[str],
    core_stems: set[str],
    current_augmentation: set[str],
    force: bool,
) -> list[str]:
    """Return finalized shards eligible for surgical recovery auditing."""
    if force:
        return []
    return sorted(input_stems & core_stems & current_augmentation)


def _ensure_recovery_audit_unblocked(audit: RecoveryAuditResult) -> None:
    """Abort rather than silently leaving any scoped malformed shard behind."""
    blocked = [
        f"{region.stem}: {region.blocked_reason}"
        for region in audit.regions
        if region.blocked_reason
    ]
    if blocked:
        raise RuntimeError(
            "Wikidata integrity audit blocked this region; its files were not changed: "
            + "; ".join(blocked)
        )


def _load_existing_core_for_publication(
    data_root: DataRoot,
    stem: str,
    core: object | None,
    *,
    required: bool,
) -> object | None:
    """Load finalized core artifacts when a repair changed or must republish them."""
    if core is not None or not required:
        return core
    from osm_polygon_wikidata_only.hf.publication import load_existing_core_artifacts

    return load_existing_core_artifacts(data_root, stem)


def _run_pre_publication_migration(
    data_root: DataRoot,
    input_stems: set[str],
) -> None:
    """Execute the safe pre-runtime Wikipedia-document migration sequence.

    This coordinator owns steps 2-8 of the documented ``sync-dir``
    ordering so that :func:`execute` stays focused on CLI concerns and
    runtime construction.  No network or Wikimedia collaborators are
    constructed here, so a crash before this function returns cannot
    strand unpublished output beyond the durable
    pending-publications manifest.

    Sequence (matching the documented contract):

    1. Load durable pending publication intent.
    2. Scope migration only to stems that still have legacy articles.
    3. Plan the migration read-only.
    4. Abort before runtime/network construction if the plan is unsafe.
    5. Persist publication intent before applying local migration.
    6. Apply migration atomically.
    7. Prepare/repoint manifests only after canonical data passes validation.
    """
    pending_stems = load_pending_publications(data_root)
    scoped_stems = input_stems | pending_stems
    legacy_stems = {path.stem for path in data_root.processed_articles.glob("*.parquet")}

    migration_plan = plan_migration(
        data_root.processed,
        stems=scoped_stems & legacy_stems,
    )
    if not migration_plan.is_safe_to_apply:
        blocked = list(migration_plan.blocked_stems)
        raise MigrationError(
            f"Plan is not safe to apply: {len(blocked)} blocked stem(s): {blocked}"
        )

    stems_to_persist = _migration_stems_to_persist(migration_plan.stems)
    add_pending_publications(data_root, stems_to_persist)

    apply_migration(migration_plan)
    for stem in sorted(stems_to_persist):
        prepare_local_retirement(data_root, stem)


def execute(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    build_upload_files: Callable[..., list[PublicationOp]] | None = None,
    _remote_inventory: RemoteInventory | None = None,
    _hub: HfHub | None = None,
) -> int:
    """Run the ``sync-dir`` CLI command by wiring collaborators to
    :func:`pipeline.sync_runner.run_sync`.

    The CLI shell owns the unified-plan count log line and
    decides whether publication assembly runs. When
    ``--push`` is disabled, both ``build_upload_files`` and
    ``submit_upload`` are ``None`` and the runner never invokes
    the publication assembly.

    When ``--push`` is enabled, the CLI shell builds the region
    file list through
    :func:`hf.publication.assemble_region_upload` (a pure assembler
    that returns the ordered list) and the runner submits it via
    the upload queue. The CLI shell and runner together produce
    exactly ONE atomic commit per region: the assembler never
    submits, and the runner submits the assembled list exactly
    once. The unified-sync path silently swallows the legacy
    world-land exception (the ``warning_callback`` is ``None``).
    """
    push_enabled = bool(getattr(args, "push", False))
    dry_run = bool(getattr(args, "dry_run", False))
    from osm_polygon_wikidata_only.pipeline.containment_migration import (
        load_retired_children,
        load_retired_parent_children,
        prepare_safe_rules,
    )

    _prepare_containment_rules(
        enabled=push_enabled,
        data_path=data_root.path,
        dry_run=dry_run,
        prepare_safe_rules=prepare_safe_rules,
    )

    retired_children = load_retired_children(data_root.processed)
    pbfs = _active_pbfs(collect_pbfs([args.input]), retired_children)
    input_stems = {pbf.name.removesuffix(".osm.pbf") for pbf in pbfs}

    _run_pre_publication_migration(data_root, input_stems)

    # Construct remote collaborators after migration is applied
    runtime = build_wikimedia_runtime(settings, data_root=data_root)
    augmentation_client = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )

    planner_cls = None
    canonical_region_paths = None
    if push_enabled:
        # Keep these imports lazy.  Some local-only callers replace the
        # manifest loader while exercising the shell; importing reconciliation
        # during that window would capture the temporary replacement and leak
        # it into later push-enabled runs.
        from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlanner
        from osm_polygon_wikidata_only.hf.repo_layout import canonical_region_paths

        planner_cls = ReconciliationPlanner

    remote_state = _prepare_remote_reconciliation(
        enabled=push_enabled,
        data_root=data_root,
        settings=settings,
        input_stems=input_stems,
        hub=_hub,
        inventory_override=_remote_inventory,
        validate_augmentation=_validate_local_augmentation_state,
        load_retired_parent_children=load_retired_parent_children,
        canonical_region_paths=canonical_region_paths,
        planner_cls=planner_cls,
    )
    reconciliation_plan = remote_state.plan
    stems_with_gaps = remote_state.stems_with_gaps
    core_repaired = remote_state.core_repaired
    augmentation_current = remote_state.augmentation_current
    containment_publications = remote_state.containment_publications
    core_will_be_repaired = False

    # 5. Plan sync states
    all_pending_stems = load_pending_publications(data_root)
    if push_enabled:
        all_pending_stems = all_pending_stems | stems_with_gaps

    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    core_stems = {name.removesuffix(".osm.pbf") for name in entries}
    if push_enabled:
        current_augmentation = {stem for stem, current in augmentation_current.items() if current}
    else:
        local_current = _validate_local_augmentation_state(data_root, sorted(core_stems))
        current_augmentation = {stem for stem, current in local_current.items() if current}

    states = _plan_sync_states(
        pbfs,
        input_stems=input_stems,
        core_stems=core_stems,
        current_augmentation=current_augmentation,
        force=settings.force or not settings.skip_existing,
        pending_stems=all_pending_stems,
        recovery_stems=set(),
        processed_path=data_root.processed,
        plan_link_migration=plan_link_migration,
    )
    if push_enabled and reconciliation_plan is not None:
        missing = set(reconciliation_plan.missing)
        for state in states:
            core_will_be_repaired = core_will_be_repaired or _core_repair_required(
                state.action,
                state.stem,
                missing,
            )

    upload_queue = _build_upload_queue(
        push=push_enabled,
        dry_run=getattr(args, "dry_run", False),
        settings=settings,
        data_root=data_root,
        num_threads=getattr(args, "upload_threads", 2),
        _hub=_hub,
    )

    containment_enqueued = False
    if push_enabled and containment_publications:
        from osm_polygon_wikidata_only.hf.publication import (
            assemble_containment_retirement_upload,
        )

        containment_ops = assemble_containment_retirement_upload(
            data_root=data_root,
            repo_id=settings.repo_id,
            parent_children=containment_publications,
            world_land_warning=None,
        )
        if upload_queue is not None:
            upload_queue.submit(
                containment_ops,
                "Retire losslessly contained regional dataset shards",
            )
            containment_enqueued = True
            LOGGER.info(
                "Enqueued containment retirement for %d child region(s)",
                sum(len(children) for children in containment_publications.values()),
            )

    counts = {action: sum(state.action is action for state in states) for action in SyncAction}
    LOGGER.info(
        "Unified sync plan: %d recovery audit, %d augmentation backlog, %d publish, %d core missing, %d complete",
        counts[SyncAction.RECOVERY],
        counts[SyncAction.AUGMENT],
        counts[SyncAction.PUBLISH],
        counts[SyncAction.PROCESS],
        counts[SyncAction.COMPLETE],
    )

    # Capture settings + clients once so the bound extraction/process
    # collaborators do not need to look them up at call time.
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

    from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
    from osm_polygon_wikidata_only.hf.publication import (
        assemble_metadata_only_upload,
        assemble_region_upload,
    )
    from osm_polygon_wikidata_only.pipeline import sync_runner as sync_runner_mod
    from osm_polygon_wikidata_only.pipeline.sync_heartbeat import SyncHeartbeat

    application = SyncApplication(
        context=SyncApplicationContext(
            data_root=data_root,
            settings=settings,
            runtime=runtime,
            augmentation_client=augmentation_client,
            states=states,
            push_enabled=push_enabled,
            dry_run=dry_run,
            pending_stems=all_pending_stems,
            stems_with_gaps=stems_with_gaps,
            reconciliation_plan=reconciliation_plan,
            upload_queue=upload_queue,
            publish_builder=build_upload_files,
            core_will_be_repaired=core_will_be_repaired,
            core_repaired=core_repaired,
            containment_enqueued=containment_enqueued,
        ),
        services=SyncApplicationServices(
            extract_pbf=_extract,
            process_extracted_pbf=_process,
            augment_region=augment_region,
            load_existing_augmentation=load_existing_augmentation_result,
            recover_region=lambda state: None,
            run_sync=sync_runner_mod.run_sync,
            plan_link_migration=plan_link_migration,
            apply_link_migration=apply_link_migration,
            audit_wikidata_integrity=audit_wikidata_integrity,
            ensure_recovery_audit_unblocked=_ensure_recovery_audit_unblocked,
            repair_wikidata_region=repair_wikidata_region,
            prepare_local_retirement=prepare_local_retirement,
            add_pending_publications=add_pending_publications,
            record_region_recovery_receipt=record_region_recovery_receipt,
            assemble_region_upload=assemble_region_upload,
            assemble_metadata_only_upload=assemble_metadata_only_upload,
            load_existing_core_for_publication=_load_existing_core_for_publication,
            commit_message=_commit_message(getattr(args, "commit_message", None)),
            log_remote_reconciliation_summary=_log_remote_reconciliation_summary,
            load_metadata_refresh_marker=load_metadata_refresh_marker,
            clear_metadata_refresh_marker=clear_metadata_refresh_marker,
            augmentation_progress=AugmentationProgress,
            sync_heartbeat=SyncHeartbeat,
            logger=LOGGER,
        ),
    )
    return application.run().return_code


def _log_remote_reconciliation_summary(
    *,
    stems_with_gaps: set[str],
    core_repaired: bool,
    metadata_repaired: bool,
    log: Callable[[str], None] = LOGGER.info,
) -> None:
    """Emit the final remote-reconciliation summary line.

    The summary is derived strictly from signals produced by the
    upload pipeline -- it must never claim maps or README were
    refreshed unless a core or metadata-only publication actually
    refreshed them. Claims are made only after the background
    upload queue has drained successfully.

    The ``log`` parameter is the ``info``-level callable that
    receives the rendered message. Tests pass a recorded logger
    spy to observe emissions without depending on caplog state
    or module-level logger configuration.
    """
    log(
        _reconciliation_summary_message(
            len(stems_with_gaps),
            core_repaired,
            metadata_repaired,
        )
    )


def _reconciliation_summary_message(
    repaired_regions: int,
    core_repaired: bool,
    metadata_repaired: bool,
) -> str:
    """Build the stable summary text from upload outcomes."""
    maps_refreshed = core_repaired or metadata_repaired
    if maps_refreshed:
        if repaired_regions:
            return (
                f"Remote reconciliation complete: {repaired_regions} "
                "regions repaired; README and maps refreshed"
            )
        return "Remote reconciliation complete: README and maps refreshed"
    if repaired_regions:
        return f"Remote reconciliation complete: {repaired_regions} regions repaired"
    return "Remote reconciliation complete: converged"


def _active_pbfs(pbfs: list[Path], retired_children: Collection[str]) -> list[Path]:
    """Return non-retired PBFs in the order supplied by the planner."""
    return [pbf for pbf in pbfs if pbf.name.removesuffix(".osm.pbf") not in retired_children]


def _migration_stems_to_persist(stems: Iterable[Any]) -> set[str]:
    """Select migration operations whose canonical output must be published."""
    return {
        stem_plan.stem
        for stem_plan in stems
        if stem_plan.operation
        in (MigrationOperation.CREATE_MISSING, MigrationOperation.UPGRADE_LEGACY)
    }


def _containment_publications_for_remote(
    parent_children: dict[str, tuple[str, ...]],
    *,
    inventory: RemoteInventory,
    canonical_region_paths: Callable[[str], dict[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Keep only contained children represented by a remote artifact."""
    publications: dict[str, tuple[str, ...]] = {}
    for parent, children in parent_children.items():
        present = tuple(
            child
            for child in children
            if _remote_child_has_artifact(child, inventory, canonical_region_paths)
        )
        if present:
            publications[parent] = present
    return publications


def _remote_child_has_artifact(
    child: str,
    inventory: RemoteInventory,
    canonical_region_paths: Callable[[str], dict[str, str]],
) -> bool:
    """Return whether any canonical artifact for a child is remote."""
    return any(inventory.contains(path) for path in canonical_region_paths(child).values())


def _reconciliation_gap_counts(
    input_stems: set[str],
    missing: set[tuple[str, str]],
) -> tuple[int, int]:
    """Count regions with missing core files and augmentation files."""
    augmentation_corpora = (
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    )
    core_count = sum(
        (stem, "polygons") in missing or (stem, "polygon_articles") in missing
        for stem in input_stems
    )
    augmentation_count = sum(
        any((stem, corpus) in missing for corpus in augmentation_corpora) for stem in input_stems
    )
    return core_count, augmentation_count


def _prepare_remote_reconciliation(
    *,
    enabled: bool,
    data_root: DataRoot,
    settings: Settings,
    input_stems: set[str],
    hub: HfHub | None,
    inventory_override: RemoteInventory | None,
    validate_augmentation: Callable[[DataRoot, list[str]], dict[str, bool]],
    load_retired_parent_children: Callable[[Path], dict[str, tuple[str, ...]]],
    canonical_region_paths: Callable[[str], dict[str, str]] | None = None,
    planner_cls: Any = None,
) -> _RemoteReconciliation:
    """Prepare remote reconciliation inputs without work on local-only runs."""
    if not enabled:
        return _RemoteReconciliation(None, None, {}, set(), {}, False)
    if canonical_region_paths is None or planner_cls is None:
        raise RuntimeError("Remote reconciliation helpers are required when push is enabled")

    augmentation_current = validate_augmentation(data_root, sorted(input_stems))
    inventory = (
        inventory_override
        if inventory_override is not None
        else RemoteInventory.fetch(
            repo_id=settings.repo_id,
            hub=hub,
            token=settings.hf_token,
        )
    )
    retired_groups = load_retired_parent_children(data_root.processed)
    containment_publications = _containment_publications_for_remote(
        retired_groups,
        inventory=inventory,
        canonical_region_paths=canonical_region_paths,
    )
    reconciliation_plan = planner_cls(
        data_root=data_root,
        inventory=inventory,
        stems=input_stems,
        augmentation_current=augmentation_current,
    ).plan()
    stems_with_gaps = set(reconciliation_plan.stems_to_publish) | set(
        reconciliation_plan.stems_to_augment
    )
    missing = set(reconciliation_plan.missing)
    core_repaired = any(
        _core_repair_required(SyncAction.PUBLISH, stem, missing) for stem in stems_with_gaps
    )
    missing_core_count, missing_aug_count = _reconciliation_gap_counts(
        input_stems,
        missing,
    )
    LOGGER.info(
        "Remote reconciliation: %d regions missing core artifacts, %d missing augmentation artifacts",
        missing_core_count,
        missing_aug_count,
    )
    return _RemoteReconciliation(
        inventory,
        reconciliation_plan,
        augmentation_current,
        stems_with_gaps,
        containment_publications,
        core_repaired,
    )


def _prepare_containment_rules(
    *,
    enabled: bool,
    data_path: Path,
    dry_run: bool,
    prepare_safe_rules: Callable[..., tuple[Collection[Any], Collection[Any]]],
    log_info: Callable[..., None] = LOGGER.info,
    log_warning: Callable[..., None] = LOGGER.warning,
) -> None:
    """Prepare lossless containment rules and report blocked candidates."""
    if not enabled:
        return
    prepared_rules, blocked_rules = prepare_safe_rules(data_path, dry_run=dry_run)
    if prepared_rules:
        log_info(
            "Prepared %d lossless contained-region retirement rule(s)",
            len(prepared_rules),
        )
    for blocked in blocked_rules:
        log_warning(
            "Containment retirement blocked for %s: %s",
            blocked.parent,
            "; ".join(blocked.blockers),
        )


def _core_repair_required(
    action: SyncAction,
    stem: str,
    missing: set[tuple[str, str]],
) -> bool:
    """Return whether a state changes core artifacts requiring refresh."""
    if action is SyncAction.PROCESS:
        return True
    if action not in (SyncAction.PUBLISH, SyncAction.AUGMENT):
        return False
    return (stem, "polygons") in missing or (stem, "polygon_articles") in missing


def _plan_sync_states(
    pbfs: list[Path],
    *,
    input_stems: set[str],
    core_stems: set[str],
    current_augmentation: set[str],
    force: bool,
    pending_stems: set[str],
    recovery_stems: set[str],
    processed_path: Path,
    plan_link_migration: Callable[..., Any],
) -> list[RegionSyncState]:
    """Build the deterministic action plan, including link migrations."""
    recovery = set(recovery_stems)
    recovery.update(
        _recovery_audit_stems(
            input_stems=input_stems,
            core_stems=core_stems,
            current_augmentation=current_augmentation,
            force=force,
        )
    )
    link_plan = plan_link_migration(processed_path, stems=input_stems)
    recovery.update(
        stem.stem for stem in link_plan.stems if stem.classification.value != "canonical"
    )
    return plan_sync_states(
        pbfs,
        core_stems=core_stems,
        augmentation_stems=current_augmentation,
        force=force,
        pending_stems=pending_stems,
        recovery_stems=recovery,
    )


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


def _validate_local_augmentation_state(
    data_root: DataRoot,
    stems: list[str],
) -> dict[str, bool]:
    """Validate the local augmentation state for every stem exactly once.

    Wraps :class:`LocalValidationProgress` so the operator gets a
    bounded, periodic startup progress signal even when this phase
    takes several minutes. Each stem is visited exactly once and
    the resulting mapping is returned for downstream planning.
    """
    progress = LocalValidationProgress(
        validator=lambda stem: augmentation_is_current(data_root, stem),
        stems=list(stems),
        log=LOGGER.info,
        clock=time.monotonic,
        progress_interval_s=30.0,
        quiet_threshold=25,
        phase_label="regions",
    )
    return progress.run()


def _post_upload_publication_cleanup(
    data_root: DataRoot,
    ops: list[PublicationOp],
    *,
    dry_run: bool,
) -> None:
    """Retire local legacy articles and clear pending intent after a confirmed upload.

    Runs *after* the Hub upload succeeds. A stem is retired only when
    the operation list contains BOTH the canonical
    ``add wikipedia/documents/<stem>.parquet`` and the matching legacy
    ``delete articles/<stem>.parquet`` for the same stem. An add
    without its matching delete, a delete for another stem, a delete
    without an add, nested or traversal paths, lookalike prefixes, or
    conflicting duplicate adds do NOT authorize local retirement or
    pending-intent cleanup. Pending intent is cleared only after
    every selected stem's local retirement succeeds.

    The commit message is never inspected; stems are derived strictly
    from ``PublicationOp`` entries that match the canonical layout.
    When ``dry_run`` is true the local filesystem is left untouched so
    repeated dry-runs remain safe.
    """
    if dry_run:
        return

    paired_stems = _paired_retirement_stems(data_root, ops)
    if not paired_stems:
        return

    retired: list[str] = []
    for stem in sorted(paired_stems):
        finalize_local_retirement(data_root, stem)
        retired.append(stem)

    remove_pending_publications(data_root, set(retired))


def _execute_upload_job(
    *,
    data_root: DataRoot,
    settings: Settings,
    ops: list[PublicationOp],
    message: str,
    num_threads: int,
    hub: HfHub | None,
    dry_run: bool,
) -> None:
    """Production upload-job callback: upload, then clean up local state.

    The upload is the network boundary. When it raises, the cleanup
    helper is never invoked, so the local legacy staging file and the
    durable pending-publications manifest survive intact for the next
    invocation to retry.

    When the upload queue has snapshotted an op (``op.snapshot_path``
    is set), the upload reads from the snapshot rather than the
    canonical local_path so the upload is durable across canonical
    mutation. The canonical local_path is preserved on the op for
    the post-upload retirement check.
    """
    upload_ops = [
        PublicationOp(
            action=op.action,
            path_in_repo=op.path_in_repo,
            local_path=op.snapshot_path or op.local_path,
        )
        for op in ops
    ]
    upload_files(
        settings.repo_id,
        ops=upload_ops,
        hub=hub,
        token=settings.hf_token,
        commit_message=message,
        num_threads=num_threads,
    )
    _post_upload_publication_cleanup(data_root, ops, dry_run=dry_run)


def _build_upload_queue(
    *,
    push: bool,
    dry_run: bool,
    settings: Settings,
    data_root: DataRoot,
    num_threads: int,
    _hub: HfHub | None = None,
) -> BackgroundUploadQueue | None:
    """Open the documented ``BackgroundUploadQueue`` and resume pending jobs."""
    if not push:
        return None

    hub = _hub if _hub is not None else (StubHfHub() if dry_run else None)

    def upload_job(ops: list[PublicationOp], message: str) -> None:
        _execute_upload_job(
            data_root=data_root,
            settings=settings,
            ops=ops,
            message=message,
            num_threads=num_threads,
            hub=hub,
            dry_run=dry_run,
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
