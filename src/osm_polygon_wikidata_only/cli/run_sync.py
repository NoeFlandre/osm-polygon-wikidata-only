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
from collections.abc import Callable
from dataclasses import replace
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
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_ARTICLES_DIR,
    REMOTE_WIKIPEDIA_DOCUMENTS_DIR,
)
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.local_validation import LocalValidationProgress
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
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

    stems_to_persist = {
        stem_plan.stem
        for stem_plan in migration_plan.stems
        if stem_plan.operation
        in (MigrationOperation.CREATE_MISSING, MigrationOperation.UPGRADE_LEGACY)
    }
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

    if push_enabled:
        prepared_rules, blocked_rules = prepare_safe_rules(data_root.path, dry_run=dry_run)
        if prepared_rules:
            LOGGER.info(
                "Prepared %d lossless contained-region retirement rule(s)", len(prepared_rules)
            )
        for blocked in blocked_rules:
            LOGGER.warning(
                "Containment retirement blocked for %s: %s",
                blocked.parent,
                "; ".join(blocked.blockers),
            )

    retired_children = load_retired_children(data_root.processed)
    pbfs = [
        pbf
        for pbf in collect_pbfs([args.input])
        if pbf.name.removesuffix(".osm.pbf") not in retired_children
    ]
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

    remote_inventory = None
    reconciliation_plan = None
    stems_with_gaps = set()
    core_will_be_repaired = False
    core_repaired = False
    augmentation_current = {}
    containment_publications: dict[str, tuple[str, ...]] = {}

    if push_enabled:
        from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlanner

        # Compute current augmentation state exactly once per input
        # stem, with bounded startup progress visibility.
        augmentation_current = _validate_local_augmentation_state(data_root, sorted(input_stems))

        if _remote_inventory is not None:
            remote_inventory = _remote_inventory
        else:
            remote_inventory = RemoteInventory.fetch(
                repo_id=settings.repo_id,
                hub=_hub,
                token=settings.hf_token,
            )
        retired_groups = load_retired_parent_children(data_root.processed)
        from osm_polygon_wikidata_only.hf.repo_layout import canonical_region_paths

        containment_publications = {
            parent: tuple(
                child
                for child in children
                if any(
                    remote_inventory.contains(path)
                    for path in canonical_region_paths(child).values()
                )
            )
            for parent, children in retired_groups.items()
        }
        containment_publications = {
            parent: children for parent, children in containment_publications.items() if children
        }
        planner = ReconciliationPlanner(
            data_root=data_root,
            inventory=remote_inventory,
            stems=input_stems,
            augmentation_current=augmentation_current,
        )
        reconciliation_plan = planner.plan()
        stems_with_gaps = set(reconciliation_plan.stems_to_publish) | set(
            reconciliation_plan.stems_to_augment
        )

        core_repaired = any(
            (stem, "polygons") in reconciliation_plan.missing
            or (stem, "polygon_articles") in reconciliation_plan.missing
            for stem in stems_with_gaps
        )

        # Log remote reconciliation progress
        missing_core_count = sum(
            1
            for stem in input_stems
            if (stem, "polygons") in reconciliation_plan.missing
            or (stem, "polygon_articles") in reconciliation_plan.missing
        )
        missing_aug_count = sum(
            1
            for stem in input_stems
            if any(
                (stem, corpus) in reconciliation_plan.missing
                for corpus in [
                    "wikipedia/documents",
                    "wikipedia/sections",
                    "wikivoyage/documents",
                    "wikivoyage/sections",
                    "wikidata/facts",
                ]
            )
        )
        LOGGER.info(
            "Remote reconciliation: %d regions missing core artifacts, %d missing augmentation artifacts",
            missing_core_count,
            missing_aug_count,
        )

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

    recovery_stems = set(
        _recovery_audit_stems(
            input_stems=input_stems,
            core_stems=core_stems,
            current_augmentation=current_augmentation,
            force=settings.force or not settings.skip_existing,
        )
    )
    states = plan_sync_states(
        pbfs,
        core_stems=core_stems,
        augmentation_stems=current_augmentation,
        force=settings.force or not settings.skip_existing,
        pending_stems=all_pending_stems,
        recovery_stems=recovery_stems,
    )
    recovered_stems: set[str] = set()
    recovery_map_refresh_stems: set[str] = set()

    if push_enabled and reconciliation_plan is not None:
        for state in states:
            if state.action == SyncAction.PROCESS:
                core_will_be_repaired = True
            elif state.action in (SyncAction.PUBLISH, SyncAction.AUGMENT):
                stem = state.stem
                if (stem, "polygons") in reconciliation_plan.missing or (
                    stem,
                    "polygon_articles",
                ) in reconciliation_plan.missing:
                    core_will_be_repaired = True

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
            audit = audit_wikidata_integrity(
                data_root,
                [state.stem],
                runtime.wikidata,
                batch_size=settings.enrichment_batch_size,
                languages=settings.languages,
                max_articles_per_qid=settings.max_articles_per_qid,
                log=LOGGER.info,
            )
            _ensure_recovery_audit_unblocked(audit)
            region = audit.region(state.stem)
            if region.requires_repair:
                repair_result = repair_wikidata_region(
                    data_root,
                    region,
                    wikidata_client=runtime.wikidata,
                    wikipedia_client=runtime.wikipedia,
                    augmentation_client=augmentation_client,
                    settings=settings,
                    log=LOGGER.info,
                    scheduler_snapshot=runtime.scheduler.snapshot,
                )
                if repair_result.changed:
                    recovered_stems.add(state.stem)
                    if repair_result.map_inputs_changed:
                        recovery_map_refresh_stems.add(state.stem)
                    augmentation_result = load_existing_augmentation_result(data_root, state.stem)
        LOGGER.info(
            "Unified sync completed %s: %s",
            state.stem,
            augmentation_result.counts,
        )
        return augmentation_result

    def _load_existing(state: RegionSyncState) -> Any:
        return load_existing_augmentation_result(data_root, state.stem)

    def _recover(state: RegionSyncState) -> Any:
        audit = audit_wikidata_integrity(
            data_root,
            [state.stem],
            runtime.wikidata,
            batch_size=settings.enrichment_batch_size,
            languages=settings.languages,
            max_articles_per_qid=settings.max_articles_per_qid,
            log=LOGGER.info,
        )
        _ensure_recovery_audit_unblocked(audit)
        plan = audit.region(state.stem)
        if not plan.requires_repair:
            if push_enabled and state.stem in all_pending_stems:
                return load_existing_augmentation_result(data_root, state.stem)
            return None
        repair_result = repair_wikidata_region(
            data_root,
            plan,
            wikidata_client=runtime.wikidata,
            wikipedia_client=runtime.wikipedia,
            augmentation_client=augmentation_client,
            settings=settings,
            log=LOGGER.info,
            scheduler_snapshot=runtime.scheduler.snapshot,
        )
        if not repair_result.changed:
            if push_enabled and state.stem in all_pending_stems:
                return load_existing_augmentation_result(data_root, state.stem)
            return None
        recovered_stems.add(state.stem)
        if repair_result.map_inputs_changed:
            recovery_map_refresh_stems.add(state.stem)
        return load_existing_augmentation_result(data_root, state.stem)

    def _prepare_publication(state: RegionSyncState, result: Any) -> None:
        # Test doubles and third-party collaborators predating canonical
        # documents may not produce a sidecar. Production augmentation
        # always returns this path; only then is retirement applicable.
        if getattr(result, "wikipedia_documents_path", None) is None:
            return
        prepare_local_retirement(data_root, state.stem)
        add_pending_publications(data_root, {state.stem})

    def _submit_upload(ops: list[PublicationOp], message: str) -> None:
        if upload_queue is None:
            return
        upload_queue.submit(ops, message)

    def _build_region_publication(
        state: object,
        augmentation: object,
        core: object | None,
    ) -> list[PublicationOp]:
        """Region-publication builder injected into ``pipeline.sync_runner``.

        Pure assembly: delegates to
        :func:`hf.publication.assemble_region_upload`, which
        returns the ordered op list and performs NO upload.
        Submission is the upload queue's responsibility, executed
        exactly once by ``_maybe_submit`` in the runner. The
        unified-sync world-land fallback is silent (``None``).
        """
        from osm_polygon_wikidata_only.hf.publication import assemble_region_upload

        stem = getattr(state, "stem", "")
        needs_existing_core = stem in recovered_stems or (
            push_enabled
            and reconciliation_plan is not None
            and (
                (stem, "polygons") in reconciliation_plan.missing
                or (stem, "polygon_articles") in reconciliation_plan.missing
            )
        )
        if core is None and needs_existing_core:
            LOGGER.info(
                "Repairing remote region %s from finalized local artifacts (no Wikimedia requests)",
                stem,
            )
            try:
                core = _load_existing_core_for_publication(
                    data_root,
                    stem,
                    core,
                    required=True,
                )
            except Exception as e:
                LOGGER.error("Failed to load local core artifacts for %s: %s", stem, e)
                raise

        if stem in recovered_stems and core is None:
            raise RuntimeError(f"Recovered region {stem!r} has no core publication artifacts")

        return assemble_region_upload(
            data_root=data_root,
            repo_id=settings.repo_id,
            stem=stem,
            augmentation=augmentation,  # type: ignore[arg-type]
            core=core,  # type: ignore[arg-type]
            world_land_warning=None,
            refresh_maps=(
                getattr(state, "action", None) is not SyncAction.RECOVERY
                or stem in recovery_map_refresh_stems
            ),
        )

    def _close_uploads() -> list[str]:
        if upload_queue is None:
            return []
        return upload_queue.close_and_wait()

    # Push-disabled: do not even hand the runner a publication
    # builder. The runner will skip _maybe_submit entirely. When the
    # CLI shell is given an override builder (legacy compatibility),
    # use it; otherwise default to the production
    # ``hf.publication.assemble_region_upload`` builder.
    publish_builder: Callable[..., list[PublicationOp]] | None = (
        (build_upload_files or _build_region_publication) if push_enabled else None
    )
    submit_callback: Callable[[list[PublicationOp], str], None] | None = (
        _submit_upload if push_enabled else None
    )

    from osm_polygon_wikidata_only.pipeline import sync_runner as sync_runner_mod

    rc = 0
    metadata_repaired = False
    success = False
    try:
        rc = sync_runner_mod.run_sync(
            states,
            extract_pbf=_extract,
            process_extracted_pbf=_process,
            augment_region=_augment,
            build_upload_files=publish_builder,
            commit_message=_commit_message(getattr(args, "commit_message", None)),
            submit_upload=submit_callback,
            close_uploads=None,
            load_existing_augmentation=_load_existing,
            recover_region=_recover,
            on_complete=_prepare_publication,
        )
        core_will_be_repaired = core_will_be_repaired or bool(recovered_stems)
        if (
            rc == 0
            and push_enabled
            and reconciliation_plan is not None
            and reconciliation_plan.repository_refresh
            and not core_will_be_repaired
            and not containment_enqueued
        ):
            # Enqueue metadata-only repair if needed and core won't be repaired by another upload
            from osm_polygon_wikidata_only.hf.publication import assemble_metadata_only_upload

            LOGGER.info("Enqueuing metadata-only repair (no region core repair planned)")
            ops = assemble_metadata_only_upload(
                data_root=data_root,
                repo_id=settings.repo_id,
                world_land_warning=None,
            )
            if submit_callback is not None:
                submit_callback(ops, "Repair remote repository metadata and maps")
                metadata_repaired = True
        success = rc == 0
    except Exception as error:
        if upload_queue is not None:
            LOGGER.error("Unified sync aborted: %s", error)
        raise
    finally:
        failures = _close_uploads()
        if failures:
            rc = 1
            success = False

    if success and rc == 0 and not failures:
        if push_enabled:
            core_repaired = core_repaired or bool(recovered_stems)
            _log_remote_reconciliation_summary(
                stems_with_gaps=stems_with_gaps,
                core_repaired=core_repaired,
                metadata_repaired=metadata_repaired,
                log=LOGGER.info,
            )
        return 0
    else:
        if rc != 0:
            LOGGER.error("Unified sync completed with failures (rc=%d)", rc)
        return rc or 1


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
    maps_refreshed = core_repaired or metadata_repaired
    if maps_refreshed and len(stems_with_gaps) > 0:
        log(
            f"Remote reconciliation complete: {len(stems_with_gaps)} "
            "regions repaired; README and maps refreshed"
        )
    elif maps_refreshed:
        log("Remote reconciliation complete: README and maps refreshed")
    elif len(stems_with_gaps) > 0:
        log(f"Remote reconciliation complete: {len(stems_with_gaps)} regions repaired")
    else:
        log("Remote reconciliation complete: converged")


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


def _is_valid_stem(stem: str) -> bool:
    """A stem must be non-empty, free of traversal and path separators."""
    if not stem or stem in {".", ".."}:
        return False
    return not ("/" in stem or "\\" in stem)


def _paired_retirement_stems(data_root: DataRoot, ops: list[PublicationOp]) -> set[str]:
    """Return stems whose ops list contains a correctly-paired add+delete.

    Each canonical add must:

    * use ``path_in_repo == wikipedia/documents/<stem>.parquet`` exactly
      (no nested paths, no traversal, no lookalike prefixes);
    * carry a ``local_path`` whose resolved absolute path equals
      ``data_root.processed / wikipedia / documents / <stem>.parquet``.

    Each legacy delete must use ``path_in_repo == articles/<stem>.parquet``
    exactly with the same restrictions.

    Duplicate or conflicting canonical add operations for the same stem
    cause that stem to be excluded (fail closed). Other correctly-paired
    stems remain authorized.
    """
    canonical_add_count: dict[str, int] = {}
    canonical_adds_valid: dict[str, Path] = {}
    legacy_deletes: set[str] = set()

    for op in ops:
        path_in_repo = op.path_in_repo
        if not isinstance(path_in_repo, str) or not path_in_repo:
            continue
        stem = Path(path_in_repo).stem
        if not _is_valid_stem(stem):
            continue
        if op.action == "add":
            expected_remote = f"{REMOTE_WIKIPEDIA_DOCUMENTS_DIR}/{stem}.parquet"
            if path_in_repo != expected_remote:
                continue
            canonical_add_count[stem] = canonical_add_count.get(stem, 0) + 1
            local = op.local_path
            if local is None:
                continue
            try:
                resolved = Path(local).resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            expected_local = (
                data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
            ).resolve()
            if resolved != expected_local:
                continue
            if not expected_local.is_file():
                continue
            prior = canonical_adds_valid.get(stem)
            if prior is not None and prior != resolved:
                canonical_adds_valid[stem] = Path("__conflict__")
            else:
                canonical_adds_valid[stem] = resolved
        elif op.action == "delete":
            expected_remote = f"{LEGACY_REMOTE_ARTICLES_DIR}/{stem}.parquet"
            if path_in_repo != expected_remote:
                continue
            legacy_deletes.add(stem)

    single_add_stems = {stem for stem, count in canonical_add_count.items() if count == 1}
    valid_singles = {
        stem
        for stem, marker in canonical_adds_valid.items()
        if marker != Path("__conflict__") and stem in single_add_stems
    }
    return valid_singles & legacy_deletes


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
    """
    upload_files(
        settings.repo_id,
        ops=ops,
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
