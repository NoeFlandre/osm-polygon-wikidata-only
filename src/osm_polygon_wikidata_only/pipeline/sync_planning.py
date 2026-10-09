"""Local migration and deterministic state planning for ``sync-dir``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    MigrationError,
    MigrationOperation,
    StemPlan,
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import prepare_local_retirement
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.link_migration import plan_link_migration
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs
from osm_polygon_wikidata_only.pipeline.pending_publications import (
    add_pending_publications,
    load_pending_publications,
)
from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)
from osm_polygon_wikidata_only.pipeline.sync_reconciliation import (
    RemoteReconciliation,
    core_repair_required,
    prepare_remote_reconciliation,
    remote_reconciliation_helpers,
    validate_local_augmentation_state,
)
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import RecoveryAuditResult

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
    from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlan
    from osm_polygon_wikidata_only.pipeline._link_migration.models import (
        MigrationPlan as LinkMigrationPlan,
    )
    from osm_polygon_wikidata_only.pipeline.containment_migration import PreparedRule, RuleAudit

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


@dataclass(slots=True)
class PreparedSyncPlan:
    """Inputs and derived state needed by the sync application."""

    pbfs: list[Path]
    input_stems: set[str]
    all_pending_stems: set[str]
    states: list[RegionSyncState]
    remote_state: RemoteReconciliation
    core_will_be_repaired: bool


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


def ensure_recovery_audit_unblocked(audit: RecoveryAuditResult) -> None:
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


def run_pre_publication_migration(
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


def prepare_sync_plan(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
    push_enabled: bool,
    dry_run: bool,
    remote_inventory: RemoteInventory | None,
    hub: HfHub | None,
) -> PreparedSyncPlan:
    """Prepare local migration, remote reconciliation, and sync states."""
    # The migration stack is needed only for a selected sync operation.
    from osm_polygon_wikidata_only.pipeline.containment_migration import (  # noqa: PLC0415
        load_retired_parent_children,
    )

    pbfs, input_stems = _prepare_local_inputs(
        args,
        data_root=data_root,
        push_enabled=push_enabled,
        dry_run=dry_run,
    )
    planner_cls, canonical_region_paths = remote_reconciliation_helpers(push_enabled)
    remote_state = prepare_remote_reconciliation(
        enabled=push_enabled,
        data_root=data_root,
        settings=settings,
        input_stems=input_stems,
        hub=hub,
        inventory_override=remote_inventory,
        validate_augmentation=validate_local_augmentation_state,
        load_retired_parent_children=load_retired_parent_children,
        canonical_region_paths=canonical_region_paths,
        planner_cls=planner_cls,
    )
    return _plan_prepared_sync(
        pbfs,
        input_stems=input_stems,
        remote_state=remote_state,
        data_root=data_root,
        settings=settings,
        push_enabled=push_enabled,
    )


def _prepare_local_inputs(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    push_enabled: bool,
    dry_run: bool,
) -> tuple[list[Path], set[str]]:
    """Prepare containment rules, select active inputs, and migrate them locally."""
    # The migration stack is needed only for a selected sync operation.
    from osm_polygon_wikidata_only.pipeline.containment_migration import (  # noqa: PLC0415
        load_retired_children,
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
    run_pre_publication_migration(data_root, input_stems)
    return pbfs, input_stems


def _plan_prepared_sync(
    pbfs: list[Path],
    *,
    input_stems: set[str],
    remote_state: RemoteReconciliation,
    data_root: DataRoot,
    settings: Settings,
    push_enabled: bool,
) -> PreparedSyncPlan:
    """Derive per-region sync states from local manifests and remote state."""
    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    core_stems = {name.removesuffix(".osm.pbf") for name in entries}
    current_augmentation = _current_augmentation_for_plan(
        push_enabled,
        remote_state.augmentation_current,
        data_root,
        core_stems,
    )
    all_pending_stems = _pending_stems_for_plan(
        data_root,
        remote_state.stems_with_gaps,
        push_enabled=push_enabled,
    )
    states = plan_sync_states_with_recovery(
        pbfs,
        inventory=StemInventory(
            input_stems=input_stems,
            core_stems=core_stems,
            current_augmentation=current_augmentation,
        ),
        force=settings.force or not settings.skip_existing,
        pending_stems=all_pending_stems,
        recovery_stems=set(),
        processed_path=data_root.processed,
        plan_link_migration=plan_link_migration,
    )
    return PreparedSyncPlan(
        pbfs=pbfs,
        input_stems=input_stems,
        all_pending_stems=all_pending_stems,
        states=states,
        remote_state=remote_state,
        core_will_be_repaired=_core_will_be_repaired(
            states,
            remote_state.plan,
            push_enabled=push_enabled,
        ),
    )


def _current_augmentation_for_plan(
    push_enabled: bool,
    remote_current: dict[str, bool],
    data_root: DataRoot,
    core_stems: set[str],
) -> set[str]:
    current = (
        remote_current
        if push_enabled
        else validate_local_augmentation_state(data_root, sorted(core_stems))
    )
    return {stem for stem, is_current in current.items() if is_current}


def _pending_stems_for_plan(
    data_root: DataRoot,
    stems_with_gaps: set[str],
    *,
    push_enabled: bool,
) -> set[str]:
    pending = load_pending_publications(data_root)
    return pending | stems_with_gaps if push_enabled else pending


def _core_will_be_repaired(
    states: list[RegionSyncState],
    reconciliation_plan: ReconciliationPlan | None,
    *,
    push_enabled: bool,
) -> bool:
    if not push_enabled or reconciliation_plan is None:
        return False
    missing = set(reconciliation_plan.missing)
    return any(core_repair_required(state.action, state.stem, missing) for state in states)


def log_sync_plan(states: list[RegionSyncState]) -> None:
    counts = {action: sum(state.action is action for state in states) for action in SyncAction}
    LOGGER.info(
        "Unified sync plan: %d recovery audit, %d augmentation backlog, %d publish, %d core missing, %d complete",
        counts[SyncAction.RECOVERY],
        counts[SyncAction.AUGMENT],
        counts[SyncAction.PUBLISH],
        counts[SyncAction.PROCESS],
        counts[SyncAction.COMPLETE],
    )


def _active_pbfs(pbfs: list[Path], retired_children: Collection[str]) -> list[Path]:
    """Return non-retired PBFs in the order supplied by the planner."""
    return [pbf for pbf in pbfs if pbf.name.removesuffix(".osm.pbf") not in retired_children]


def _migration_stems_to_persist(stems: Iterable[StemPlan]) -> set[str]:
    """Select migration operations whose canonical output must be published."""
    return {
        stem_plan.stem
        for stem_plan in stems
        if stem_plan.operation
        in (MigrationOperation.CREATE_MISSING, MigrationOperation.UPGRADE_LEGACY)
    }


def _prepare_containment_rules(
    *,
    enabled: bool,
    data_path: Path,
    dry_run: bool,
    prepare_safe_rules: Callable[..., tuple[Collection[PreparedRule], Collection[RuleAudit]]],
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


@dataclass(frozen=True, slots=True)
class StemInventory:
    """Stem sets read from the processed, core and augmentation stores."""

    input_stems: set[str]
    core_stems: set[str]
    current_augmentation: set[str]


def plan_sync_states_with_recovery(
    pbfs: list[Path],
    *,
    inventory: StemInventory,
    force: bool,
    pending_stems: set[str],
    recovery_stems: set[str],
    processed_path: Path,
    plan_link_migration: Callable[..., LinkMigrationPlan],
) -> list[RegionSyncState]:
    """Build the deterministic action plan, including link migrations."""
    recovery = set(recovery_stems)
    recovery.update(
        _recovery_audit_stems(
            input_stems=inventory.input_stems,
            core_stems=inventory.core_stems,
            current_augmentation=inventory.current_augmentation,
            force=force,
        )
    )
    link_plan = plan_link_migration(processed_path, stems=inventory.input_stems)
    recovery.update(
        stem.stem for stem in link_plan.stems if stem.classification.value != "canonical"
    )
    return plan_sync_states(
        pbfs,
        core_stems=inventory.core_stems,
        augmentation_stems=inventory.current_augmentation,
        force=force,
        pending_stems=pending_stems,
        recovery_stems=recovery,
    )
