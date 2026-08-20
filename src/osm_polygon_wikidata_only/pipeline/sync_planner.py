"""Pure planning of work required to converge regional dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SyncAction(StrEnum):
    RECOVERY = "recovery"
    AUGMENT = "augment"
    PROCESS = "process"
    PUBLISH = "publish"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class RegionSyncState:
    stem: str
    pbf_path: Path
    action: SyncAction


def _action_for_stem(
    stem: str,
    *,
    core_stems: set[str],
    augmentation_stems: set[str],
    force: bool,
    pending_stems: set[str],
    recovery_stems: set[str],
) -> SyncAction:
    """Choose the highest-priority action for one region."""
    if force or stem not in core_stems:
        return SyncAction.PROCESS
    return _action_for_existing_stem(
        stem,
        augmentation_stems=augmentation_stems,
        pending_stems=pending_stems,
        recovery_stems=recovery_stems,
    )


def _action_for_existing_stem(
    stem: str,
    *,
    augmentation_stems: set[str],
    pending_stems: set[str],
    recovery_stems: set[str],
) -> SyncAction:
    """Choose an action after core artifacts are known to exist."""
    if stem not in augmentation_stems:
        return SyncAction.AUGMENT
    if stem in recovery_stems:
        return SyncAction.RECOVERY
    if stem in pending_stems:
        return SyncAction.PUBLISH
    return SyncAction.COMPLETE


def _sync_priority(action: SyncAction) -> int:
    """Return deterministic execution priority."""
    return {
        SyncAction.RECOVERY: 0,
        SyncAction.AUGMENT: 1,
        SyncAction.PUBLISH: 2,
        SyncAction.PROCESS: 3,
        SyncAction.COMPLETE: 4,
    }[action]


def plan_sync_states(
    pbfs: list[Path],
    *,
    core_stems: set[str],
    augmentation_stems: set[str],
    force: bool = False,
    pending_stems: set[str] | None = None,
    recovery_stems: set[str] | None = None,
) -> list[RegionSyncState]:
    """Classify PBFs and produce a deterministic action plan.

    Action priority (lowest value runs first):

    0. RECOVERY - incremental Wikidata integrity audit/recovery:
       the region is finalized locally and eligible for a bounded
       one-region audit. Healthy regions write/reuse a receipt and
       require no publication. Affected regions fetch only the
       damaged QIDs, preserve every healthy row, commit via a
       durable journal, then enqueue an atomic publication. No PBF
       extraction runs.
    1. AUGMENT - existing augmentation backlog (in-place fix-up of a
       region whose core is finalized but whose augmentation is stale
       or missing). AUGMENT performs Wikimedia sidecar work and, on
       success, enqueues an atomic remote publication for the region.
    2. PUBLISH - safe, Wikimedia-free publish-only reconciliation
       repairs (a finalized local artifact that the remote is
       missing). PUBLISH runs BEFORE PROCESS so a remotely missing
       finalized artifact is not blocked behind expensive new-core
       PBF extraction. The repair uses the already-loaded local
       augmentation result and only enqueues a Hugging Face upload;
       it does not invoke any Wikidata, Wikipedia, or Wikivoyage
       call.
    3. PROCESS - new core processing (extraction + enrichment +
       augmentation) for regions whose local core is missing. The
       runner may prefetch the next PBF concurrently while
       enriching the current region.
    4. COMPLETE - regions already converged (no action required).

    Within each priority bucket, states are sorted alphabetically by
    stem for deterministic execution.
    """
    pending = pending_stems or set()
    recovery = recovery_stems or set()
    states = [
        RegionSyncState(
            stem := pbf.name.removesuffix(".osm.pbf"),
            pbf,
            _action_for_stem(
                stem,
                core_stems=core_stems,
                augmentation_stems=augmentation_stems,
                force=force,
                pending_stems=pending,
                recovery_stems=recovery,
            ),
        )
        for pbf in pbfs
    ]
    return sorted(states, key=lambda state: (_sync_priority(state.action), state.stem))


__all__ = ["RegionSyncState", "SyncAction", "plan_sync_states"]
