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

    0. RECOVERY - exhaustive Wikidata integrity recovery: the
       region is finalized locally but the audit classifies one
       or more QIDs as ``REPAIR_REQUIRED``. Recovery fetches
       only the affected QIDs, preserves every healthy row,
       commits via a durable journal, then enqueues an atomic
       publication for the region. No PBF extraction runs.
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
    states: list[RegionSyncState] = []
    for pbf in pbfs:
        stem = pbf.name.removesuffix(".osm.pbf")
        if stem in recovery:
            action = SyncAction.RECOVERY
        elif force or stem not in core_stems:
            action = SyncAction.PROCESS
        elif stem not in augmentation_stems:
            action = SyncAction.AUGMENT
        elif stem in pending:
            action = SyncAction.PUBLISH
        else:
            action = SyncAction.COMPLETE
        states.append(RegionSyncState(stem, pbf, action))
    priority = {
        SyncAction.RECOVERY: 0,
        SyncAction.AUGMENT: 1,
        SyncAction.PUBLISH: 2,
        SyncAction.PROCESS: 3,
        SyncAction.COMPLETE: 4,
    }
    return sorted(states, key=lambda state: (priority[state.action], state.stem))


__all__ = ["RegionSyncState", "SyncAction", "plan_sync_states"]
