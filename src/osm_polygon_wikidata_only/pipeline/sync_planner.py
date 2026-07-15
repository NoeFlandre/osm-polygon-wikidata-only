"""Pure planning of work required to converge regional dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SyncAction(StrEnum):
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
) -> list[RegionSyncState]:
    """Classify PBFs and order missing augmentation before new core work."""
    pending = pending_stems or set()
    states: list[RegionSyncState] = []
    for pbf in pbfs:
        stem = pbf.name.removesuffix(".osm.pbf")
        if force or stem not in core_stems:
            action = SyncAction.PROCESS
        elif stem not in augmentation_stems:
            action = SyncAction.AUGMENT
        elif stem in pending:
            action = SyncAction.PUBLISH
        else:
            action = SyncAction.COMPLETE
        states.append(RegionSyncState(stem, pbf, action))
    priority = {
        SyncAction.AUGMENT: 0,
        SyncAction.PROCESS: 1,
        SyncAction.PUBLISH: 2,
        SyncAction.COMPLETE: 3,
    }
    return sorted(states, key=lambda state: (priority[state.action], state.stem))


__all__ = ["RegionSyncState", "SyncAction", "plan_sync_states"]
