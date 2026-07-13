"""State-driven sequencing for the unified dataset pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .sync_planner import RegionSyncState, SyncAction


def run_sync_plan(
    states: Iterable[RegionSyncState],
    *,
    process_region: Callable[[RegionSyncState], object],
    augment_region: Callable[[RegionSyncState], object],
) -> list[str]:
    """Execute missing stages in plan order and return completed stems."""
    completed: list[str] = []
    for state in states:
        if state.action is SyncAction.COMPLETE:
            continue
        if state.action is SyncAction.PROCESS:
            process_region(state)
        augment_region(state)
        completed.append(state.stem)
    return completed


__all__ = ["run_sync_plan"]
