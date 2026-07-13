"""Tests for unified core/augmentation sequencing."""

from pathlib import Path

from osm_polygon_wikidata_only.pipeline.sync_orchestrator import run_sync_plan
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction


def test_sync_plan_drains_backlog_and_augments_new_core_immediately() -> None:
    states = [
        RegionSyncState("backlog", Path("backlog.osm.pbf"), SyncAction.AUGMENT),
        RegionSyncState("new", Path("new.osm.pbf"), SyncAction.PROCESS),
        RegionSyncState("done", Path("done.osm.pbf"), SyncAction.COMPLETE),
    ]
    events: list[str] = []

    completed = run_sync_plan(
        states,
        process_region=lambda state: events.append(f"process:{state.stem}"),
        augment_region=lambda state: events.append(f"augment:{state.stem}"),
    )

    assert events == ["augment:backlog", "process:new", "augment:new"]
    assert completed == ["backlog", "new"]
