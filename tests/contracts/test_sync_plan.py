"""Freeze the unified sync plan behaviour.

The plan is the input to ``pipeline.sync_runner.run_sync_plan`` and
its ordering invariant (augmentation backlog first, then core
missing, then complete) is a behavioural contract.
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)


def test_plan_orders_augment_before_process() -> None:
    pbfs = [Path("a-latest.osm.pbf"), Path("b-latest.osm.pbf")]
    states = plan_sync_states(
        pbfs,
        core_stems=set(),  # both missing
        augmentation_stems=set(),
        force=False,
    )
    assert [s.action for s in states] == [SyncAction.PROCESS, SyncAction.PROCESS]
    assert [s.stem for s in states] == ["a-latest", "b-latest"]


def test_plan_marks_completed_when_core_and_augmentation_present() -> None:
    pbfs = [Path("a-latest.osm.pbf")]
    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest"},
        augmentation_stems={"a-latest"},
    )
    assert states[0].action is SyncAction.COMPLETE


def test_plan_orders_augment_before_process_for_mixed_set() -> None:
    pbfs = [
        Path("a-latest.osm.pbf"),  # core done, augmentation pending -> AUGMENT
        Path("b-latest.osm.pbf"),  # core missing -> PROCESS
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest"},
        augmentation_stems=set(),
    )
    assert states[0].action is SyncAction.AUGMENT
    assert states[0].stem == "a-latest"
    assert states[1].action is SyncAction.PROCESS
    assert states[1].stem == "b-latest"


def test_plan_force_marks_all_as_process() -> None:
    pbfs = [Path("a-latest.osm.pbf"), Path("b-latest.osm.pbf")]
    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest", "b-latest"},
        augmentation_stems={"a-latest", "b-latest"},
        force=True,
    )
    assert all(s.action is SyncAction.PROCESS for s in states)


def test_plan_orders_complete_last() -> None:
    pbfs = [
        Path("c-latest.osm.pbf"),  # complete
        Path("a-latest.osm.pbf"),  # augment
        Path("b-latest.osm.pbf"),  # process
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest", "c-latest"},
        augmentation_stems={"c-latest"},
    )
    actions = [s.action for s in states]
    # AUGMENT first, then PROCESS, then COMPLETE.
    assert actions.index(SyncAction.AUGMENT) < actions.index(SyncAction.PROCESS)
    assert actions.index(SyncAction.PROCESS) < actions.index(SyncAction.COMPLETE)


def test_run_sync_plan_returns_completed_stems() -> None:
    from osm_polygon_wikidata_only.pipeline.sync_orchestrator import run_sync_plan

    state_a = RegionSyncState("a", Path("a.osm.pbf"), SyncAction.PROCESS)
    state_b = RegionSyncState("b", Path("b.osm.pbf"), SyncAction.AUGMENT)
    state_c = RegionSyncState("c", Path("c.osm.pbf"), SyncAction.COMPLETE)

    process_calls: list[str] = []
    augment_calls: list[str] = []

    def process(region: RegionSyncState) -> None:
        process_calls.append(region.stem)

    def augment(region: RegionSyncState) -> None:
        augment_calls.append(region.stem)

    completed = run_sync_plan(
        [state_a, state_b, state_c],
        process_region=process,
        augment_region=augment,
    )
    # COMPLETE skipped; PROCESS and AUGMENT both contribute to the result.
    assert sorted(completed) == ["a", "b"]
    assert process_calls == ["a"]
    assert augment_calls == ["a", "b"]
