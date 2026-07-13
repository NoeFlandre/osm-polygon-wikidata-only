"""Tests for deterministic unified-pipeline state planning."""

from pathlib import Path

from osm_polygon_wikidata_only.pipeline.sync_planner import SyncAction, plan_sync_states


def test_planner_prioritizes_augmentation_backlog_then_unprocessed() -> None:
    pbfs = [Path("z-latest.osm.pbf"), Path("a-latest.osm.pbf"), Path("m-latest.osm.pbf")]

    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest", "m-latest"},
        augmentation_stems={"a-latest"},
    )

    assert [(state.stem, state.action) for state in states] == [
        ("m-latest", SyncAction.AUGMENT),
        ("z-latest", SyncAction.PROCESS),
        ("a-latest", SyncAction.COMPLETE),
    ]


def test_planner_treats_stale_augmentation_as_backlog() -> None:
    states = plan_sync_states(
        [Path("a-latest.osm.pbf")],
        core_stems={"a-latest"},
        augmentation_stems=set(),
    )
    assert states[0].action is SyncAction.AUGMENT


def test_force_reprocesses_every_raw_pbf() -> None:
    states = plan_sync_states(
        [Path("a-latest.osm.pbf")],
        core_stems={"a-latest"},
        augmentation_stems={"a-latest"},
        force=True,
    )
    assert states[0].action is SyncAction.PROCESS
