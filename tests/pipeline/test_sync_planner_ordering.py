"""Targeted reordering tests for the sync planner.

The planner must produce a deterministic priority that runs the
existing AUGMENT backlog first, then PUBLISH-only repairs, then new
core PROCESSING, then COMPLETE/NOOP stems last. PUBLISH moves ahead
of PROCESS so safe, Wikimedia-free repairs are not blocked behind
expensive PBF extraction.
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.pipeline.sync_planner import (
    SyncAction,
    plan_sync_states,
)


def test_plan_orders_publish_after_augment_and_before_process() -> None:
    pbfs = [
        Path("augment-latest.osm.pbf"),  # AUGMENT backlog
        Path("publish-latest.osm.pbf"),  # PUBLISH
        Path("process-latest.osm.pbf"),  # PROCESS (core missing)
        Path("complete-latest.osm.pbf"),  # COMPLETE
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"augment-latest", "publish-latest", "complete-latest"},
        augmentation_stems={"publish-latest", "complete-latest"},
        pending_stems={"publish-latest"},
    )
    actions = [s.action for s in states]
    # AUGMENT first, then PUBLISH, then PROCESS, then COMPLETE.
    assert actions.index(SyncAction.AUGMENT) < actions.index(SyncAction.PUBLISH)
    assert actions.index(SyncAction.PUBLISH) < actions.index(SyncAction.PROCESS)
    assert actions.index(SyncAction.PROCESS) < actions.index(SyncAction.COMPLETE)


def test_plan_publish_does_not_wait_for_process() -> None:
    """PUBLISH must run after AUGMENT and before PROCESS, even when
    the input ordering is reversed."""
    pbfs = [
        Path("process-latest.osm.pbf"),  # PROCESS (core missing)
        Path("publish-latest.osm.pbf"),  # PUBLISH
        Path("augment-latest.osm.pbf"),  # AUGMENT (core done, aug stale)
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"publish-latest", "augment-latest"},
        # augment-latest is NOT in augmentation_stems -> AUGMENT backlog.
        augmentation_stems={"publish-latest"},
        pending_stems={"publish-latest"},
    )
    actions = [s.action for s in states]
    assert actions[0] is SyncAction.AUGMENT
    assert actions[1] is SyncAction.PUBLISH
    assert actions[2] is SyncAction.PROCESS


def test_plan_publish_keeps_alphabetical_order_within_action() -> None:
    pbfs = [
        Path("z-latest.osm.pbf"),
        Path("a-latest.osm.pbf"),
        Path("m-latest.osm.pbf"),
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"a-latest", "m-latest", "z-latest"},
        # All three are in core AND in augmentation (so they
        # could be PUBLISH); only those in pending_stems are
        # actually selected as PUBLISH.
        augmentation_stems={"a-latest", "m-latest", "z-latest"},
        pending_stems={"a-latest", "m-latest", "z-latest"},
    )
    publish_stems = [s.stem for s in states if s.action is SyncAction.PUBLISH]
    assert publish_stems == ["a-latest", "m-latest", "z-latest"]


def test_plan_augment_backlog_runs_before_publish() -> None:
    """If a stem is BOTH a publish candidate AND an augment candidate,
    the planner must treat it as AUGMENT (the backlog takes priority
    because the augmentation needs to run first)."""
    pbfs = [Path("dual-latest.osm.pbf")]
    states = plan_sync_states(
        pbfs,
        core_stems={"dual-latest"},
        # augmentation_stems does NOT include this stem -> AUGMENT backlog
        augmentation_stems=set(),
        # pending_stems includes it -> would otherwise be PUBLISH
        pending_stems={"dual-latest"},
    )
    assert states[0].action is SyncAction.AUGMENT


def test_plan_force_process_overrides_publish() -> None:
    """Force must dominate the priority order."""
    pbfs = [
        Path("p-latest.osm.pbf"),
        Path("a-latest.osm.pbf"),
    ]
    states = plan_sync_states(
        pbfs,
        core_stems={"p-latest", "a-latest"},
        augmentation_stems={"p-latest", "a-latest"},
        pending_stems={"p-latest"},
        force=True,
    )
    assert all(s.action is SyncAction.PROCESS for s in states)
