"""Unit contract for the runner's deterministic action partition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction
from osm_polygon_wikidata_only.pipeline.sync_runner import (
    _partition_states,
    _validate_required_collaborators,
)


def test_partition_states_routes_actions_without_reordering(tmp_path: Path) -> None:
    """Each action bucket keeps input order while excluding other actions."""
    states = [
        RegionSyncState("process-a", tmp_path / "a.osm.pbf", SyncAction.PROCESS),
        RegionSyncState("recovery-a", tmp_path / "b.osm.pbf", SyncAction.RECOVERY),
        RegionSyncState("augment-a", tmp_path / "c.osm.pbf", SyncAction.AUGMENT),
        RegionSyncState("publish-a", tmp_path / "d.osm.pbf", SyncAction.PUBLISH),
        RegionSyncState("process-b", tmp_path / "e.osm.pbf", SyncAction.PROCESS),
        RegionSyncState("complete", tmp_path / "f.osm.pbf", SyncAction.COMPLETE),
    ]

    process, augment, publish, recovery = _partition_states(states)

    assert [state.stem for state in process] == ["process-a", "process-b"]
    assert [state.stem for state in augment] == ["augment-a"]
    assert [state.stem for state in publish] == ["publish-a"]
    assert [state.stem for state in recovery] == ["recovery-a"]


def test_validate_required_collaborators_keeps_the_public_error_contract() -> None:
    """Missing core callbacks fail with the runner's stable message."""
    with pytest.raises(RuntimeError, match="requires extract_pbf"):
        _validate_required_collaborators(None, lambda _: Any, lambda _: Any)
