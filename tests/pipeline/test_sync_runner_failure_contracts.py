"""Failure and shutdown contracts for the pure sync runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.pipeline.sync_runner import (
    RegionSyncState,
    SyncAction,
    run_sync,
)


def _state(tmp_path: Path, stem: str, action: SyncAction) -> RegionSyncState:
    path = tmp_path / f"{stem}.osm.pbf"
    path.touch()
    return RegionSyncState(stem, path, action)


def test_upload_failures_return_nonzero_after_successful_processing(tmp_path: Path) -> None:
    completed: list[str] = []
    closed: list[str] = []

    rc = run_sync(
        [_state(tmp_path, "alpha", SyncAction.AUGMENT)],
        extract_pbf=lambda _path: object(),
        process_extracted_pbf=lambda _extracted: object(),
        augment_region=lambda _state: {"ok": True},
        on_complete=lambda state, _result: completed.append(state.stem),
        close_uploads=lambda: closed.append("closed") or ["alpha upload"],
    )

    assert rc == 1
    assert completed == ["alpha"]
    assert closed == ["closed"]


def test_close_uploads_runs_when_augmentation_raises(tmp_path: Path) -> None:
    closed: list[str] = []

    def fail(_state: RegionSyncState) -> Any:
        raise RuntimeError("augmentation failed")

    with pytest.raises(RuntimeError, match="augmentation failed"):
        run_sync(
            [_state(tmp_path, "alpha", SyncAction.AUGMENT)],
            extract_pbf=lambda _path: object(),
            process_extracted_pbf=lambda _extracted: object(),
            augment_region=fail,
            close_uploads=lambda: closed.append("closed") or [],
        )

    assert closed == ["closed"]


def test_process_failure_does_not_augment_or_complete_later_states(tmp_path: Path) -> None:
    events: list[str] = []

    def fail(_extracted: Any) -> Any:
        events.append("process")
        raise RuntimeError("processing failed")

    first = _state(tmp_path, "first", SyncAction.PROCESS)
    second = _state(tmp_path, "second", SyncAction.PROCESS)
    with pytest.raises(RuntimeError, match="processing failed"):
        run_sync(
            [first, second],
            extract_pbf=lambda path: (
                events.append(f"extract:{path.name.removesuffix('.osm.pbf')}") or object()
            ),
            process_extracted_pbf=fail,
            augment_region=lambda state: events.append(f"augment:{state.stem}"),
            on_complete=lambda state, _result: events.append(f"complete:{state.stem}"),
            close_uploads=lambda: [],
        )

    assert events[0] == "extract:first"
    assert "process" in events
    assert "augment:first" not in events
    assert "complete:first" not in events


def test_recovery_state_requires_a_recovery_collaborator(tmp_path: Path) -> None:
    state = _state(tmp_path, "recovery", SyncAction.RECOVERY)

    with pytest.raises(RuntimeError, match="recover_region collaborator"):
        run_sync(
            [state],
            extract_pbf=lambda _path: object(),
            process_extracted_pbf=lambda _extracted: object(),
            augment_region=lambda _state: object(),
        )
