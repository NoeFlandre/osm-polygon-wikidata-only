"""Integration tests for exhaustive Wikidata integrity recovery wiring.

Recovery must run BEFORE any PBF extraction, augmentation, or
publication work in the unified sync plan. Healthy finalized
regions are not touched; only regions whose QID audit returns
``REPAIR_REQUIRED`` enter the recovery path. Recovery must never
invoke extraction or Wikimedia collaborators for repaired regions.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.cli.run_sync import (
    _ensure_recovery_audit_unblocked,
    _recovery_audit_stems,
)
from osm_polygon_wikidata_only.pipeline.sync_runner import (
    RegionSyncState,
    SyncAction,
    run_sync,
)
from osm_polygon_wikidata_only.pipeline.wikidata_recovery import (
    RecoveryAuditResult,
    RegionAuditResult,
)


def _state(stem: str, action: SyncAction, root: Path) -> RegionSyncState:
    return RegionSyncState(stem, root / f"{stem}.osm.pbf", action)


def test_recovery_runs_before_augment_publish_process(tmp_path: Path) -> None:
    events: list[str] = []
    recovery_finished = threading.Event()

    def fake_extract(pbf_path: Path) -> Any:
        assert recovery_finished.is_set(), "PBF extraction started before recovery finished"
        events.append(f"extract:{pbf_path.name.removesuffix('.osm.pbf')}")
        return object()

    def fake_process(_extracted: Any) -> Any:
        events.append("process:never-relevant")
        return object()

    def fake_augment(state: RegionSyncState) -> Any:
        assert recovery_finished.is_set()
        events.append(f"augment:{state.stem}")
        return object()

    def fake_recover(state: RegionSyncState) -> Any:
        events.append(f"recover:{state.stem}")
        recovery_finished.set()
        return object()

    recover_state = _state("recover-region", SyncAction.RECOVERY, tmp_path)
    augment_state = _state("augment-backlog", SyncAction.AUGMENT, tmp_path)
    publish_state = _state("publish-only", SyncAction.PUBLISH, tmp_path)
    process_state = _state("process-core", SyncAction.PROCESS, tmp_path)
    (tmp_path / "process-core.osm.pbf").write_bytes(b"")

    def fake_load_existing(state: RegionSyncState) -> Any:
        events.append(f"load-existing:{state.stem}")
        return object()

    rc = run_sync(
        [process_state, publish_state, augment_state, recover_state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        load_existing_augmentation=fake_load_existing,
        recover_region=fake_recover,
    )
    assert rc == 0
    # Recovery must precede every other non-PUBLISH-load action.
    recover_index = events.index("recover:recover-region")
    assert recover_index < events.index("augment:augment-backlog")
    process_index = next(i for i, e in enumerate(events) if e.startswith("process:"))
    assert recover_index < process_index


def test_augment_completion_callback_runs_once(tmp_path: Path) -> None:
    state = _state("augment", SyncAction.AUGMENT, tmp_path)
    completions: list[str] = []

    rc = run_sync(
        [state],
        extract_pbf=lambda _path: object(),
        process_extracted_pbf=lambda _extracted: object(),
        augment_region=lambda _state: object(),
        on_complete=lambda completed, _result: completions.append(completed.stem),
    )

    assert rc == 0
    assert completions == ["augment"]


def test_startup_audit_only_scopes_finalized_current_regions() -> None:
    assert _recovery_audit_stems(
        input_stems={"healthy", "backlog", "missing-core", "outside"},
        core_stems={"healthy", "backlog", "outside"},
        current_augmentation={"healthy", "outside"},
        force=False,
    ) == ["healthy", "outside"]
    assert (
        _recovery_audit_stems(
            input_stems={"healthy"},
            core_stems={"healthy"},
            current_augmentation={"healthy"},
            force=True,
        )
        == []
    )


def test_blocked_recovery_audit_fails_closed() -> None:
    blocked = RegionAuditResult(
        stem="broken",
        fingerprints=(),
        classifications=(),
        polygon_ids_by_qid=(),
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
        blocked_reason="missing canonical document table",
    )
    audit = RecoveryAuditResult(
        regions=(blocked,),
        qids=(),
        upstream_validation_count=0,
        authoritative_cache_hits=0,
    )

    try:
        _ensure_recovery_audit_unblocked(audit)
    except RuntimeError as error:
        assert "broken" in str(error)
        assert "missing canonical document table" in str(error)
    else:
        raise AssertionError("blocked audit must abort sync-dir")


def test_recovery_does_not_call_extract(tmp_path: Path) -> None:
    extract_calls: list[Path] = []

    def fake_extract(pbf_path: Path) -> Any:
        extract_calls.append(pbf_path)
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(_state: RegionSyncState) -> Any:
        return object()

    def fake_recover(_state: RegionSyncState) -> Any:
        return object()

    states = [_state("recover-region", SyncAction.RECOVERY, tmp_path)]
    rc = run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        recover_region=fake_recover,
    )
    assert rc == 0
    assert extract_calls == []


def test_recovery_publishes_before_process_upload(tmp_path: Path) -> None:
    submit_order: list[str] = []
    recovery_submitted = threading.Event()

    def fake_extract(pbf_path: Path) -> Any:
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(_state: RegionSyncState) -> Any:
        return object()

    def fake_recover(_state: RegionSyncState) -> Any:
        return object()

    def fake_build(state: RegionSyncState, augmentation: Any, core: Any) -> list[Any]:
        return [f"op-for-{state.stem}"]

    def fake_submit(ops: list[Any], message: str) -> None:
        submit_order.append(message)
        if "recover-region" in message:
            recovery_submitted.set()
        if "process-core" in message:
            assert recovery_submitted.is_set()

    def fake_commit(state: RegionSyncState) -> str:
        return f"Sync complete region {state.stem}"

    recover_state = _state("recover-region", SyncAction.RECOVERY, tmp_path)
    process_state = _state("process-core", SyncAction.PROCESS, tmp_path)
    (tmp_path / "process-core.osm.pbf").write_bytes(b"")

    rc = run_sync(
        [process_state, recover_state],
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        recover_region=fake_recover,
        build_upload_files=fake_build,
        commit_message=fake_commit,
        submit_upload=fake_submit,
    )
    assert rc == 0
    assert submit_order == [
        "Sync complete region recover-region",
        "Sync complete region process-core",
    ]


def test_recovery_reuses_existing_publication_when_no_qid_changes(tmp_path: Path) -> None:
    """Healthy finalized regions never enter the recovery path."""

    def fake_extract(pbf_path: Path) -> Any:
        return object()

    def fake_process(_extracted: Any) -> Any:
        return object()

    def fake_augment(_state: RegionSyncState) -> Any:
        return object()

    def fake_load_existing(_state: RegionSyncState) -> Any:
        return object()

    recover_calls: list[str] = []

    def fake_recover(state: RegionSyncState) -> Any:
        recover_calls.append(state.stem)
        return object()

    states = [_state("healthy", SyncAction.PUBLISH, tmp_path)]
    rc = run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        recover_region=fake_recover,
        load_existing_augmentation=fake_load_existing,
    )
    assert rc == 0
    assert recover_calls == []
