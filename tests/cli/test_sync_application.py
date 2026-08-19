"""Characterization tests for the unified sync deep module.

These tests are intentionally written before ``cli.sync_application`` exists.
They define the small application boundary that ``cli.run_sync.execute`` will
delegate to while pinning ordering, publication, recovery, and cleanup
semantics without touching network or PBF data.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_wikidata_only.cli import run_sync
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction

MODULE_NAME = "osm_polygon_wikidata_only.cli.sync_application"


def _application_module() -> Any:
    """Load the new façade, failing clearly until it is implemented."""
    assert importlib.util.find_spec(MODULE_NAME) is not None, (
        "the deep sync application module must exist before these tests can pass"
    )
    return importlib.import_module(MODULE_NAME)


def _state(stem: str, action: SyncAction, root: Path) -> RegionSyncState:
    return RegionSyncState(stem, root / f"{stem}.osm.pbf", action)


class _Heartbeat:
    def __enter__(self) -> _Heartbeat:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Progress:
    def __init__(self) -> None:
        self.snapshot = lambda: {"phase": "test"}


class _Queue:
    def __init__(self, *, close_result: list[str] | None = None) -> None:
        self.close_result = close_result or []
        self.submissions: list[tuple[list[Any], str]] = []
        self.synchronous: list[tuple[list[Any], str]] = []
        self.closed = False

    def submit(self, ops: list[Any], message: str) -> None:
        self.submissions.append((ops, message))

    def close_and_wait(self) -> list[str]:
        self.closed = True
        return self.close_result

    def upload_synchronously(self, ops: list[Any], message: str) -> None:
        self.synchronous.append((ops, message))


def _context(
    module: Any,
    tmp_path: Path,
    *,
    states: list[RegionSyncState] | None = None,
    push_enabled: bool = False,
    queue: _Queue | None = None,
    reconciliation_plan: Any = None,
) -> Any:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    return module.SyncApplicationContext(
        data_root=root,
        settings=Settings(repo_id="test/repo"),
        runtime=SimpleNamespace(
            wikidata=object(),
            wikipedia=object(),
            scheduler=SimpleNamespace(snapshot=lambda: {"requests": 0}),
            session=SimpleNamespace(auth_snapshot=lambda: {"authenticated": True}),
            cache=None,
        ),
        augmentation_client=object(),
        states=states or [],
        push_enabled=push_enabled,
        dry_run=False,
        pending_stems=set(),
        stems_with_gaps=set(),
        reconciliation_plan=reconciliation_plan,
        upload_queue=queue,
        publish_builder=(lambda *_args, **_kwargs: ["region-op"]) if push_enabled else None,
    )


def _services(
    module: Any,
    events: list[str],
    *,
    runner: Any | None = None,
    audit: Any | None = None,
    repair: Any | None = None,
    marker: Any = None,
    document_path: Path | None = Path("/tmp/sync-application-test-documents.parquet"),
    link_plan: Any | None = None,
) -> Any:
    if runner is None:

        def runner(_states: list[Any], **_callbacks: Any) -> int:
            return 0

    if audit is None:
        audit = SimpleNamespace(
            region=lambda _stem: SimpleNamespace(requires_repair=False),
        )
    if repair is None:
        repair = SimpleNamespace(changed=False, map_inputs_changed=False)

    def extract(path: Path) -> object:
        events.append(f"extract:{path.stem}")
        return object()

    def process(_extracted: object) -> object:
        events.append("process")
        return object()

    def augment(
        _data_root: DataRoot,
        stem: str,
        _client: object,
        *,
        progress: object,
    ) -> object:
        del progress
        events.append(f"augment:{stem}")
        return SimpleNamespace(wikipedia_documents_path=document_path, counts={})

    def load_existing(_data_root: DataRoot, stem: str) -> object:
        events.append(f"load:{stem}")
        return SimpleNamespace(counts={})

    def recover(state: RegionSyncState) -> object | None:
        events.append(f"recover:{state.stem}")
        return None

    def plan_links(*_args: Any, **_kwargs: Any) -> Any:
        return link_plan if link_plan is not None else SimpleNamespace(stems=[])

    def audit_fn(*_args: Any, **_kwargs: Any) -> Any:
        return audit

    def repair_fn(*_args: Any, **_kwargs: Any) -> Any:
        return repair

    def assemble_region(*_args: Any, **_kwargs: Any) -> list[Any]:
        return ["region-op"]

    def assemble_metadata(*_args: Any, **_kwargs: Any) -> list[Any]:
        return ["metadata-op"]

    def log_summary(*_args: Any, **_kwargs: Any) -> None:
        events.append("summary")

    def load_marker(*_args: Any, **_kwargs: Any) -> Any:
        return marker

    def clear_marker(*_args: Any, **_kwargs: Any) -> None:
        events.append("clear-marker")

    return module.SyncApplicationServices(
        extract_pbf=extract,
        process_extracted_pbf=process,
        augment_region=augment,
        load_existing_augmentation=load_existing,
        recover_region=recover,
        run_sync=runner,
        plan_link_migration=plan_links,
        apply_link_migration=lambda *_args, **_kwargs: None,
        audit_wikidata_integrity=audit_fn,
        ensure_recovery_audit_unblocked=lambda _audit: None,
        repair_wikidata_region=repair_fn,
        prepare_local_retirement=lambda *_args, **_kwargs: None,
        add_pending_publications=lambda *_args, **_kwargs: None,
        record_region_recovery_receipt=lambda *_args, **_kwargs: None,
        assemble_region_upload=assemble_region,
        assemble_metadata_only_upload=assemble_metadata,
        load_existing_core_for_publication=lambda *_args, **_kwargs: object(),
        commit_message=lambda state: f"commit:{state.stem}",
        log_remote_reconciliation_summary=log_summary,
        load_metadata_refresh_marker=load_marker,
        clear_metadata_refresh_marker=clear_marker,
        augmentation_progress=_Progress,
        sync_heartbeat=lambda **_kwargs: _Heartbeat(),
        logger=logging.getLogger("sync-application-tests"),
    )


def test_execute_delegates_to_the_deep_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI composition root must delegate post-plan work to one façade."""
    source = Path(run_sync.__file__).read_text(encoding="utf-8")
    assert "SyncApplication(" in source
    assert "SyncApplicationContext(" in source
    assert "SyncApplicationServices(" in source


def test_application_passes_the_complete_callback_contract_to_runner(tmp_path: Path) -> None:
    module = _application_module()
    captured: dict[str, Any] = {}
    events: list[str] = []

    def runner(states: list[Any], **callbacks: Any) -> int:
        captured["states"] = states
        captured.update(callbacks)
        return 0

    context = _context(module, tmp_path)
    application = module.SyncApplication(
        context=context,
        services=_services(module, events, runner=runner),
    )

    result = application.run()

    assert result.return_code == 0
    assert captured["states"] is context.states
    assert all(
        callable(captured[name])
        for name in (
            "extract_pbf",
            "process_extracted_pbf",
            "augment_region",
            "load_existing_augmentation",
            "recover_region",
        )
    )


def test_push_disabled_execution_never_exposes_publication_callbacks(tmp_path: Path) -> None:
    module = _application_module()
    captured: dict[str, Any] = {}

    def runner(_states: list[Any], **callbacks: Any) -> int:
        captured.update(callbacks)
        return 0

    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=False),
        services=_services(module, [], runner=runner),
    )

    assert application.run().return_code == 0
    assert captured["build_upload_files"] is None
    assert captured["submit_upload"] is None
    assert captured["close_uploads"] is None


def test_push_enabled_runner_submits_one_atomic_region_commit(tmp_path: Path) -> None:
    module = _application_module()
    queue = _Queue()
    events: list[str] = []

    def runner(states: list[Any], **callbacks: Any) -> int:
        state = states[0]
        augmentation = callbacks["augment_region"](state)
        callbacks["on_complete"](state, augmentation)
        callbacks["build_upload_files"](state, augmentation, None)
        callbacks["submit_upload"](["region-op"], callbacks["commit_message"](state))
        return 0

    context = _context(
        module,
        tmp_path,
        states=[_state("alpha", SyncAction.AUGMENT, tmp_path)],
        push_enabled=True,
        queue=queue,
    )
    result = module.SyncApplication(
        context=context,
        services=_services(module, events, runner=runner),
    ).run()

    assert result.return_code == 0
    assert queue.submissions == [(["region-op"], "commit:alpha")]
    assert events == ["augment:alpha", "load:alpha", "summary"]


def test_recovery_result_marks_core_and_map_repair_flags(tmp_path: Path) -> None:
    module = _application_module()
    events: list[str] = []
    recovered = SimpleNamespace(requires_repair=True)
    audit = SimpleNamespace(region=lambda _stem: recovered)
    repair = SimpleNamespace(changed=True, map_inputs_changed=True)
    captured: dict[str, Any] = {}

    def runner(_states: list[Any], **callbacks: Any) -> int:
        captured["result"] = callbacks["recover_region"](
            _state("alpha", SyncAction.RECOVERY, tmp_path)
        )
        return 0

    application = module.SyncApplication(
        context=_context(
            module,
            tmp_path,
            states=[_state("alpha", SyncAction.RECOVERY, tmp_path)],
            push_enabled=True,
            queue=_Queue(),
        ),
        services=_services(module, events, runner=runner, audit=audit, repair=repair),
    )

    result = application.run()

    assert captured["result"] is not None
    assert result.core_will_be_repaired is True
    assert result.core_repaired is True


def test_upload_queue_failure_forces_nonzero_result(tmp_path: Path) -> None:
    module = _application_module()
    queue = _Queue(close_result=["upload-job"])
    application = module.SyncApplication(
        context=_context(module, tmp_path, queue=queue),
        services=_services(module, []),
    )

    result = application.run()

    assert result.return_code == 1
    assert queue.closed is True


def test_runner_exception_propagates_after_queue_closes(tmp_path: Path) -> None:
    module = _application_module()
    queue = _Queue()

    def runner(_states: list[Any], **_callbacks: Any) -> int:
        raise RuntimeError("runner failed")

    application = module.SyncApplication(
        context=_context(module, tmp_path, queue=queue),
        services=_services(module, [], runner=runner),
    )

    with pytest.raises(RuntimeError, match="runner failed"):
        application.run()
    assert queue.closed is True


def test_runner_exception_without_queue_still_propagates(tmp_path: Path) -> None:
    module = _application_module()

    def runner(_states: list[Any], **_callbacks: Any) -> int:
        raise RuntimeError("runner failed without queue")

    application = module.SyncApplication(
        context=_context(module, tmp_path, queue=None),
        services=_services(module, [], runner=runner),
    )

    with pytest.raises(RuntimeError, match="runner failed without queue"):
        application.run()


def test_metadata_marker_refreshes_only_after_successful_run(tmp_path: Path) -> None:
    module = _application_module()
    queue = _Queue()
    events: list[str] = []
    plan = SimpleNamespace(repository_refresh=True)
    application = module.SyncApplication(
        context=_context(
            module,
            tmp_path,
            push_enabled=True,
            queue=queue,
            reconciliation_plan=plan,
        ),
        services=_services(
            module,
            events,
            marker={"stems": ["alpha"]},
        ),
    )

    result = application.run()

    assert result.return_code == 0
    assert result.metadata_repaired is True
    assert queue.synchronous == [(["metadata-op"], "Repair remote repository metadata and maps")]
    assert "clear-marker" in events


def test_result_keeps_initial_reconciliation_flags(tmp_path: Path) -> None:
    module = _application_module()
    context = _context(module, tmp_path, push_enabled=False)
    context.core_will_be_repaired = True
    context.core_repaired = True

    result = module.SyncApplication(
        context=context,
        services=_services(module, []),
    ).run()

    assert result.core_will_be_repaired is True
    assert result.core_repaired is True
    assert result.metadata_repaired is False


def test_failed_uploads_do_not_report_success(tmp_path: Path) -> None:
    module = _application_module()
    services = replace(
        _services(module, []),
        log_remote_reconciliation_summary=lambda **_kwargs: pytest.fail(
            "failed uploads must not report successful reconciliation"
        ),
    )
    application = module.SyncApplication(
        context=_context(
            module,
            tmp_path,
            push_enabled=True,
            queue=_Queue(close_result=["failed-upload"]),
        ),
        services=services,
    )

    result = application.run()

    assert result.return_code == 1


def test_finish_promotes_zero_code_with_close_failures(tmp_path: Path) -> None:
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, []),
    )

    result = application._finish(0, ["failed-upload"], False)

    assert result.return_code == 1


def test_nonzero_runner_result_is_preserved(tmp_path: Path) -> None:
    module = _application_module()

    def runner(_states: list[Any], **_callbacks: Any) -> int:
        return 3

    result = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, [], runner=runner),
    ).run()

    assert result.return_code == 3


def test_metadata_marker_requires_an_upload_queue(tmp_path: Path) -> None:
    module = _application_module()
    context = _context(
        module,
        tmp_path,
        push_enabled=True,
        reconciliation_plan=SimpleNamespace(repository_refresh=False),
        queue=None,
    )

    with pytest.raises(RuntimeError, match="Metadata refresh requires an upload queue"):
        module.SyncApplication(
            context=context,
            services=_services(module, [], marker={"stems": ["alpha"]}),
        ).run()


def test_metadata_repair_without_submit_callback_does_not_report_repair(
    tmp_path: Path,
) -> None:
    module = _application_module()
    context = _context(
        module,
        tmp_path,
        push_enabled=True,
        reconciliation_plan=SimpleNamespace(repository_refresh=True),
    )
    application = module.SyncApplication(
        context=context,
        services=_services(module, []),
    )

    assert application._enqueue_metadata_repair(0, None) is False


def test_augment_handles_missing_documents_and_empty_actionable_plan(tmp_path: Path) -> None:
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path, states=[]),
        services=_services(module, [], document_path=None),
    )

    result = application._augment(_state("alpha", SyncAction.AUGMENT, tmp_path))

    assert result.counts == {}


def test_augment_migrates_documents_and_keeps_unchanged_repair_result(
    tmp_path: Path,
) -> None:
    module = _application_module()
    link_plan = SimpleNamespace(
        stems=[SimpleNamespace(classification=SimpleNamespace(value="migratable"))]
    )
    audit = SimpleNamespace(
        region=lambda _stem: SimpleNamespace(requires_repair=True),
    )
    repair = SimpleNamespace(changed=False, map_inputs_changed=False)
    application = module.SyncApplication(
        context=_context(
            module,
            tmp_path,
            states=[_state("alpha", SyncAction.AUGMENT, tmp_path)],
        ),
        services=_services(
            module,
            [],
            link_plan=link_plan,
            audit=audit,
            repair=repair,
        ),
    )

    result = application._augment(_state("alpha", SyncAction.AUGMENT, tmp_path))

    assert result.counts == {}
    assert application._recovered_stems == set()


def test_augment_repair_marks_recovered_region_without_map_refresh(tmp_path: Path) -> None:
    module = _application_module()
    audit = SimpleNamespace(
        region=lambda _stem: SimpleNamespace(requires_repair=True),
    )
    repair = SimpleNamespace(changed=True, map_inputs_changed=False)
    application = module.SyncApplication(
        context=_context(
            module,
            tmp_path,
            states=[_state("alpha", SyncAction.AUGMENT, tmp_path)],
        ),
        services=_services(module, [], audit=audit, repair=repair),
    )

    application._augment(_state("alpha", SyncAction.AUGMENT, tmp_path))

    assert application._recovered_stems == {"alpha"}
    assert application._recovery_map_refresh_stems == set()


@pytest.mark.parametrize(
    ("classification", "stems", "expected"),
    [
        ("canonical", [], False),
        ("canonical", [SimpleNamespace(classification=SimpleNamespace(value="canonical"))], False),
        ("migratable", [SimpleNamespace(classification=SimpleNamespace(value="migratable"))], True),
    ],
)
def test_link_migration_gate_handles_empty_canonical_and_migratable_plans(
    tmp_path: Path,
    classification: str,
    stems: list[Any],
    expected: bool,
) -> None:
    del classification
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, [], link_plan=SimpleNamespace(stems=stems)),
    )

    assert application._migrate_links_if_needed("alpha") is expected


def test_blocked_link_migration_is_rejected(tmp_path: Path) -> None:
    module = _application_module()
    link_plan = SimpleNamespace(
        stems=[
            SimpleNamespace(
                classification=SimpleNamespace(value="BLOCKED"),
                reason="unsafe",
            )
        ]
    )
    application = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, [], link_plan=link_plan),
    )

    with pytest.raises(RuntimeError, match="blocked for alpha"):
        application._migrate_links_if_needed("alpha")


def test_healthy_recovery_uses_pending_augmentation(tmp_path: Path) -> None:
    module = _application_module()
    context = _context(module, tmp_path, push_enabled=True)
    context.pending_stems.add("alpha")
    application = module.SyncApplication(
        context=context,
        services=_services(module, []),
    )

    result = application._recover(_state("alpha", SyncAction.RECOVERY, tmp_path))

    assert result.counts == {}


def test_healthy_recovery_without_push_returns_none_after_migration(tmp_path: Path) -> None:
    module = _application_module()
    link_plan = SimpleNamespace(
        stems=[SimpleNamespace(classification=SimpleNamespace(value="migratable"))]
    )
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=False),
        services=_services(
            module,
            [],
            link_plan=link_plan,
            audit=SimpleNamespace(
                region=lambda _stem: SimpleNamespace(
                    requires_repair=False,
                    classifications={},
                ),
            ),
        ),
    )

    assert application._recover(_state("alpha", SyncAction.RECOVERY, tmp_path)) is None


def test_healthy_recovery_with_migration_loads_existing_when_pushing(tmp_path: Path) -> None:
    module = _application_module()
    link_plan = SimpleNamespace(
        stems=[SimpleNamespace(classification=SimpleNamespace(value="migratable"))]
    )
    audit = SimpleNamespace(
        region=lambda _stem: SimpleNamespace(
            requires_repair=False, classifications={"Q1": "current"}
        ),
    )
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=True, queue=_Queue()),
        services=_services(module, [], link_plan=link_plan, audit=audit),
    )

    result = application._recover(_state("alpha", SyncAction.RECOVERY, tmp_path))

    assert result.counts == {}
    assert application._recovered_stems == {"alpha"}
    assert application._recovery_classifications == {"alpha": {"Q1": "current"}}


def test_recovery_repair_without_change_falls_back_to_healthy_path(tmp_path: Path) -> None:
    module = _application_module()
    audit = SimpleNamespace(
        region=lambda _stem: SimpleNamespace(requires_repair=True),
    )
    repair = SimpleNamespace(changed=False, map_inputs_changed=False)
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=False),
        services=_services(module, [], audit=audit, repair=repair),
    )

    assert application._recover(_state("alpha", SyncAction.RECOVERY, tmp_path)) is None


def test_prepare_publication_records_recovery_receipt_and_submit_handles_no_queue(
    tmp_path: Path,
) -> None:
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, []),
    )
    application._recovery_classifications["alpha"] = {"Q1": "current"}

    application._prepare_publication(
        _state("alpha", SyncAction.AUGMENT, tmp_path),
        SimpleNamespace(wikipedia_documents_path=Path("documents.parquet")),
    )
    application._submit_upload(["op"], "message")


def test_prepare_publication_without_documents_or_receipt_is_a_noop(tmp_path: Path) -> None:
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path),
        services=_services(module, []),
    )

    application._prepare_publication(
        _state("alpha", SyncAction.AUGMENT, tmp_path),
        SimpleNamespace(wikipedia_documents_path=None),
    )
    application._prepare_publication(
        _state("beta", SyncAction.AUGMENT, tmp_path),
        SimpleNamespace(wikipedia_documents_path=Path("documents.parquet")),
    )


def test_default_region_publication_loads_missing_core_and_refreshes_maps(
    tmp_path: Path,
) -> None:
    module = _application_module()
    plan = SimpleNamespace(missing={("alpha", "polygons")})
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=True, reconciliation_plan=plan),
        services=_services(module, []),
    )
    state = _state("alpha", SyncAction.PROCESS, tmp_path)

    ops = application._build_region_publication(state, object(), None)

    assert ops == ["region-op"]
    assert application._refresh_maps(state, "alpha") is True
    recovery_state = _state("alpha", SyncAction.RECOVERY, tmp_path)
    assert application._refresh_maps(recovery_state, "alpha") is False
    application._recovery_map_refresh_stems.add("alpha")
    assert application._refresh_maps(recovery_state, "alpha") is True


def test_recovered_region_without_core_is_rejected(tmp_path: Path) -> None:
    module = _application_module()
    services = _services(module, [])
    services = replace(
        services,
        load_existing_core_for_publication=lambda *_args, **_kwargs: None,
    )
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=True),
        services=services,
    )
    application._recovered_stems.add("alpha")

    with pytest.raises(RuntimeError, match="has no core publication artifacts"):
        application._build_region_publication(
            _state("alpha", SyncAction.RECOVERY, tmp_path),
            object(),
            None,
        )


def test_publication_keeps_supplied_core_without_reconciliation_plan(tmp_path: Path) -> None:
    module = _application_module()
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=False),
        services=_services(module, []),
    )

    assert application._build_region_publication(
        _state("alpha", SyncAction.PROCESS, tmp_path),
        object(),
        object(),
    ) == ["region-op"]


def test_publication_core_loader_logs_and_rethrows_failures(tmp_path: Path) -> None:
    module = _application_module()
    services = replace(
        _services(module, []),
        load_existing_core_for_publication=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("core missing")
        ),
    )
    plan = SimpleNamespace(missing={("alpha", "polygons")})
    application = module.SyncApplication(
        context=_context(module, tmp_path, push_enabled=True, reconciliation_plan=plan),
        services=services,
    )

    with pytest.raises(OSError, match="core missing"):
        application._build_region_publication(
            _state("alpha", SyncAction.PROCESS, tmp_path),
            object(),
            None,
        )
