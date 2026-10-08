"""Focused characterization of the sync composition and runtime boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.augmentation import progress as augmentation_progress
from osm_polygon_wikidata_only.cli import run_sync, sync_runtime
from osm_polygon_wikidata_only.cli.sync_runtime import load_existing_core_for_publication
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf import publication as hf_publication
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline import processor, sync_heartbeat, sync_runner
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction
from osm_polygon_wikidata_only.pipeline.sync_planning import PreparedSyncPlan
from osm_polygon_wikidata_only.pipeline.sync_reconciliation import RemoteReconciliation


def test_execute_wires_the_plan_runtime_and_upload_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(repo_id="org/dataset")
    args = argparse.Namespace(push=True, dry_run=True, upload_threads=7)
    hub = object()
    inventory = RemoteInventory({"remote.parquet"})
    runtime = object()
    augmentation_client = object()
    queue = object()

    def publish_builder(*_args: Any, **_kwargs: Any) -> list[PublicationOp]:
        return []

    prepared = PreparedSyncPlan(
        pbfs=[],
        input_stems={"alpha"},
        all_pending_stems={"pending"},
        states=[RegionSyncState("planned", Path("planned.osm.pbf"), SyncAction.COMPLETE)],
        remote_state=RemoteReconciliation(
            inventory=None,
            plan=None,
            augmentation_current={},
            stems_with_gaps=set(),
            containment_publications={"parent": ("child",)},
            core_repaired=False,
        ),
        core_will_be_repaired=False,
    )
    calls: list[tuple[str, Any]] = []

    def prepare(
        actual_args: argparse.Namespace,
        *,
        data_root: DataRoot,
        settings: Settings,
        push_enabled: bool,
        dry_run: bool,
        remote_inventory: object | None,
        hub: object | None,
    ) -> PreparedSyncPlan:
        calls.append(
            (
                "prepare",
                (actual_args, data_root, settings, push_enabled, dry_run, remote_inventory, hub),
            )
        )
        return prepared

    def run_application(actual_args: argparse.Namespace, **kwargs: Any) -> int:
        calls.append(("application", (actual_args, kwargs)))
        return 17

    monkeypatch.setattr(run_sync.sync_planning, "prepare_sync_plan", prepare)
    monkeypatch.setattr(run_sync, "build_wikimedia_runtime", lambda *_a, **_k: runtime)
    monkeypatch.setattr(sync_runtime, "build_augmentation_client", lambda *_a: augmentation_client)
    monkeypatch.setattr(
        run_sync.sync_publication,
        "build_upload_queue",
        lambda **kwargs: calls.append(("queue", kwargs)) or queue,
    )
    monkeypatch.setattr(
        run_sync.sync_publication,
        "enqueue_containment_retirement",
        lambda *args, **kwargs: calls.append(("retirement", (args, kwargs))) or True,
    )
    monkeypatch.setattr(
        run_sync.sync_planning,
        "log_sync_plan",
        lambda states: calls.append(("log", states)),
    )
    monkeypatch.setattr(sync_runtime, "run_sync_application", run_application)

    result = run_sync.execute(
        args,
        data_root=data_root,
        settings=settings,
        build_upload_files=publish_builder,
        _remote_inventory=inventory,
        _hub=cast(Any, hub),
    )

    assert result == 17
    assert calls == [
        ("prepare", (args, data_root, settings, True, True, inventory, hub)),
        (
            "queue",
            {
                "push": True,
                "dry_run": True,
                "settings": settings,
                "data_root": data_root,
                "num_threads": 7,
                "_hub": hub,
            },
        ),
        (
            "retirement",
            ((data_root, settings, {"parent": ("child",)}, queue), {"push_enabled": True}),
        ),
        ("log", prepared.states),
        (
            "application",
            (
                args,
                {
                    "data_root": data_root,
                    "settings": settings,
                    "runtime": runtime,
                    "augmentation_client": augmentation_client,
                    "prepared": prepared,
                    "push_enabled": True,
                    "dry_run": True,
                    "upload_queue": queue,
                    "containment_enqueued": True,
                    "publish_builder": publish_builder,
                },
            ),
        ),
    ]


def test_execute_defaults_optional_cli_flags_and_thread_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = argparse.Namespace()
    data_root = DataRoot(tmp_path)
    settings = Settings()
    prepared = PreparedSyncPlan(
        pbfs=[],
        input_stems=set(),
        all_pending_stems=set(),
        states=[],
        remote_state=RemoteReconciliation(None, None, {}, set(), {}, False),
        core_will_be_repaired=False,
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(run_sync.sync_planning, "prepare_sync_plan", lambda *_a, **_k: prepared)
    monkeypatch.setattr(run_sync, "build_wikimedia_runtime", lambda *_a, **_k: object())
    monkeypatch.setattr(sync_runtime, "build_augmentation_client", lambda *_a: object())
    monkeypatch.setattr(
        run_sync.sync_publication,
        "build_upload_queue",
        lambda **kwargs: observed.setdefault("queue", kwargs),
    )
    monkeypatch.setattr(
        run_sync.sync_publication, "enqueue_containment_retirement", lambda *_a, **_k: False
    )
    monkeypatch.setattr(run_sync.sync_planning, "log_sync_plan", lambda _states: None)

    def run_application(_args: argparse.Namespace, **kwargs: Any) -> int:
        observed["application"] = kwargs
        return 0

    monkeypatch.setattr(sync_runtime, "run_sync_application", run_application)

    run_sync.execute(args, data_root=data_root, settings=settings)

    assert observed["queue"] == {
        "push": False,
        "dry_run": False,
        "settings": settings,
        "data_root": data_root,
        "num_threads": 2,
        "_hub": None,
    }
    assert observed["application"]["push_enabled"] is False
    assert observed["application"]["dry_run"] is False
    assert observed["application"]["publish_builder"] is None


def test_run_sync_application_builds_complete_context_and_returns_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings()
    args = argparse.Namespace(commit_message="publish")
    runtime = object()
    augmentation_client = object()
    queue = object()
    plan = cast(Any, object())
    states = [RegionSyncState("state", Path("state.osm.pbf"), SyncAction.COMPLETE)]
    prepared = PreparedSyncPlan(
        pbfs=[],
        input_stems={"input"},
        all_pending_stems={"pending"},
        states=states,
        remote_state=RemoteReconciliation(None, plan, {}, {"gap"}, {"parent": ("child",)}, True),
        core_will_be_repaired=True,
    )
    services = object()
    captured: dict[str, Any] = {}

    class Application:
        def __init__(self, *, context: Any, services: Any) -> None:
            captured["context"] = context
            captured["services"] = services

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(return_code=23)

    monkeypatch.setattr(sync_runtime, "SyncApplication", Application)

    def build_services(actual_args: argparse.Namespace, **kwargs: Any) -> object:
        captured["service_args"] = actual_args
        captured["service_kwargs"] = kwargs
        return services

    monkeypatch.setattr(sync_runtime, "build_sync_services", build_services)

    result = sync_runtime.run_sync_application(
        args,
        data_root=data_root,
        settings=settings,
        runtime=cast(Any, runtime),
        augmentation_client=cast(Any, augmentation_client),
        prepared=prepared,
        push_enabled=True,
        dry_run=True,
        upload_queue=cast(Any, queue),
        containment_enqueued=True,
        publish_builder=lambda *_a, **_k: [],
    )

    context = captured["context"]
    assert result == 23
    assert captured["services"] is services
    assert captured["service_args"] is args
    assert captured["service_kwargs"] == {
        "data_root": data_root,
        "settings": settings,
        "runtime": runtime,
    }
    assert context.data_root is data_root
    assert context.settings is settings
    assert context.runtime is runtime
    assert context.augmentation_client is augmentation_client
    assert context.states is states
    assert context.push_enabled is True
    assert context.dry_run is True
    assert context.pending_stems == {"pending"}
    assert context.stems_with_gaps == {"gap"}
    assert context.reconciliation_plan is plan
    assert context.upload_queue is queue
    assert context.publish_builder is not None
    assert context.core_will_be_repaired is True
    assert context.core_repaired is True
    assert context.containment_enqueued is True


def test_build_sync_services_binds_current_runtime_collaborators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(skip_existing=True)
    runtime = SimpleNamespace(wikidata=object(), wikipedia=object(), cache=object())
    args = argparse.Namespace()
    calls: list[tuple[object, dict[str, Any]]] = []
    extracted = object()

    def extract(path: Path, *, settings: Settings) -> object:
        calls.append((path, {"settings": settings}))
        return extracted

    def process(value: object, **kwargs: Any) -> object:
        calls.append((value, kwargs))
        return "processed"

    monkeypatch.setattr(processor, "extract_pbf", extract)
    monkeypatch.setattr(processor, "process_extracted_pbf", process)
    services = sync_runtime.build_sync_services(
        args,
        data_root=data_root,
        settings=settings,
        runtime=cast(Any, runtime),
    )

    path = Path("region.osm.pbf")
    assert services.extract_pbf(path) is extracted
    assert services.process_extracted_pbf(extracted) == "processed"
    assert calls == [
        (path, {"settings": settings}),
        (
            extracted,
            {
                "data_root": data_root,
                "wikidata_client": runtime.wikidata,
                "wikipedia_client": runtime.wikipedia,
                "settings": Settings(skip_existing=False),
                "cache": runtime.cache,
            },
        ),
    ]
    expected_services = {
        "augment_region": sync_runtime.augment_region,
        "load_existing_augmentation": sync_runtime.load_existing_augmentation_result,
        "plan_link_migration": sync_runtime.plan_link_migration,
        "apply_link_migration": sync_runtime.apply_link_migration,
        "audit_wikidata_integrity": sync_runtime.audit_wikidata_integrity,
        "ensure_recovery_audit_unblocked": sync_runtime.ensure_recovery_audit_unblocked,
        "repair_wikidata_region": sync_runtime.repair_wikidata_region,
        "prepare_local_retirement": sync_runtime.prepare_local_retirement,
        "add_pending_publications": sync_runtime.add_pending_publications,
        "record_region_recovery_receipt": sync_runtime.record_region_recovery_receipt,
        "load_existing_core_for_publication": sync_runtime.load_existing_core_for_publication,
        "log_remote_reconciliation_summary": sync_runtime.log_remote_reconciliation_summary,
        "load_metadata_refresh_marker": sync_runtime.load_metadata_refresh_marker,
        "set_metadata_refresh_marker": sync_runtime.set_metadata_refresh_marker,
        "clear_metadata_refresh_marker": sync_runtime.clear_metadata_refresh_marker,
    }
    for name, collaborator in expected_services.items():
        assert getattr(services, name) is collaborator, name
    assert services.run_sync is sync_runner.run_sync
    assert services.assemble_region_upload is hf_publication.assemble_region_upload
    assert services.assemble_metadata_only_upload is hf_publication.assemble_metadata_only_upload
    assert services.augmentation_progress is augmentation_progress.AugmentationProgress
    assert services.sync_heartbeat is sync_heartbeat.SyncHeartbeat
    assert services.logger is sync_runtime.LOGGER
    assert (
        services.commit_message(
            RegionSyncState("region", Path("region.osm.pbf"), SyncAction.COMPLETE)
        )
        == "Sync complete region region"
    )


def test_build_sync_services_honors_commit_message_override(tmp_path: Path) -> None:
    services = sync_runtime.build_sync_services(
        argparse.Namespace(commit_message="custom publication"),
        data_root=DataRoot(tmp_path),
        settings=Settings(),
        runtime=cast(Any, SimpleNamespace(wikidata=object(), wikipedia=object(), cache=None)),
    )

    assert (
        services.commit_message(
            RegionSyncState("region", Path("region.osm.pbf"), SyncAction.COMPLETE)
        )
        == "custom publication"
    )


def test_build_augmentation_client_uses_the_shared_runtime_and_cache_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings()
    scheduler = object()
    session = object()
    cache = object()
    data_root = DataRoot(tmp_path)
    runtime = SimpleNamespace(
        settings=settings,
        scheduler=scheduler,
        session=session,
    )
    constructed: dict[str, Any] = {}

    def build_cache(path: Path, *, contract_version: str) -> object:
        constructed["cache_args"] = (path, contract_version)
        return cache

    def build_client(
        actual_settings: Settings,
        actual_cache: object,
        *,
        scheduler: object,
        session: object,
    ) -> object:
        constructed["client_args"] = (actual_settings, actual_cache, scheduler, session)
        return "client"

    monkeypatch.setattr(sync_runtime, "JsonFileCache", build_cache)
    monkeypatch.setattr(sync_runtime, "AugmentationWikimediaClient", build_client)

    result = sync_runtime.build_augmentation_client(data_root, cast(Any, runtime))

    assert result == "client"
    assert constructed == {
        "cache_args": (data_root.cache / "augmentation", "text-sidecars-v1"),
        "client_args": (settings, cache, scheduler, session),
    }


def test_publication_core_loader_uses_existing_core_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_wikidata_only.hf.publication as publication

    existing = object()
    assert (
        cast(Any, load_existing_core_for_publication)(
            SimpleNamespace(), "region", existing, required=True
        )
        is existing
    )
    assert (
        cast(Any, load_existing_core_for_publication)(
            SimpleNamespace(), "region", None, required=False
        )
        is None
    )
    monkeypatch.setattr(publication, "load_existing_core_artifacts", lambda _root, stem: (stem,))
    assert cast(Any, load_existing_core_for_publication)(
        SimpleNamespace(), "region", None, required=True
    ) == ("region",)
