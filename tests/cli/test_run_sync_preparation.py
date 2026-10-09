"""Tests for the small, deterministic preparation helpers used by sync-dir."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import MigrationOperation
from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    StemPlan as ArticleStemPlan,
)
from osm_polygon_wikidata_only.cli.sync_publication import enqueue_containment_retirement
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, delete_op
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline import (
    containment_migration,
    sync_planning,
    sync_reconciliation,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan as LinkMigrationPlan,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemClassification,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemPlan as LinkStemPlan,
)
from osm_polygon_wikidata_only.pipeline.containment_migration import (
    PreparedRule,
    RuleAudit,
)
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction
from osm_polygon_wikidata_only.pipeline.sync_planning import (
    PreparedSyncPlan,
    _active_pbfs,
    _migration_stems_to_persist,
    _prepare_containment_rules,
    plan_sync_states_with_recovery,
)
from osm_polygon_wikidata_only.pipeline.sync_reconciliation import (
    RemoteReconciliation,
    _containment_publications_for_remote,
    _reconciliation_gap_counts,
    _remote_child_has_artifact,
    core_repair_required,
    prepare_remote_reconciliation,
    reconciliation_summary_message,
)


def test_active_pbfs_excludes_retired_stems_without_reordering() -> None:
    """Retired shards are removed while active PBF order is preserved."""
    pbfs = [Path("b.osm.pbf"), Path("a.osm.pbf"), Path("c.osm.pbf")]

    assert _active_pbfs(pbfs, {"a"}) == [Path("b.osm.pbf"), Path("c.osm.pbf")]


def test_prepare_containment_rules_reports_prepared_and_blocked_rules() -> None:
    """Containment setup logs both successful and blocked rules."""
    info: list[tuple[str, tuple[Any, ...]]] = []
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    calls: list[tuple[Path, bool]] = []

    def prepare(path: Path, *, dry_run: bool) -> tuple[list[PreparedRule], list[RuleAudit]]:
        calls.append((path, dry_run))
        return [PreparedRule("prepared", ())], [RuleAudit("parent", (), ("reason", "second"))]

    _prepare_containment_rules(
        enabled=True,
        data_path=Path("/data"),
        dry_run=True,
        prepare_safe_rules=prepare,
        log_info=lambda message, *args: info.append((message, args)),
        log_warning=lambda message, *args: warnings.append((message, args)),
    )

    assert calls == [(Path("/data"), True)]
    assert info == [("Prepared %d lossless contained-region retirement rule(s)", (1,))]
    assert warnings == [("Containment retirement blocked for %s: %s", ("parent", "reason; second"))]


def test_core_repair_required_matches_action_and_missing_artifacts() -> None:
    """Only core work or missing core files marks a state for map refresh."""
    missing = {("region", "polygons")}

    assert core_repair_required(SyncAction.PROCESS, "other", set()) is True
    assert core_repair_required(SyncAction.PUBLISH, "region", missing) is True
    assert core_repair_required(SyncAction.AUGMENT, "region", missing) is True
    assert core_repair_required(SyncAction.PUBLISH, "other", missing) is False
    assert core_repair_required(SyncAction.COMPLETE, "region", missing) is False


def test_plan_sync_states_with_recovery_adds_noncanonical_link_migrations_to_recovery() -> None:
    """Legacy link layouts are included in the recovery action set."""
    link_plan = LinkMigrationPlan(
        processed_dir=Path("processed"),
        stems=(
            LinkStemPlan(
                stem="region",
                classification=StemClassification.MIGRATABLE,
                reason="",
                polygons_fingerprint="",
                links_fingerprint="",
                documents_fingerprint="",
                row_count=0,
                canonical_digest=None,
            ),
        ),
    )

    states = plan_sync_states_with_recovery(
        [Path("region.osm.pbf")],
        inventory=sync_planning.StemInventory(
            input_stems={"region"}, core_stems={"region"}, current_augmentation={"region"}
        ),
        force=False,
        pending_stems=set(),
        recovery_stems=set(),
        processed_path=Path("processed"),
        plan_link_migration=lambda _path, *, stems: link_plan,
    )

    assert [(state.stem, state.action) for state in states] == [("region", SyncAction.RECOVERY)]


def test_reconciliation_summary_message_has_stable_four_case_contract() -> None:
    """Summary wording reflects repair and map-refresh signals only."""
    assert reconciliation_summary_message(2, True, False) == (
        "Remote reconciliation complete: 2 regions repaired; README and maps refreshed"
    )
    assert reconciliation_summary_message(0, False, True) == (
        "Remote reconciliation complete: README and maps refreshed"
    )
    assert reconciliation_summary_message(1, False, False) == (
        "Remote reconciliation complete: 1 regions repaired"
    )
    assert reconciliation_summary_message(0, False, False) == (
        "Remote reconciliation complete: converged"
    )


def test_reconciliation_gap_counts_separate_core_and_text_artifacts() -> None:
    """Missing core and augmentation artifacts are counted independently."""
    missing = {
        ("a", "polygon_articles"),
        ("a", "wikipedia/documents"),
        ("b", "wikidata/facts"),
    }

    assert _reconciliation_gap_counts({"a", "b", "c"}, missing) == (1, 2)


def test_prepare_remote_reconciliation_disabled_returns_empty_state(tmp_path: Path) -> None:
    """Non-push runs avoid all remote inventory and validation work."""
    result = prepare_remote_reconciliation(
        enabled=False,
        data_root=DataRoot(tmp_path),
        settings=Settings(),
        input_stems=set(),
        source=sync_reconciliation.RemoteSource(hub=None, inventory_override=None),
        validate_augmentation=lambda *_: pytest.fail("must not validate"),
        load_retired_parent_children=lambda *_: pytest.fail("must not load"),
        helpers=sync_reconciliation.RemoteHelpers(canonical_region_paths=None, planner_cls=None),
    )

    assert result.inventory is None
    assert result.plan is None
    assert result.augmentation_current == {}
    assert result.stems_with_gaps == set()
    assert result.containment_publications == {}
    assert result.core_repaired is False


def test_migration_stems_to_persist_selects_only_creating_operations() -> None:
    """Only operations that create or upgrade canonical documents persist intent."""
    plans = [
        ArticleStemPlan("create", MigrationOperation.CREATE_MISSING, "", "", None, 0, None),
        ArticleStemPlan("upgrade", MigrationOperation.UPGRADE_LEGACY, "", "", None, 0, None),
        ArticleStemPlan("canonical", MigrationOperation.ALREADY_CANONICAL, "", "", None, 0, None),
    ]

    assert _migration_stems_to_persist(plans) == {"create", "upgrade"}


def test_containment_publications_keep_children_present_on_remote() -> None:
    """Only contained children with any canonical remote artifact are published."""
    inventory = RemoteInventory({"child/polygons.parquet"})

    def paths(stem: str) -> dict[str, str]:
        return {"polygons": f"{stem}/polygons.parquet"}

    assert _remote_child_has_artifact("child", inventory, paths) is True
    assert _remote_child_has_artifact("empty", inventory, paths) is False
    assert _containment_publications_for_remote(
        {"parent": ("child", "empty")},
        inventory=inventory,
        canonical_region_paths=paths,
    ) == {"parent": ("child",)}


@pytest.mark.parametrize(
    ("push_enabled", "parent_children", "queue"),
    [
        (False, {"parent": ("child",)}, object()),
        (True, {}, object()),
        (True, {"parent": ("child",)}, None),
    ],
)
def test_enqueue_containment_retirement_skips_ineligible_runs(
    tmp_path: Path,
    push_enabled: bool,
    parent_children: dict[str, tuple[str, ...]],
    queue: object | None,
) -> None:
    assert (
        enqueue_containment_retirement(
            data_root=DataRoot(tmp_path),
            settings=Settings(repo_id="org/repo"),
            parent_children=parent_children,
            upload_queue=cast(Any, queue),
            push_enabled=push_enabled,
        )
        is False
    )


def test_enqueue_containment_retirement_submits_one_remote_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[tuple[object, str]] = []

    def assemble(**kwargs: object) -> list[PublicationOp]:
        assert kwargs["repo_id"] == "org/repo"
        assert kwargs["parent_children"] == {"parent": ("child", "other")}
        return [delete_op("child/polygons.parquet")]

    class Queue:
        def submit(self, operations: list[PublicationOp], description: str) -> None:
            submitted.append((operations, description))

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.cli.sync_publication._assemble_containment_retirement_upload",
        assemble,
    )

    assert (
        enqueue_containment_retirement(
            data_root=DataRoot(tmp_path),
            settings=Settings(repo_id="org/repo"),
            parent_children={"parent": ("child", "other")},
            upload_queue=cast(Any, Queue()),
            push_enabled=True,
        )
        is True
    )
    assert submitted == [
        (
            [delete_op("child/polygons.parquet")],
            "Retire losslessly contained regional dataset shards",
        )
    ]


def test_prepare_remote_reconciliation_validates_inputs_and_builds_remote_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(repo_id="org/dataset", hf_token="token")
    inventory = RemoteInventory({"child/polygons.parquet"})
    hub = object()
    calls: list[tuple[str, Any]] = []
    planner_plan = type(
        "Plan",
        (),
        {
            "stems_to_publish": {"alpha"},
            "stems_to_augment": {"beta"},
            "missing": {("alpha", "polygons"), ("beta", "wikipedia/documents")},
        },
    )()

    def validate(root: DataRoot, stems: list[str]) -> dict[str, bool]:
        calls.append(("validate", (root, stems)))
        return {"alpha": True, "beta": False}

    def retired(root: Path) -> dict[str, tuple[str, ...]]:
        calls.append(("retired", root))
        return {"parent": ("child", "missing")}

    def canonical_paths(stem: str) -> dict[str, str]:
        return {"polygons": f"{stem}/polygons.parquet"}

    class Planner:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("planner", kwargs))

        def plan(self) -> Any:
            return planner_plan

    with caplog.at_level("INFO", logger="osm_polygon_wikidata_only.cli"):
        result = prepare_remote_reconciliation(
            enabled=True,
            data_root=data_root,
            settings=settings,
            input_stems={"beta", "alpha"},
            source=sync_reconciliation.RemoteSource(
                hub=cast(Any, hub), inventory_override=inventory
            ),
            validate_augmentation=validate,
            load_retired_parent_children=retired,
            helpers=sync_reconciliation.RemoteHelpers(
                canonical_region_paths=canonical_paths, planner_cls=cast(Any, Planner)
            ),
        )

    assert calls == [
        ("validate", (data_root, ["alpha", "beta"])),
        ("retired", data_root.processed),
        (
            "planner",
            {
                "data_root": data_root,
                "inventory": inventory,
                "stems": {"alpha", "beta"},
                "augmentation_current": {"alpha": True, "beta": False},
            },
        ),
    ]
    assert result.inventory is inventory
    assert result.plan is planner_plan
    assert result.augmentation_current == {"alpha": True, "beta": False}
    assert result.stems_with_gaps == {"alpha", "beta"}
    assert result.containment_publications == {"parent": ("child",)}
    assert result.core_repaired is True
    assert (
        "Remote reconciliation: 1 regions missing core artifacts, 1 missing augmentation artifacts"
        in caplog.messages
    )


def test_remote_reconciliation_helpers_require_both_push_dependencies() -> None:
    assert sync_reconciliation.remote_reconciliation_helpers(False) == (None, None)
    planner, canonical_paths = sync_reconciliation.remote_reconciliation_helpers(True)
    assert planner is not None
    assert callable(canonical_paths)


def test_require_remote_helpers_rejects_partial_dependency_pairs() -> None:
    def paths(_stem: str) -> dict[str, str]:
        return {}

    planner = cast(Any, type("Planner", (), {}))
    assert sync_reconciliation.require_remote_helpers(paths, planner) == (paths, planner)
    for partial in ((None, planner), (paths, None), (None, None)):
        with pytest.raises(RuntimeError, match="helpers are required"):
            sync_reconciliation.require_remote_helpers(*partial)


def test_require_remote_helpers_keeps_the_operator_error_text_stable() -> None:
    with pytest.raises(RuntimeError) as error:
        sync_reconciliation.require_remote_helpers(None, None)
    assert str(error.value) == "Remote reconciliation helpers are required when push is enabled"


def test_remote_inventory_uses_override_or_forwards_all_fetch_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = RemoteInventory({"existing.parquet"})
    fetch_calls: list[dict[str, Any]] = []

    def fetch(**kwargs: Any) -> RemoteInventory:
        fetch_calls.append(kwargs)
        return override

    monkeypatch.setattr(sync_reconciliation.RemoteInventory, "fetch", fetch)
    assert (
        sync_reconciliation._remote_inventory(
            override,
            repo_id="org/dataset",
            hub=cast(Any, object()),
            token="ignored",
        )
        is override
    )
    assert fetch_calls == []

    hub = object()
    assert (
        sync_reconciliation._remote_inventory(
            None,
            repo_id="org/dataset",
            hub=cast(Any, hub),
            token="token",
        )
        is override
    )
    assert fetch_calls == [{"repo_id": "org/dataset", "hub": hub, "token": "token"}]


def test_prepare_remote_reconciliation_forwards_inventory_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(repo_id="org/dataset", hf_token="token")
    hub = object()
    inventory = RemoteInventory({"alpha/polygons.parquet"})
    fetch_calls: list[dict[str, Any]] = []
    planner_calls: list[dict[str, Any]] = []

    def fetch(**kwargs: Any) -> RemoteInventory:
        fetch_calls.append(kwargs)
        return inventory

    class Planner:
        def __init__(self, **kwargs: Any) -> None:
            planner_calls.append(kwargs)

        def plan(self) -> Any:
            return type(
                "Plan",
                (),
                {"stems_to_publish": set(), "stems_to_augment": set(), "missing": set()},
            )()

    monkeypatch.setattr(RemoteInventory, "fetch", fetch)
    result = sync_reconciliation.prepare_remote_reconciliation(
        enabled=True,
        data_root=data_root,
        settings=settings,
        input_stems={"alpha"},
        source=sync_reconciliation.RemoteSource(hub=cast(Any, hub), inventory_override=None),
        validate_augmentation=lambda _root, _stems: {"alpha": False},
        load_retired_parent_children=lambda _path: {},
        helpers=sync_reconciliation.RemoteHelpers(
            canonical_region_paths=lambda stem: {"polygons": f"{stem}/polygons.parquet"},
            planner_cls=cast(Any, Planner),
        ),
    )

    assert fetch_calls == [{"repo_id": "org/dataset", "hub": hub, "token": "token"}]
    assert planner_calls == [
        {
            "data_root": data_root,
            "inventory": inventory,
            "stems": {"alpha"},
            "augmentation_current": {"alpha": False},
        }
    ]
    assert result.inventory is inventory
    assert result.plan is not None
    assert result.stems_with_gaps == set()
    assert result.core_repaired is False


def test_local_augmentation_validation_passes_progress_policy_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    captured: dict[str, Any] = {}

    class Progress:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> dict[str, bool]:
            return {stem: captured["validator"](stem) for stem in captured["stems"]}

    def current(root: DataRoot, stem: str) -> bool:
        captured.setdefault("validated", []).append((root, stem))
        return stem == "alpha"

    monkeypatch.setattr(sync_reconciliation, "LocalValidationProgress", Progress)
    monkeypatch.setattr(sync_reconciliation, "augmentation_is_current", current)
    result = sync_reconciliation.validate_local_augmentation_state(
        data_root,
        ["alpha", "beta"],
    )

    assert result == {"alpha": True, "beta": False}
    assert captured["stems"] == ["alpha", "beta"]
    assert captured["progress_interval_s"] == 30.0
    assert captured["quiet_threshold"] == 25
    assert captured["phase_label"] == "regions"
    assert captured["validated"] == [(data_root, "alpha"), (data_root, "beta")]


def test_plan_prepared_sync_forwards_all_inputs_and_records_repair_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    settings = Settings(force=False, skip_existing=False)
    pbfs = [Path("alpha.osm.pbf")]
    remote = RemoteReconciliation(
        None,
        cast(Any, type("Plan", (), {"missing": {("alpha", "polygons")}})()),
        {"alpha": True},
        {"alpha"},
        {},
        False,
    )
    states = [RegionSyncState("alpha", pbfs[0], SyncAction.PUBLISH)]
    observed: dict[str, Any] = {}

    def manifest(path: Path) -> dict[str, dict[str, Any]]:
        observed["manifest_path"] = path
        return {"alpha.osm.pbf": {}, "beta.osm.pbf": {}}

    monkeypatch.setattr(sync_planning, "load_manifest", manifest)

    def plan_states(actual_pbfs: list[Path], **kwargs: Any) -> list[RegionSyncState]:
        observed["states"] = (actual_pbfs, kwargs)
        return states

    monkeypatch.setattr(sync_planning, "load_pending_publications", lambda _root: {"queued"})
    monkeypatch.setattr(sync_planning, "plan_sync_states_with_recovery", plan_states)

    result = sync_planning._plan_prepared_sync(
        pbfs,
        input_stems={"alpha"},
        remote_state=remote,
        data_root=data_root,
        settings=settings,
        push_enabled=True,
    )

    assert observed["manifest_path"] == data_root.processed_manifests / "processed_pbfs.json"
    actual_pbfs, state_kwargs = observed["states"]
    assert actual_pbfs is pbfs
    assert state_kwargs == {
        "inventory": sync_planning.StemInventory(
            input_stems={"alpha"}, core_stems={"alpha", "beta"}, current_augmentation={"alpha"}
        ),
        "force": True,
        "pending_stems": {"queued", "alpha"},
        "recovery_stems": set(),
        "processed_path": data_root.processed,
        "plan_link_migration": sync_planning.plan_link_migration,
    }
    assert result == PreparedSyncPlan(
        pbfs=pbfs,
        input_stems={"alpha"},
        all_pending_stems={"queued", "alpha"},
        states=states,
        remote_state=remote,
        core_will_be_repaired=True,
    )


def test_current_augmentation_and_pending_sets_follow_push_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    validation_calls: list[tuple[DataRoot, list[str]]] = []

    def validate(root: DataRoot, stems: list[str]) -> dict[str, bool]:
        validation_calls.append((root, stems))
        return {"local": True, "stale": False}

    monkeypatch.setattr(sync_planning, "validate_local_augmentation_state", validate)

    assert sync_planning._current_augmentation_for_plan(
        True,
        {"remote": True, "stale": False},
        data_root,
        {"alpha", "beta"},
    ) == {"remote"}
    assert sync_planning._current_augmentation_for_plan(
        False,
        {"ignored": True},
        data_root,
        {"beta", "alpha"},
    ) == {"local"}
    assert validation_calls == [(data_root, ["alpha", "beta"])]

    monkeypatch.setattr(sync_planning, "load_pending_publications", lambda _root: {"queued"})
    assert sync_planning._pending_stems_for_plan(data_root, {"gap"}, push_enabled=True) == {
        "queued",
        "gap",
    }
    assert sync_planning._pending_stems_for_plan(data_root, {"gap"}, push_enabled=False) == {
        "queued"
    }


def test_prepare_sync_plan_wires_local_remote_and_derived_plan_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = argparse.Namespace(input=Path("raw"))
    data_root = DataRoot(tmp_path)
    settings = Settings()
    inventory = RemoteInventory({"remote.parquet"})
    hub = object()
    pbfs = [Path("alpha.osm.pbf"), Path("retired.osm.pbf")]
    input_stems = {"alpha"}
    planner = cast(Any, type("Planner", (), {}))

    def canonical_paths(stem: str) -> dict[str, str]:
        return {"polygons": f"{stem}/polygons.parquet"}

    reconciliation_plan = type("Plan", (), {"missing": {("alpha", "polygons")}})()
    remote = RemoteReconciliation(
        inventory,
        cast(Any, reconciliation_plan),
        {"alpha": True},
        {"alpha"},
        {},
        False,
    )
    calls: dict[str, Any] = {}

    def prepare_rules(
        data_path: Path,
        *,
        dry_run: bool,
    ) -> tuple[list[PreparedRule], list[RuleAudit]]:
        calls["containment"] = (data_path, dry_run)
        return [], []

    def retired_children(path: Path) -> set[str]:
        calls["retired_children"] = path
        return {"retired"}

    def collect(inputs: list[Path]) -> list[Path]:
        calls["inputs"] = inputs
        return pbfs

    def migrate(root: DataRoot, stems: set[str]) -> None:
        calls["migration"] = (root, stems)

    def remote_helpers(enabled: bool) -> tuple[Any, Any]:
        calls["helpers"] = enabled
        return planner, canonical_paths

    def remote_plan(**kwargs: Any) -> RemoteReconciliation:
        calls["remote"] = kwargs
        return remote

    states = [RegionSyncState("alpha", pbfs[0], SyncAction.PUBLISH)]

    def plan_states(actual_pbfs: list[Path], **kwargs: Any) -> list[RegionSyncState]:
        calls["states"] = (actual_pbfs, kwargs)
        return states

    monkeypatch.setattr(containment_migration, "prepare_safe_rules", prepare_rules)
    monkeypatch.setattr(containment_migration, "load_retired_children", retired_children)
    monkeypatch.setattr(containment_migration, "load_retired_parent_children", lambda _path: {})
    monkeypatch.setattr(sync_planning, "collect_pbfs", collect)
    monkeypatch.setattr(sync_planning, "run_pre_publication_migration", migrate)
    monkeypatch.setattr(sync_planning, "load_manifest", lambda _path: {"alpha.osm.pbf": {}})
    monkeypatch.setattr(sync_planning, "load_pending_publications", lambda _root: {"queued"})
    monkeypatch.setattr(sync_planning, "plan_sync_states_with_recovery", plan_states)
    monkeypatch.setattr(sync_planning, "remote_reconciliation_helpers", remote_helpers)
    monkeypatch.setattr(sync_planning, "prepare_remote_reconciliation", remote_plan)

    result = sync_planning.prepare_sync_plan(
        args,
        data_root=data_root,
        settings=settings,
        push_enabled=True,
        dry_run=True,
        remote_inventory=inventory,
        hub=cast(Any, hub),
    )

    assert calls["inputs"] == [Path("raw")]
    assert calls["containment"] == (data_root.path, True)
    assert calls["retired_children"] == data_root.processed
    assert calls["migration"] == (data_root, input_stems)
    assert calls["helpers"] is True
    assert calls["states"][0] == [pbfs[0]]
    assert calls["states"][1] == {
        "inventory": sync_planning.StemInventory(
            input_stems=input_stems, core_stems={"alpha"}, current_augmentation={"alpha"}
        ),
        "force": True,
        "pending_stems": {"queued", "alpha"},
        "recovery_stems": set(),
        "processed_path": data_root.processed,
        "plan_link_migration": sync_planning.plan_link_migration,
    }
    assert calls["remote"] == {
        "enabled": True,
        "data_root": data_root,
        "settings": settings,
        "input_stems": input_stems,
        "source": sync_reconciliation.RemoteSource(
            hub=cast(Any, hub), inventory_override=inventory
        ),
        "validate_augmentation": sync_planning.validate_local_augmentation_state,
        "load_retired_parent_children": containment_migration.load_retired_parent_children,
        "helpers": sync_reconciliation.RemoteHelpers(
            canonical_region_paths=canonical_paths, planner_cls=planner
        ),
    }
    assert result == sync_planning.PreparedSyncPlan(
        pbfs=[pbfs[0]],
        input_stems=input_stems,
        all_pending_stems={"queued", "alpha"},
        states=states,
        remote_state=remote,
        core_will_be_repaired=True,
    )


def test_prepare_local_inputs_applies_push_flags_and_migrates_active_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = argparse.Namespace(input=Path("raw"))
    data_root = DataRoot(tmp_path)
    retired = {"retired"}
    pbfs = [
        Path("alpha.osm.pbf"),
        Path("retired.osm.pbf"),
        Path("beta.osm.pbf"),
    ]
    calls: dict[str, Any] = {}

    def prepare_rules(
        data_path: Path,
        *,
        dry_run: bool,
    ) -> tuple[list[PreparedRule], list[RuleAudit]]:
        calls["rules"] = {"data_path": data_path, "dry_run": dry_run}
        return [], []

    def collect(inputs: list[Path]) -> list[Path]:
        calls["inputs"] = inputs
        return pbfs

    def migrate(root: DataRoot, stems: set[str]) -> None:
        calls["migration"] = (root, stems)

    monkeypatch.setattr(containment_migration, "load_retired_children", lambda _path: retired)
    monkeypatch.setattr(containment_migration, "prepare_safe_rules", prepare_rules)
    monkeypatch.setattr(sync_planning, "collect_pbfs", collect)
    monkeypatch.setattr(sync_planning, "run_pre_publication_migration", migrate)
    active_pbfs, stems = sync_planning._prepare_local_inputs(
        args,
        data_root=data_root,
        push_enabled=True,
        dry_run=True,
    )

    assert calls["inputs"] == [Path("raw")]
    assert calls["rules"] == {"data_path": data_root.path, "dry_run": True}
    assert active_pbfs == [Path("alpha.osm.pbf"), Path("beta.osm.pbf")]
    assert stems == {"alpha", "beta"}
    assert calls["migration"] == (data_root, {"alpha", "beta"})


def test_core_repair_and_sync_plan_logging_preserve_action_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = {("publish", "polygons"), ("augment", "polygon_articles")}
    assert sync_reconciliation.core_repair_required(SyncAction.PROCESS, "complete", set()) is True
    assert sync_reconciliation.core_repair_required(SyncAction.PUBLISH, "publish", missing) is True
    assert sync_reconciliation.core_repair_required(SyncAction.AUGMENT, "augment", missing) is True
    assert (
        sync_reconciliation.core_repair_required(SyncAction.COMPLETE, "complete", missing) is False
    )

    states = [
        RegionSyncState("a", Path("a.osm.pbf"), SyncAction.RECOVERY),
        RegionSyncState("b", Path("b.osm.pbf"), SyncAction.AUGMENT),
        RegionSyncState("c", Path("c.osm.pbf"), SyncAction.PUBLISH),
        RegionSyncState("d", Path("d.osm.pbf"), SyncAction.PROCESS),
        RegionSyncState("e", Path("e.osm.pbf"), SyncAction.COMPLETE),
    ]
    with caplog.at_level("INFO", logger="osm_polygon_wikidata_only.cli"):
        sync_planning.log_sync_plan(states)
    assert caplog.messages[-1] == (
        "Unified sync plan: 1 recovery audit, 1 augmentation backlog, 1 publish, "
        "1 core missing, 1 complete"
    )


def test_core_will_be_repaired_requires_push_and_checks_state_stems() -> None:
    states = [
        RegionSyncState("alpha", Path("alpha.osm.pbf"), SyncAction.PUBLISH),
        RegionSyncState("beta", Path("beta.osm.pbf"), SyncAction.COMPLETE),
    ]
    plan = cast(Any, type("Plan", (), {"missing": {("alpha", "polygon_articles")}})())

    assert sync_planning._core_will_be_repaired(states, plan, push_enabled=True) is True
    assert sync_planning._core_will_be_repaired(states, None, push_enabled=True) is False
    assert sync_planning._core_will_be_repaired(states, plan, push_enabled=False) is False
    assert (
        sync_planning._core_will_be_repaired(
            [RegionSyncState("beta", Path("beta.osm.pbf"), SyncAction.PUBLISH)],
            plan,
            push_enabled=True,
        )
        is False
    )


def test_remote_reconciliation_summary_passes_metadata_signal_to_logger() -> None:
    messages: list[str] = []
    sync_reconciliation.log_remote_reconciliation_summary(
        stems_with_gaps=set(),
        core_repaired=False,
        metadata_repaired=True,
        log=messages.append,
    )
    assert messages == ["Remote reconciliation complete: README and maps refreshed"]


def test_recovery_audit_guard_reports_each_blocked_region() -> None:
    audit = type(
        "Audit",
        (),
        {
            "regions": [
                type("Region", (), {"stem": "alpha", "blocked_reason": "missing links"})(),
                type("Region", (), {"stem": "beta", "blocked_reason": "invalid schema"})(),
            ]
        },
    )()
    with pytest.raises(RuntimeError) as error:
        sync_planning.ensure_recovery_audit_unblocked(cast(Any, audit))
    assert str(error.value) == (
        "Wikidata integrity audit blocked this region; its files were not changed: "
        "alpha: missing links; beta: invalid schema"
    )


def test_plan_sync_states_with_recovery_forwards_forced_and_migratable_stems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbfs = [Path("alpha.osm.pbf"), Path("beta.osm.pbf")]
    migration = LinkMigrationPlan(
        processed_dir=Path("processed"),
        stems=(
            LinkStemPlan("alpha", StemClassification.CANONICAL, "", "", "", "", 0, None),
            LinkStemPlan("beta", StemClassification.MIGRATABLE, "", "", "", "", 0, None),
        ),
    )
    observed: dict[str, Any] = {}

    def plan_links(processed: Path, *, stems: set[str]) -> LinkMigrationPlan:
        observed["migration"] = (processed, stems)
        return migration

    def plan(actual_pbfs: list[Path], **kwargs: Any) -> list[RegionSyncState]:
        observed["plan"] = (actual_pbfs, kwargs)
        return []

    monkeypatch.setattr(sync_planning, "plan_sync_states", plan)
    result = sync_planning.plan_sync_states_with_recovery(
        pbfs,
        inventory=sync_planning.StemInventory(
            input_stems={"alpha", "beta"}, core_stems={"alpha"}, current_augmentation={"alpha"}
        ),
        force=True,
        pending_stems={"pending"},
        recovery_stems={"forced"},
        processed_path=Path("processed"),
        plan_link_migration=plan_links,
    )

    assert result == []
    assert observed["migration"] == (Path("processed"), {"alpha", "beta"})
    assert observed["plan"] == (
        pbfs,
        {
            "core_stems": {"alpha"},
            "augmentation_stems": {"alpha"},
            "force": True,
            "pending_stems": {"pending"},
            "recovery_stems": {"forced", "beta"},
        },
    )
