"""Tests for the small, deterministic preparation helpers used by sync-dir."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import MigrationOperation
from osm_polygon_wikidata_only.cli.run_sync import (
    _active_pbfs,
    _containment_publications_for_remote,
    _core_repair_required,
    _enqueue_containment_retirement,
    _migration_stems_to_persist,
    _plan_sync_states,
    _prepare_containment_rules,
    _prepare_remote_reconciliation,
    _reconciliation_gap_counts,
    _reconciliation_summary_message,
    _remote_child_has_artifact,
)
from osm_polygon_wikidata_only.pipeline.sync_planner import SyncAction


def test_active_pbfs_excludes_retired_stems_without_reordering() -> None:
    """Retired shards are removed while active PBF order is preserved."""
    pbfs = [Path("b.osm.pbf"), Path("a.osm.pbf"), Path("c.osm.pbf")]

    assert _active_pbfs(pbfs, {"a"}) == [Path("b.osm.pbf"), Path("c.osm.pbf")]


def test_prepare_containment_rules_reports_prepared_and_blocked_rules() -> None:
    """Containment setup logs both successful and blocked rules."""
    info: list[tuple[str, tuple[Any, ...]]] = []
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    calls: list[tuple[Path, bool]] = []

    def prepare(path: Path, *, dry_run: bool) -> tuple[list[object], list[Any]]:
        calls.append((path, dry_run))
        return [object()], [SimpleNamespace(parent="parent", blockers=["reason"])]

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
    assert warnings == [("Containment retirement blocked for %s: %s", ("parent", "reason"))]


def test_core_repair_required_matches_action_and_missing_artifacts() -> None:
    """Only core work or missing core files marks a state for map refresh."""
    missing = {("region", "polygons")}

    assert _core_repair_required(SyncAction.PROCESS, "other", set()) is True
    assert _core_repair_required(SyncAction.PUBLISH, "region", missing) is True
    assert _core_repair_required(SyncAction.AUGMENT, "region", missing) is True
    assert _core_repair_required(SyncAction.PUBLISH, "other", missing) is False
    assert _core_repair_required(SyncAction.COMPLETE, "region", missing) is False


def test_plan_sync_states_adds_noncanonical_link_migrations_to_recovery() -> None:
    """Legacy link layouts are included in the recovery action set."""
    link_plan = SimpleNamespace(
        stems=[SimpleNamespace(stem="region", classification=SimpleNamespace(value="legacy"))]
    )

    states = _plan_sync_states(
        [Path("region.osm.pbf")],
        input_stems={"region"},
        core_stems={"region"},
        current_augmentation={"region"},
        force=False,
        pending_stems=set(),
        recovery_stems=set(),
        processed_path=Path("processed"),
        plan_link_migration=lambda _path, *, stems: link_plan,
    )

    assert [(state.stem, state.action) for state in states] == [("region", SyncAction.RECOVERY)]


def test_reconciliation_summary_message_has_stable_four_case_contract() -> None:
    """Summary wording reflects repair and map-refresh signals only."""
    assert _reconciliation_summary_message(2, True, False) == (
        "Remote reconciliation complete: 2 regions repaired; README and maps refreshed"
    )
    assert _reconciliation_summary_message(0, False, True) == (
        "Remote reconciliation complete: README and maps refreshed"
    )
    assert _reconciliation_summary_message(1, False, False) == (
        "Remote reconciliation complete: 1 regions repaired"
    )
    assert _reconciliation_summary_message(0, False, False) == (
        "Remote reconciliation complete: converged"
    )


def test_reconciliation_gap_counts_separate_core_and_text_artifacts() -> None:
    """Missing core and augmentation artifacts are counted independently."""
    missing = {
        ("a", "polygons"),
        ("a", "wikipedia/documents"),
        ("b", "wikidata/facts"),
    }

    assert _reconciliation_gap_counts({"a", "b", "c"}, missing) == (1, 2)


def test_prepare_remote_reconciliation_disabled_returns_empty_state() -> None:
    """Non-push runs avoid all remote inventory and validation work."""
    result = _prepare_remote_reconciliation(
        enabled=False,
        data_root=None,
        settings=None,
        input_stems=set(),
        hub=None,
        inventory_override=None,
        validate_augmentation=lambda *_: pytest.fail("must not validate"),
        load_retired_parent_children=lambda *_: pytest.fail("must not load"),
        canonical_region_paths=None,
        planner_cls=None,
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
        SimpleNamespace(stem="create", operation=MigrationOperation.CREATE_MISSING),
        SimpleNamespace(stem="upgrade", operation=MigrationOperation.UPGRADE_LEGACY),
        SimpleNamespace(stem="canonical", operation=MigrationOperation.ALREADY_CANONICAL),
    ]

    assert _migration_stems_to_persist(plans) == {"create", "upgrade"}


def test_containment_publications_keep_children_present_on_remote() -> None:
    """Only contained children with any canonical remote artifact are published."""
    inventory = SimpleNamespace(contains=lambda path: path == "child/polygons.parquet")

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
        _enqueue_containment_retirement(
            data_root=SimpleNamespace(processed=tmp_path),
            settings=SimpleNamespace(repo_id="org/repo"),
            parent_children=parent_children,
            upload_queue=queue,
            push_enabled=push_enabled,
        )
        is False
    )


def test_enqueue_containment_retirement_submits_one_remote_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[tuple[object, str]] = []

    def assemble(**kwargs: object) -> list[str]:
        assert kwargs["repo_id"] == "org/repo"
        assert kwargs["parent_children"] == {"parent": ("child", "other")}
        return ["operation"]

    class Queue:
        def submit(self, operations: object, description: str) -> None:
            submitted.append((operations, description))

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.cli.run_sync._assemble_containment_retirement_upload",
        assemble,
    )

    assert (
        _enqueue_containment_retirement(
            data_root=SimpleNamespace(processed=tmp_path),
            settings=SimpleNamespace(repo_id="org/repo"),
            parent_children={"parent": ("child", "other")},
            upload_queue=Queue(),
            push_enabled=True,
        )
        is True
    )
    assert submitted == [(["operation"], "Retire losslessly contained regional dataset shards")]
