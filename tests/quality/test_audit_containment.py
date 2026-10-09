"""Tests for the read-only containment audit command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import audit_containment, commands
from osm_polygon_wikidata_only.pipeline.containment_migration import ChildAudit, RuleAudit
from osm_polygon_wikidata_only.pipeline.containment_policy import ContainmentRule


def test_audit_main_separates_safe_and_blocked_parents_and_skips_retired_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules = (
        ContainmentRule("safe-latest", ("child-latest",)),
        ContainmentRule("blocked-latest", ("child-b-latest", "retired-latest")),
        ContainmentRule("done-latest", ("retired-latest",)),
    )
    audited: list[ContainmentRule] = []

    def audit(_processed: Path, pending: ContainmentRule) -> RuleAudit:
        audited.append(pending)
        if pending.parent == "safe-latest":
            return RuleAudit(pending.parent, (ChildAudit("child-latest", ()),), ())
        return RuleAudit(pending.parent, (), ("blocked",))

    monkeypatch.setattr(audit_containment, "CONTAINMENT_RULES", rules)
    monkeypatch.setattr(audit_containment, "load_retired_children", lambda _p: {"retired-latest"})
    monkeypatch.setattr(audit_containment, "audit_rule", audit)
    output = tmp_path / "audit.json"

    assert commands.main(["audit-containment", str(tmp_path), "--output", str(output)]) == 1
    assert capsys.readouterr().err == (
        "osm-polygon-wikidata-only audit-containment: error: 1 blocked parent(s): blocked-latest\n"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["retired_children"] == ["retired-latest"]
    assert payload["safe_parents"] == ["safe-latest"]
    assert payload["blocked_parents"] == ["blocked-latest"]
    assert [report["safe_to_stage"] for report in payload["reports"]] == [True, False]
    assert audited[1] == ContainmentRule("blocked-latest", ("child-b-latest",))


def test_audit_main_prints_the_payload_and_passes_without_blocked_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(audit_containment, "CONTAINMENT_RULES", ())
    monkeypatch.setattr(audit_containment, "load_retired_children", lambda _p: set())

    assert audit_containment.main([str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["blocked_parents"] == []


def test_audit_main_reports_unreadable_processed_root_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(_processed: Path, _rule: ContainmentRule) -> RuleAudit:
        raise FileNotFoundError("processed/polygons missing")

    monkeypatch.setattr(
        audit_containment, "CONTAINMENT_RULES", (ContainmentRule("parent", ("child",)),)
    )
    monkeypatch.setattr(audit_containment, "load_retired_children", lambda _p: set())
    monkeypatch.setattr(audit_containment, "audit_rule", missing)

    assert audit_containment.main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "osm-polygon-wikidata-only audit-containment: error: processed/polygons missing\n"
    )


def test_audit_main_reports_malformed_retirement_manifest_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def malformed(_processed: Path) -> set[str]:
        raise ValueError("Malformed containment retirement manifest")

    monkeypatch.setattr(audit_containment, "load_retired_children", malformed)

    assert audit_containment.main([str(tmp_path)]) == 1
    assert "Malformed containment retirement manifest" in capsys.readouterr().err
