"""Tests for the read-only containment audit payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline.containment_migration import ChildAudit, RuleAudit
from osm_polygon_wikidata_only.pipeline.containment_policy import ContainmentRule
from scripts import audit_containment
from scripts.audit_containment import _audit_payload


def test_audit_payload_separates_safe_and_blocked_parents() -> None:
    reports = (
        RuleAudit("safe-parent", (ChildAudit("child", ()),), ()),
        RuleAudit("blocked-parent", (), ("missing child",)),
    )

    payload = _audit_payload({"retired-child"}, reports)

    assert payload["retired_children"] == ["retired-child"]
    assert payload["safe_parents"] == ["safe-parent"]
    assert payload["blocked_parents"] == ["blocked-parent"]
    assert payload["reports"][1]["safe_to_stage"] is False


def test_audit_main_writes_json_and_returns_blocked_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = ContainmentRule("parent-latest", ("child-latest", "retired-latest"))
    monkeypatch.setattr(audit_containment, "CONTAINMENT_RULES", (rule,))
    monkeypatch.setattr(
        audit_containment,
        "load_retired_children",
        lambda _processed: {"retired-latest"},
    )
    monkeypatch.setattr(
        audit_containment,
        "audit_rule",
        lambda _processed, pending: RuleAudit(pending.parent, (), ("blocked",)),
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_containment", str(tmp_path), "--output", str(output)],
    )

    assert audit_containment.main() == 2
    assert json.loads(output.read_text(encoding="utf-8"))["blocked_parents"] == ["parent-latest"]
