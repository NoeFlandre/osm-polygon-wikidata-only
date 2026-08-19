"""Contracts for containment manifest statistics aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_wikidata_only.pipeline.containment_migration as containment_migration
from osm_polygon_wikidata_only.pipeline.containment_migration import (
    ChildAudit,
    PreparedRule,
    RuleAudit,
    _document_manifest_stats,
    _has_required_files,
    _pending_rule,
    _polygon_manifest_stats,
    load_retired_parent_children,
    prepare_safe_rules,
)
from osm_polygon_wikidata_only.pipeline.containment_policy import (
    TABLE_CONTRACTS,
    ContainmentRule,
)

PARENT = "parent-latest"
CHILD = "child-latest"


def _seed_required_files(processed: Path) -> None:
    for contract in TABLE_CONTRACTS:
        schema = pa.schema(
            [
                pa.field(column, pa.int64() if column == "osm_id" else pa.string())
                for column in contract.identity_columns
            ]
        )
        row = {
            column: 1 if column == "osm_id" else f"{column}-1"
            for column in contract.identity_columns
        }
        for stem in (PARENT, CHILD):
            path = processed / contract.subdir / f"{stem}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def test_polygon_manifest_stats_count_values_and_ignore_bad_tag_json() -> None:
    rows = [
        {
            "area_bucket": "small",
            "tag_keys": '["wikidata", "name"]',
            "wikidata": "Q42",
            "has_wikipedia": True,
            "text_available": True,
        },
        {
            "area_bucket": "large",
            "tag_keys": "not-json",
            "wikidata": "",
            "has_wikipedia": False,
            "text_available": False,
        },
        {
            "area_bucket": "small",
            "tag_keys": '["name"]',
            "wikidata": "Q42",
            "has_wikipedia": True,
            "text_available": False,
        },
    ]

    assert _polygon_manifest_stats(rows) == {
        "polygon_count": 3,
        "unique_wikidata_count": 1,
        "rows_with_wikipedia": 2,
        "rows_with_full_text": 1,
        "area_bucket_counts": {"small": 2, "large": 1},
        "top_tag_keys": {"wikidata": 1, "name": 2},
    }


def test_document_manifest_stats_sort_languages_and_sum_characters() -> None:
    rows = [
        {"language": "fr", "article_length_chars": 12},
        {"language": "en", "article_length_chars": 30},
        {"language": "fr", "article_length_chars": 8},
    ]

    assert _document_manifest_stats(rows) == {
        "article_count": 3,
        "language_count": 2,
        "languages": ["en", "fr"],
        "total_full_text_chars": 50,
    }


def test_retired_parent_children_groups_and_sorts_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "processed" / "manifests" / "containment_retirements.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "contained-region-v1",
                "retired": {
                    "child-b": {"parent": "parent"},
                    "child-a": {"parent": "parent"},
                    "other": {"parent": "another"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_retired_parent_children(tmp_path / "processed") == {
        "another": ("other",),
        "parent": ("child-a", "child-b"),
    }


def test_retired_parent_children_returns_empty_without_manifest(tmp_path: Path) -> None:
    assert load_retired_parent_children(tmp_path / "processed") == {}


def test_retired_parent_children_rejects_malformed_parent(tmp_path: Path) -> None:
    manifest = tmp_path / "processed" / "manifests" / "containment_retirements.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "contained-region-v1",
                "retired": {"child": {"parent": None}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed containment retirement entry"):
        load_retired_parent_children(tmp_path / "processed")


def test_pending_rule_excludes_retired_children() -> None:
    rule = ContainmentRule(PARENT, (CHILD, "other-latest"))
    assert _pending_rule(rule, frozenset({CHILD})) == ContainmentRule(PARENT, ("other-latest",))
    assert _pending_rule(rule, frozenset({CHILD, "other-latest"})) is None


def test_required_containment_files_are_checked_as_one_contract(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _seed_required_files(processed)
    rule = ContainmentRule(PARENT, (CHILD,))
    assert _has_required_files(processed, rule)
    (processed / "polygons" / f"{CHILD}.parquet").unlink()
    assert not _has_required_files(processed, rule)


def test_prepare_safe_rules_skips_retired_and_incomplete_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired = ContainmentRule(PARENT, (CHILD,))
    incomplete = ContainmentRule("other-parent-latest", ("other-child-latest",))
    monkeypatch.setattr(containment_migration, "CONTAINMENT_RULES", (retired, incomplete))
    monkeypatch.setattr(
        containment_migration, "load_retired_children", lambda _processed: frozenset({CHILD})
    )
    monkeypatch.setattr(
        containment_migration,
        "audit_rule",
        lambda *_args: pytest.fail("incomplete rules must not be audited"),
    )
    prepared, blocked = prepare_safe_rules(tmp_path, dry_run=False)
    assert prepared == ()
    assert blocked == ()


def test_prepare_safe_rules_collects_blocked_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _seed_required_files(processed)
    rule = ContainmentRule(PARENT, (CHILD,))
    blocked_audit = RuleAudit(PARENT, (ChildAudit(CHILD, ()),), ("unsafe",))
    monkeypatch.setattr(containment_migration, "CONTAINMENT_RULES", (rule,))
    monkeypatch.setattr(
        containment_migration, "load_retired_children", lambda _processed: frozenset()
    )
    monkeypatch.setattr(containment_migration, "audit_rule", lambda *_args: blocked_audit)
    prepared, blocked = prepare_safe_rules(tmp_path, dry_run=False)
    assert prepared == ()
    assert blocked == (blocked_audit,)


def test_prepare_safe_rules_dry_run_does_not_mutate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _seed_required_files(processed)
    rule = ContainmentRule(PARENT, (CHILD,))
    safe_audit = RuleAudit(PARENT, (ChildAudit(CHILD, ()),), ())
    monkeypatch.setattr(containment_migration, "CONTAINMENT_RULES", (rule,))
    monkeypatch.setattr(
        containment_migration, "load_retired_children", lambda _processed: frozenset()
    )
    monkeypatch.setattr(containment_migration, "audit_rule", lambda *_args: safe_audit)
    monkeypatch.setattr(
        containment_migration,
        "prepare_local_rule",
        lambda *_args: pytest.fail("dry-run must not prepare a rule"),
    )
    prepared, blocked = prepare_safe_rules(tmp_path, dry_run=True)
    assert prepared == ()
    assert blocked == ()


def test_prepare_safe_rules_prepares_audited_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _seed_required_files(processed)
    rule = ContainmentRule(PARENT, (CHILD,))
    safe_audit = RuleAudit(PARENT, (ChildAudit(CHILD, ()),), ())
    expected = PreparedRule(PARENT, (CHILD,))
    monkeypatch.setattr(containment_migration, "CONTAINMENT_RULES", (rule,))
    monkeypatch.setattr(
        containment_migration, "load_retired_children", lambda _processed: frozenset()
    )
    monkeypatch.setattr(containment_migration, "audit_rule", lambda *_args: safe_audit)
    monkeypatch.setattr(containment_migration, "prepare_local_rule", lambda *_args: expected)
    prepared, blocked = prepare_safe_rules(tmp_path, dry_run=False)
    assert prepared == (expected,)
    assert blocked == ()
