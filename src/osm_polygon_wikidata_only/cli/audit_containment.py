"""``osm-polygon-wikidata-only audit-containment``: read-only containment audit.

Audits the configured whole-file containment retirements and prints a JSON
report. Exit status is 2 when any parent is blocked, 0 otherwise.
``scripts/audit_containment.py`` is a thin shim over :func:`main`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Sequence
from dataclasses import asdict
from pathlib import Path

from osm_polygon_wikidata_only.pipeline.containment_migration import (
    RuleAudit,
    audit_rule,
    load_retired_children,
)
from osm_polygon_wikidata_only.pipeline.containment_policy import (
    CONTAINMENT_RULES,
    ContainmentRule,
)


def _audit_payload(retired: Collection[str], reports: Sequence[RuleAudit]) -> dict[str, object]:
    safe_parents: list[str] = []
    blocked_parents: list[str] = []
    serialized_reports: list[dict[str, object]] = []
    for report in reports:
        serialized_reports.append(asdict(report) | {"safe_to_stage": report.safe_to_stage})
        if report.safe_to_stage:
            safe_parents.append(report.parent)
        else:
            blocked_parents.append(report.parent)
    return {
        "retired_children": sorted(retired),
        "safe_parents": safe_parents,
        "blocked_parents": blocked_parents,
        "reports": serialized_reports,
    }


def _pending_rules(retired: Collection[str]) -> tuple[ContainmentRule, ...]:
    return tuple(
        ContainmentRule(
            rule.parent,
            tuple(child for child in rule.children if child not in retired),
        )
        for rule in CONTAINMENT_RULES
    )


def _audit_reports(processed: Path, rules: Sequence[ContainmentRule]) -> list[RuleAudit]:
    return [audit_rule(processed, rule) for rule in rules if rule.children]


DESCRIPTION = "Read-only audit of configured whole-file containment retirements."
EXIT_BLOCKED = 2


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the audit options on *parser*."""
    parser.add_argument("data_root", type=Path, help="Data root containing processed/")
    parser.add_argument("--output", type=Path, help="Write the JSON report here instead of stdout")


def run(args: argparse.Namespace) -> int:
    """Audit the pending containment rules and emit the JSON report."""
    processed = args.data_root / "processed"
    retired = load_retired_children(processed)
    reports = _audit_reports(processed, _pending_rules(retired))
    payload = _audit_payload(retired, reports)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not payload["blocked_parents"] else EXIT_BLOCKED


def main(argv: Sequence[str] | None = None) -> int:
    """Legacy ``scripts/audit_containment.py`` entry point."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    return run(parser.parse_args(argv))


__all__ = ["DESCRIPTION", "add_arguments", "main", "run"]
