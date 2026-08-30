"""Read-only audit of configured whole-file containment retirements."""

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
    return [report for rule in rules if rule.children for report in (audit_rule(processed, rule),)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    processed = args.data_root / "processed"
    retired = load_retired_children(processed)
    reports = _audit_reports(processed, _pending_rules(retired))
    payload = _audit_payload(retired, reports)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not payload["blocked_parents"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
