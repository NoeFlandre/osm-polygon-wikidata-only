"""``osm-polygon-wikidata-only audit-containment``: read-only containment audit.

Audits the configured whole-file containment retirements and prints a JSON
report. Exit status is 1 when any parent is blocked or the data root cannot
be read, and 0 otherwise. The exit status 2 is not used for blocked parents,
so it keeps meaning an argparse usage error. See ``docs/cli-reference.md``.
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

from .errors import CliFailure, report_cli_error
from .parser import (
    AUDIT_CONTAINMENT_DESCRIPTION as DESCRIPTION,
)
from .parser import (
    add_audit_containment_arguments as add_arguments,
)

PROG = "osm-polygon-wikidata-only audit-containment"


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


def _write_payload(output: Path | None, rendered: str) -> None:
    """Write the rendered JSON report to ``output``, or to stdout when it is unset."""
    if output:
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


def _blocked_result(reports: Sequence[RuleAudit]) -> int:
    """Return 0 when every audited parent is safe, else the one-line stderr failure."""
    blocked = [report.parent for report in reports if not report.safe_to_stage]
    if not blocked:
        return 0
    return report_cli_error(
        PROG, CliFailure(f"{len(blocked)} blocked parent(s): {', '.join(blocked)}")
    )


def run(args: argparse.Namespace) -> int:
    """Audit the pending containment rules and emit the JSON report.

    A blocked parent is an expected failure: the report is still printed, and
    the one-line stderr summary carries exit status 1.
    """
    processed = args.data_root / "processed"
    try:
        retired = load_retired_children(processed)
        reports = _audit_reports(processed, _pending_rules(retired))
    except (OSError, ValueError) as error:
        return report_cli_error(PROG, error)
    payload = _audit_payload(retired, reports)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_payload(args.output, rendered)
    return _blocked_result(reports)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the audit-containment arguments. A usage error or ``--help`` exits here."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit from command-line arguments and return the exit status."""
    return run(parse_args(argv))


__all__ = ["DESCRIPTION", "add_arguments", "main", "parse_args", "run"]
