"""Fail when any single test exceeds the per-test time budget (issue #125).

Reads a pytest ``--junitxml`` report, prints the slowest tests, and exits
non-zero when one of them is slower than the budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_BUDGET_SECONDS = 3.0
DEFAULT_OVERRIDES = Path(__file__).with_name("slow_test_budgets.json")
_TOP = 20


def test_durations(report: Path) -> list[tuple[float, str]]:
    """Return ``(seconds, test id)`` for every test case, slowest first."""
    root = ET.parse(report).getroot()  # noqa: S314 - our own pytest report
    cases = [
        (float(case.get("time", "0")), f"{case.get('classname', '')}::{case.get('name', '')}")
        for case in root.iter("testcase")
    ]
    return sorted(cases, reverse=True)


def over_budget(
    durations: Sequence[tuple[float, str]],
    budget: float,
    overrides: Mapping[str, float] | None = None,
) -> list[tuple[float, str]]:
    """Return the tests slower than their budget (``overrides`` raise it per test)."""
    limits = overrides or {}
    return [entry for entry in durations if entry[0] > limits.get(entry[1], budget)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    args = parser.parse_args(argv)
    durations = test_durations(args.junit_xml)
    for seconds, name in durations[:_TOP]:
        print(f"{seconds:7.2f}s  {name}")
    overrides = (
        json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides.is_file() else {}
    )
    offenders = over_budget(durations, args.budget, overrides)
    for seconds, name in offenders:
        print(f"OVER BUDGET {seconds:.2f}s > {args.budget:.2f}s: {name}", file=sys.stderr)
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
