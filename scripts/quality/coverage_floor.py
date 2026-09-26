"""Enforce a per-file line-coverage floor on a coverage.py JSON report.

The aggregate ``fail_under`` threshold can hide individual modules that are
barely exercised. This check fails when any measured file falls below the
floor, except for explicitly listed exemptions (files that only run in a
separate process, where in-process coverage cannot observe them).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileCoverage:
    """Line coverage of one measured source file."""

    path: str
    percent: float
    statements: int


def file_coverages(report: Mapping[str, object]) -> list[FileCoverage]:
    """Return per-file line coverage from a coverage.py JSON report."""

    files = report.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("coverage report has no 'files' mapping")
    result: list[FileCoverage] = []
    for path, entry in files.items():
        summary = entry.get("summary") if isinstance(entry, Mapping) else None
        if not isinstance(summary, Mapping):
            raise ValueError(f"coverage entry for {path} has no summary")
        result.append(_file_coverage(str(path), summary))
    return sorted(result, key=lambda item: item.path)


def _file_coverage(path: str, summary: Mapping[str, object]) -> FileCoverage:
    statements = _count(summary.get("num_statements"), path)
    covered = _count(summary.get("covered_lines"), path)
    percent = 100.0 if statements == 0 else 100.0 * covered / statements
    return FileCoverage(path, percent, statements)


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"coverage entry for {path} lacks line counts")
    return value


def below_floor(
    coverages: Iterable[FileCoverage],
    *,
    minimum: float,
    exempt: Sequence[str] = (),
) -> list[FileCoverage]:
    """Return non-exempt files whose line coverage is below ``minimum`` percent."""

    if not math.isfinite(minimum) or not 0.0 <= minimum <= 100.0:
        raise ValueError("minimum must be a percentage between 0 and 100")
    return [
        item
        for item in coverages
        if item.percent < minimum and not _is_exempt(item.path, exempt)
    ]


def _is_exempt(path: str, exempt: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized == entry or normalized.endswith("/" + entry) for entry in exempt)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True, help="coverage.py JSON report")
    parser.add_argument("--minimum", type=float, default=85.0, help="per-file floor in percent")
    parser.add_argument(
        "--exempt",
        action="append",
        default=[],
        help="repository-relative file excluded from the floor (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Check the report and print every file below the floor."""

    args = _parse_args(argv)
    report = json.loads(args.coverage.read_text(encoding="utf-8"))
    failures = below_floor(file_coverages(report), minimum=args.minimum, exempt=args.exempt)
    if failures:
        print(f"Files below the {args.minimum:g}% per-file coverage floor:", file=sys.stderr)
        for item in failures:
            print(f"  {item.percent:5.1f}%  {item.path} ({item.statements} statements)", file=sys.stderr)
        return 1
    print(f"Per-file coverage floor passed: every file is at least {args.minimum:g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
