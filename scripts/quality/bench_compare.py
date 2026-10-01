"""Compare a pytest-benchmark JSON run with a committed baseline (issue #125).

A benchmark regresses when its median is more than ``--threshold`` slower than
the baseline median. In ``warn`` mode regressions are reported but the exit
status stays zero; ``enforce`` mode fails the build. Benchmarks absent from the
baseline are listed as new and never fail. A benchmark that is in the baseline but missing from the run also fails in
enforce mode, so coverage cannot silently disappear. A missing baseline is reported and
also never fails, so the first run on a new runner type can record one.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_THRESHOLD = 0.25


def medians(report: Mapping[str, object]) -> dict[str, float]:
    """Return ``{benchmark fullname: median seconds}`` from a benchmark JSON report."""
    entries = report.get("benchmarks")
    if not isinstance(entries, list):
        raise ValueError("benchmark report has no 'benchmarks' list")
    return {str(entry["fullname"]): float(entry["stats"]["median"]) for entry in entries}


def regressions(
    baseline: Mapping[str, float], current: Mapping[str, float], threshold: float
) -> list[tuple[str, float]]:
    """Return ``(name, slowdown ratio)`` for medians slower than the allowed threshold."""
    return sorted(
        (name, median / baseline[name] - 1.0)
        for name, median in current.items()
        if baseline.get(name, 0.0) > 0.0 and median / baseline[name] - 1.0 > threshold
    )


def missing(baseline: Mapping[str, float], current: Mapping[str, float]) -> list[str]:
    """Return baseline benchmarks absent from the current run (deleted or skipped)."""
    return sorted(set(baseline) - set(current))


def _load(path: Path) -> dict[str, float]:
    return medians(json.loads(path.read_bytes()))


def _missing_lines(baseline: Mapping[str, float], current: Mapping[str, float]) -> list[str]:
    return [
        f"MISSING {name}: in the baseline but not in this run"
        for name in missing(baseline, current)
    ]


def _regression_lines(
    baseline: Mapping[str, float], current: Mapping[str, float], threshold: float
) -> list[str]:
    return [
        f"REGRESSION {name}: median {slowdown:+.0%} vs baseline"
        for name, slowdown in regressions(baseline, current, threshold)
    ]


def _report(baseline: Mapping[str, float], current: Mapping[str, float], threshold: float) -> bool:
    """Print the comparison and return whether any benchmark regressed or vanished."""
    added = [f"NEW {name}" for name in sorted(set(current) - set(baseline))]
    absent = _missing_lines(baseline, current)
    found = _regression_lines(baseline, current, threshold)
    clean = [] if found else [f"No median regression above {threshold:.0%}."]
    print("\n".join([*added, *absent, *found, *clean]))
    return bool(found or absent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--mode", choices=("warn", "enforce"), default="warn")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.baseline.is_file():
        print(f"No baseline at {args.baseline}; nothing to compare against.")
        return 0
    regressed = _report(_load(args.baseline), _load(args.current), args.threshold)
    return 1 if regressed and args.mode == "enforce" else 0


if __name__ == "__main__":
    raise SystemExit(main())
