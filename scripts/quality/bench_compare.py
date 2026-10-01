"""Compare a pytest-benchmark JSON run with a committed baseline (issue #125).

A benchmark regresses when its median is more than ``--threshold`` slower than
the baseline median. In ``warn`` mode regressions are reported but the exit
status stays zero; ``enforce`` mode fails the build. Benchmarks absent from the
baseline are listed as new and never fail. A missing baseline is reported and
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


def _load(path: Path) -> dict[str, float]:
    return medians(json.loads(path.read_text(encoding="utf-8")))


def _report(baseline: Mapping[str, float], current: Mapping[str, float], threshold: float) -> bool:
    """Print the comparison and return whether any benchmark regressed."""
    for name in sorted(set(current) - set(baseline)):
        print(f"NEW {name}")
    found = regressions(baseline, current, threshold)
    for name, slowdown in found:
        print(f"REGRESSION {name}: median {slowdown:+.0%} vs baseline")
    if not found:
        print(f"No median regression above {threshold:.0%}.")
    return bool(found)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--mode", choices=("warn", "enforce"), default="warn")
    args = parser.parse_args(argv)
    if not args.baseline.is_file():
        print(f"No baseline at {args.baseline}; nothing to compare against.")
        return 0
    regressed = _report(_load(args.baseline), _load(args.current), args.threshold)
    return 1 if regressed and args.mode == "enforce" else 0


if __name__ == "__main__":
    raise SystemExit(main())
