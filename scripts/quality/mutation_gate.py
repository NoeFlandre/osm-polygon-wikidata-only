"""Reject mutation regressions; permit only explicit, source-bound equivalence reviews."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

MutationResult = tuple[str, str]
_STATUSES = frozenset(
    {
        "killed",
        "survived",
        "no tests",
        "timeout",
        "suspicious",
        "skipped",
        "not checked",
        "segfault",
        "caught by type check",
    }
)


class MutationGateError(RuntimeError):
    """The mutation report is empty or contains a non-killed mutant."""


def parse_results(report: str) -> list[MutationResult]:
    """Parse the status lines emitted by ``mutmut results --all``."""

    results: list[MutationResult] = []
    for line in report.splitlines():
        stripped = line.strip()
        if ": " not in stripped:
            continue
        name, status = stripped.rsplit(": ", 1)
        if status in _STATUSES:
            results.append((name, status))
    return results


def _non_killed(results: Sequence[MutationResult]) -> list[MutationResult]:
    return [(name, status) for name, status in results if status != "killed"]


def ensure_all_killed(
    results: Sequence[MutationResult], *, equivalents: frozenset[str] = frozenset()
) -> None:
    """Reject non-killed results except explicitly reviewed, surviving equivalents."""

    if not results:
        raise MutationGateError("No mutants were reported")
    non_killed = _unreviewed(_non_killed(results), equivalents)
    if non_killed:
        details = ", ".join(f"{name}: {status}" for name, status in non_killed)
        raise MutationGateError(f"Non-killed mutants ({len(non_killed)}): {details}")


def _unreviewed(
    results: Sequence[MutationResult], equivalents: frozenset[str]
) -> list[MutationResult]:
    return [
        (name, status)
        for name, status in results
        if not (status == "survived" and name in equivalents)
    ]


def main(argv: Sequence[str] = ()) -> int:
    """Read a mutmut report and keep reviewed equivalents distinct from kills."""

    results = parse_results(sys.stdin.read())
    equivalents = _reviewed_names(argv, results)
    ensure_all_killed(results, equivalents=equivalents)
    killed = sum(status == "killed" for _, status in results)
    suffix = f"; {len(equivalents)} reviewed equivalents" if equivalents else ""
    print(f"Mutation gate passed: {killed} mutants killed{suffix}")
    return 0


def _reviewed_names(argv: Sequence[str], results: Sequence[MutationResult]) -> frozenset[str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equivalents", type=Path)
    parser.add_argument("--mutants-root", type=Path, default=Path("mutants"))
    args = parser.parse_args(argv)
    if args.equivalents is None:
        return frozenset()
    from scripts.quality.mutation_equivalents import reviewed_equivalents

    return reviewed_equivalents(
        results, args.equivalents, source_root=Path.cwd(), mutants_root=args.mutants_root
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
