"""Fail a quality gate when mutmut reports anything other than killed."""

from __future__ import annotations

import sys
from collections.abc import Sequence

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


def ensure_all_killed(results: Sequence[MutationResult]) -> None:
    """Raise with mutant names unless every result is ``killed``."""

    if not results:
        raise MutationGateError("No mutants were reported")
    non_killed = [(name, status) for name, status in results if status != "killed"]
    if non_killed:
        details = ", ".join(f"{name}: {status}" for name, status in non_killed)
        raise MutationGateError(f"Non-killed mutants ({len(non_killed)}): {details}")


def main() -> int:
    """Read a mutmut report from stdin and enforce the zero-survivor policy."""

    results = parse_results(sys.stdin.read())
    ensure_all_killed(results)
    print(f"Mutation gate passed: {len(results)} mutants killed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
