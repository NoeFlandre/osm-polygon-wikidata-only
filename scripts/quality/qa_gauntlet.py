"""Run the project's deterministic quality checks in a fixed fail-fast order."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Stage:
    """One named command in the completion gate."""

    name: str
    command: tuple[str, ...]


CommandRunner = Callable[[Sequence[str]], int]


def build_stages() -> tuple[Stage, ...]:
    """Return the canonical QA stages in their required execution order."""

    return (
        Stage("baseline", ("just", "baseline")),
        Stage("ruff", ("just", "ruff")),
        Stage("ty", ("just", "ty")),
        Stage("tests", ("just", "tests")),
        Stage("acceptance tests", ("just", "acceptance-tests")),
        Stage("architecture checks", ("just", "architecture-checks")),
        Stage("CRAP", ("just", "crap-all")),
        Stage("mutation tests", ("just", "mutation")),
        Stage("smoke test", ("just", "smoke-test")),
        Stage("diff review", ("just", "diff-review")),
    )


def _run_command(command: Sequence[str]) -> int:
    """Run one command without a shell and return its exit status."""

    try:
        completed = subprocess.run(command, check=False)  # noqa: S603 - commands are fixed below
    except OSError as exc:
        print(f"Unable to start {' '.join(command)}: {exc}", file=sys.stderr)
        return 127
    return completed.returncode


def run_gauntlet(runner: CommandRunner = _run_command) -> int:
    """Run each stage once, stopping immediately at the first failure."""

    stages = build_stages()
    for index, stage in enumerate(stages, start=1):
        print(f"QA stage {index}/{len(stages)}: {stage.name}", flush=True)
        status = runner(stage.command)
        if status:
            print(
                f"QA gauntlet stopped at {stage.name} (exit {status})",
                file=sys.stderr,
                flush=True,
            )
            return status
    print("QA gauntlet passed: all stages completed", flush=True)
    return 0


def main() -> int:
    """Run the canonical completion gate."""

    return run_gauntlet()


if __name__ == "__main__":
    raise SystemExit(main())
