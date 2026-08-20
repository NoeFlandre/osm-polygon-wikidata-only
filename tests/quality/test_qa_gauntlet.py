"""Contracts for the deterministic, fail-fast QA gauntlet."""

from __future__ import annotations

from collections.abc import Sequence

from scripts.quality.qa_gauntlet import build_stages, run_gauntlet


def test_gauntlet_stages_are_in_the_required_order() -> None:
    assert [stage.name for stage in build_stages()] == [
        "baseline",
        "ruff",
        "ty",
        "tests",
        "acceptance tests",
        "architecture checks",
        "CRAP",
        "mutation tests",
        "smoke test",
        "diff review",
    ]


def test_gauntlet_delegates_each_stage_to_just() -> None:
    commands = [stage.command for stage in build_stages()]

    assert commands == [
        ("just", "baseline"),
        ("just", "ruff"),
        ("just", "ty"),
        ("just", "tests"),
        ("just", "acceptance-tests"),
        ("just", "architecture-checks"),
        ("just", "crap-all"),
        ("just", "mutation"),
        ("just", "smoke-test"),
        ("just", "diff-review"),
    ]


def test_gauntlet_stops_at_first_failed_stage() -> None:
    calls: list[tuple[str, ...]] = []

    # A failing command is identified by its stage name, not a shell string.
    def failing_runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 9 if tuple(command) == ("just", "crap-all") else 0

    assert run_gauntlet(failing_runner) == 9
    assert calls == [
        ("just", "baseline"),
        ("just", "ruff"),
        ("just", "ty"),
        ("just", "tests"),
        ("just", "acceptance-tests"),
        ("just", "architecture-checks"),
        ("just", "crap-all"),
    ]


def test_gauntlet_returns_zero_when_all_stages_pass() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 0

    assert run_gauntlet(runner) == 0
    assert len(calls) == len(build_stages())
