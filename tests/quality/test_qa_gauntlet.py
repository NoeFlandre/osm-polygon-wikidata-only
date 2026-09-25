"""Contracts for the deterministic, fail-fast QA gauntlet."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from io import BufferedWriter, BytesIO, TextIOWrapper

import scripts.quality.qa_gauntlet as qa_gauntlet
from scripts.quality.qa_gauntlet import build_stages, run_gauntlet


def test_gauntlet_runs_the_required_stages_in_order_through_just() -> None:
    assert [(stage.name, stage.command) for stage in build_stages()] == [
        ("baseline", ("just", "baseline")),
        ("ruff", ("just", "ruff")),
        ("ty", ("just", "ty")),
        ("tests", ("just", "tests")),
        ("property tests", ("just", "property-tests")),
        ("acceptance tests", ("just", "acceptance-tests")),
        ("architecture checks", ("just", "architecture-checks")),
        ("CRAP", ("just", "crap-report")),
        ("mutation tests", ("just", "mutation")),
        ("smoke test", ("just", "smoke-test")),
        ("diff review", ("just", "diff-review")),
    ]


def test_main_runs_stage_commands_without_a_shell_and_returns_their_status(
    monkeypatch, capsys
) -> None:
    calls: list[tuple[Sequence[str], bool]] = []

    def fake_run(
        command: Sequence[str], *, check: bool
    ) -> subprocess.CompletedProcess[Sequence[str]]:
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(qa_gauntlet.subprocess, "run", fake_run)

    assert qa_gauntlet.main() == 7
    assert calls == [(("just", "baseline"), False)]
    assert capsys.readouterr().err == "QA gauntlet stopped at baseline (exit 7)\n"


def test_main_reports_an_unstartable_command(monkeypatch, capsys) -> None:
    def raise_os_error(
        command: Sequence[str], *, check: bool
    ) -> subprocess.CompletedProcess[Sequence[str]]:
        raise OSError("just is unavailable")

    monkeypatch.setattr(qa_gauntlet.subprocess, "run", raise_os_error)

    assert qa_gauntlet.main() == 127
    assert capsys.readouterr().err == (
        "Unable to start just baseline: just is unavailable\n"
        "QA gauntlet stopped at baseline (exit 127)\n"
    )


def test_gauntlet_stops_at_first_failed_stage(capsys) -> None:
    calls: list[tuple[str, ...]] = []

    # A failing command is identified by its stage name, not a shell string.
    def failing_runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 9 if tuple(command) == ("just", "crap-report") else 0

    assert run_gauntlet(failing_runner) == 9
    assert calls == [
        ("just", "baseline"),
        ("just", "ruff"),
        ("just", "ty"),
        ("just", "tests"),
        ("just", "property-tests"),
        ("just", "acceptance-tests"),
        ("just", "architecture-checks"),
        ("just", "crap-report"),
    ]
    output = capsys.readouterr()
    assert output.out == (
        "QA stage 1/11: baseline\n"
        "QA stage 2/11: ruff\n"
        "QA stage 3/11: ty\n"
        "QA stage 4/11: tests\n"
        "QA stage 5/11: property tests\n"
        "QA stage 6/11: acceptance tests\n"
        "QA stage 7/11: architecture checks\n"
        "QA stage 8/11: CRAP\n"
    )
    assert output.err == "QA gauntlet stopped at CRAP (exit 9)\n"


def test_progress_is_flushed_before_each_stage_runner_starts() -> None:
    raw = BytesIO()
    stream = TextIOWrapper(BufferedWriter(raw), encoding="utf-8")
    observed_before_runner: list[str] = []
    progress_lines = [
        "QA stage 1/11: baseline\n",
        "QA stage 2/11: ruff\n",
        "QA stage 3/11: ty\n",
        "QA stage 4/11: tests\n",
        "QA stage 5/11: property tests\n",
        "QA stage 6/11: acceptance tests\n",
        "QA stage 7/11: architecture checks\n",
        "QA stage 8/11: CRAP\n",
        "QA stage 9/11: mutation tests\n",
        "QA stage 10/11: smoke test\n",
        "QA stage 11/11: diff review\n",
    ]

    def runner(_command: Sequence[str]) -> int:
        observed_before_runner.append(raw.getvalue().decode("utf-8"))
        return 0

    try:
        with redirect_stdout(stream):
            assert run_gauntlet(runner) == 0
    finally:
        stream.flush()

    assert observed_before_runner == [
        "".join(progress_lines[:index]) for index in range(1, len(progress_lines) + 1)
    ]


def test_success_message_is_flushed_before_run_returns() -> None:
    raw = BytesIO()
    stream = TextIOWrapper(BufferedWriter(raw), encoding="utf-8")

    try:
        with redirect_stdout(stream):
            assert run_gauntlet(lambda _command: 0) == 0
            output_before_cleanup = raw.getvalue().decode("utf-8")
    finally:
        stream.flush()

    assert output_before_cleanup.endswith("QA gauntlet passed: all stages completed\n")


def test_failure_message_is_flushed_before_run_returns() -> None:
    raw = BytesIO()
    stream = TextIOWrapper(BufferedWriter(raw), encoding="utf-8")

    try:
        with redirect_stderr(stream):
            assert run_gauntlet(lambda _command: 9) == 9
            error_before_cleanup = raw.getvalue().decode("utf-8")
    finally:
        stream.flush()

    assert error_before_cleanup == "QA gauntlet stopped at baseline (exit 9)\n"
