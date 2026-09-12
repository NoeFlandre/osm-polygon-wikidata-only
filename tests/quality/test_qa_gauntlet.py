"""Contracts for the deterministic, fail-fast QA gauntlet."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from io import BufferedWriter, BytesIO, TextIOWrapper

import scripts.quality.qa_gauntlet as qa_gauntlet
from scripts.quality.qa_gauntlet import build_stages, run_gauntlet


def test_gauntlet_stages_are_in_the_required_order() -> None:
    assert [stage.name for stage in build_stages()] == [
        "baseline",
        "ruff",
        "ty",
        "tests",
        "property tests",
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
        ("just", "property-tests"),
        ("just", "acceptance-tests"),
        ("just", "architecture-checks"),
        ("just", "crap-report"),
        ("just", "mutation"),
        ("just", "smoke-test"),
        ("just", "diff-review"),
    ]


def test_run_command_returns_exit_status_without_shell(monkeypatch) -> None:
    calls: list[tuple[Sequence[str], bool]] = []

    def fake_run(
        command: Sequence[str], *, check: bool
    ) -> subprocess.CompletedProcess[Sequence[str]]:
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(qa_gauntlet.subprocess, "run", fake_run)

    assert qa_gauntlet._run_command(("just", "ruff")) == 7
    assert calls == [(("just", "ruff"), False)]


def test_run_command_reports_an_unstartable_command(monkeypatch, capsys) -> None:
    def raise_os_error(
        command: Sequence[str], *, check: bool
    ) -> subprocess.CompletedProcess[Sequence[str]]:
        raise OSError("just is unavailable")

    monkeypatch.setattr(qa_gauntlet.subprocess, "run", raise_os_error)

    assert qa_gauntlet._run_command(("just", "baseline")) == 127
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Unable to start just baseline: just is unavailable\n"


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


def test_gauntlet_returns_zero_when_all_stages_pass(capsys) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(tuple(command))
        return 0

    assert run_gauntlet(runner) == 0
    assert len(calls) == len(build_stages())
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
        "QA stage 9/11: mutation tests\n"
        "QA stage 10/11: smoke test\n"
        "QA stage 11/11: diff review\n"
        "QA gauntlet passed: all stages completed\n"
    )
    assert output.err == ""
