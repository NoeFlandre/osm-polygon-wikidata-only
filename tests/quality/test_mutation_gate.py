"""Tests for the zero-survivor mutation gate."""

from __future__ import annotations

import io
import sys

import pytest

from scripts.quality.mutation_gate import (
    MutationGateError,
    _non_killed,
    ensure_all_killed,
    main,
    parse_results,
)


def test_parse_results_reads_mutmut_status_lines() -> None:
    report = """
    Mutant results
    --------------
        module.py:1:replace: killed
        module.py:2:replace: survived
    """

    assert parse_results(report) == [
        ("module.py:1:replace", "killed"),
        ("module.py:2:replace", "survived"),
    ]


def test_ensure_all_killed_accepts_only_killed_mutants() -> None:
    ensure_all_killed([("one", "killed"), ("two", "killed")])


def test_ensure_all_killed_reports_non_killed_mutants() -> None:
    with pytest.raises(MutationGateError, match="two: survived"):
        ensure_all_killed([("one", "killed"), ("two", "survived")])


def test_ensure_all_killed_rejects_empty_reports() -> None:
    with pytest.raises(MutationGateError, match="No mutants"):
        ensure_all_killed([])


def test_non_killed_extracts_only_actionable_mutants() -> None:
    assert _non_killed([("one", "killed"), ("two", "survived")]) == [("two", "survived")]


def test_mutation_gate_main_accepts_a_killed_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("one: killed\n"))

    assert main() == 0
    assert "1 mutants killed" in capsys.readouterr().out
