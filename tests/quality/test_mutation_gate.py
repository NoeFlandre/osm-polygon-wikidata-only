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


def test_reviewed_survivor_is_not_confused_with_a_killed_mutant() -> None:
    results = [("one", "killed"), ("equivalent", "survived")]
    ensure_all_killed(results, equivalents=frozenset({"equivalent"}))
    assert results[1] == ("equivalent", "survived")


@pytest.mark.parametrize(
    "status", ["timeout", "no tests", "not checked", "suspicious", "skipped", "segfault"]
)
def test_equivalence_cannot_excuse_an_incomplete_check(status) -> None:
    with pytest.raises(MutationGateError, match=f"equivalent: {status}"):
        ensure_all_killed([("equivalent", status)], equivalents=frozenset({"equivalent"}))


def test_review_does_not_allow_a_different_survivor() -> None:
    with pytest.raises(MutationGateError, match="different: survived"):
        ensure_all_killed(
            [("equivalent", "survived"), ("different", "survived")],
            equivalents=frozenset({"equivalent"}),
        )


def test_non_killed_extracts_only_actionable_mutants() -> None:
    assert _non_killed([("one", "killed"), ("two", "survived")]) == [("two", "survived")]


def test_mutation_gate_main_accepts_a_killed_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("one: killed\n"))

    assert main() == 0
    assert "1 mutants killed" in capsys.readouterr().out
