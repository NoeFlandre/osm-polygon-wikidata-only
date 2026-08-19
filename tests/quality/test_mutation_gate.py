"""Tests for the zero-survivor mutation gate."""

from __future__ import annotations

import pytest

from scripts.quality.mutation_gate import (
    MutationGateError,
    ensure_all_killed,
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
