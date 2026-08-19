"""Tests for the function-level CRAP quality report."""

from __future__ import annotations

import pytest

from scripts.quality.crap_score import (
    CrapEntry,
    crap_score,
    entries_from_reports,
    evaluate_threshold,
)


def test_crap_score_uses_standard_formula() -> None:
    assert crap_score(10, 0.75) == 11.5625


def test_crap_score_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="complexity"):
        crap_score(0, 0.5)
    with pytest.raises(ValueError, match="coverage"):
        crap_score(5, 1.1)


def test_evaluate_threshold_returns_worst_functions() -> None:
    entries = [
        CrapEntry("module.py", "safe", 2, 1.0),
        CrapEntry("module.py", "risky", 10, 0.0),
    ]

    assert [entry.name for entry in evaluate_threshold(entries, maximum=30.0)] == ["risky"]


def test_evaluate_threshold_rejects_exact_limit() -> None:
    entry = CrapEntry("module.py", "boundary", 6, 1.0)

    assert evaluate_threshold([entry], maximum=6.0) == [entry]


def test_evaluate_threshold_accepts_scores_below_limit() -> None:
    entry = CrapEntry("module.py", "safe", 5, 1.0)

    assert evaluate_threshold([entry], maximum=6.0) == []


def test_entries_from_reports_joins_radon_with_coverage_summary() -> None:
    entries = entries_from_reports(
        {
            "files": {
                "module.py": {
                    "functions": {"parse": {"summary": {"percent_statements_covered": 75.0}}}
                }
            }
        },
        {"module.py": [{"type": "function", "name": "parse", "complexity": 10, "lineno": 4}]},
    )

    assert entries == [CrapEntry("module.py", "parse", 10, 0.75, 4)]
