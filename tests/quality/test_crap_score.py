"""Tests for the function-level CRAP quality report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.quality.crap_score import (
    CrapEntry,
    crap_score,
    entries_from_reports,
    evaluate_threshold,
    main,
)


def _file(**data: object) -> dict[str, object]:
    return {"files": {"module.py": {"functions": {}, **data}}}


def _function(**block: object) -> dict[str, object]:
    return {"module.py": [{"type": "function", "name": "f", "complexity": 1, **block}]}


def test_crap_score_uses_standard_formula() -> None:
    assert crap_score(10, 0.75) == 11.5625
    assert crap_score(3, 1.0) == 3


@pytest.mark.parametrize(
    ("complexity", "coverage", "message"),
    [
        (0, 0.5, "complexity"),
        (True, 0.5, "complexity"),
        ("5", 0.5, "complexity"),
        (5, 1.1, "coverage"),
        (5, -0.1, "coverage"),
        (5, True, "coverage"),
        (5, "0.5", "coverage"),
        (5, float("nan"), "coverage"),
        (5, float("inf"), "coverage"),
    ],
)
def test_crap_score_rejects_invalid_inputs(
    complexity: object, coverage: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        crap_score(complexity, coverage)  # type: ignore[arg-type]


def test_evaluate_threshold_fails_scores_at_or_above_the_limit_worst_first() -> None:
    safe = CrapEntry("module.py", "safe", 5, 1.0)
    boundary = CrapEntry("module.py", "boundary", 6, 1.0)
    risky = CrapEntry("module.py", "risky", 10, 0.0)

    assert evaluate_threshold([safe, boundary, risky], maximum=6.0) == [risky, boundary]
    for maximum in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="maximum"):
            evaluate_threshold([], maximum=maximum)


def test_entries_from_reports_joins_function_coverage_and_qualifies_methods() -> None:
    entries = entries_from_reports(
        {
            "files": {
                "/tmp/src/module.py": {
                    "functions": {
                        "parse": {"summary": {"percent_statements_covered": 75.0}},
                        "Parser.parse": {"summary": {"percent_covered": 100}},
                    }
                }
            }
        },
        {
            "src/module.py": [
                {"type": "class", "name": "Parser", "complexity": 4, "lineno": 1},
                {"type": "function", "name": "parse", "complexity": 10, "lineno": 4},
                {"type": "method", "name": "parse", "classname": "Parser", "complexity": 2},
            ]
        },
    )

    assert entries == [
        CrapEntry("src/module.py", "parse", 10, 0.75, 4),
        CrapEntry("src/module.py", "Parser.parse", 2, 1.0, 0),
    ]


def test_entries_from_reports_falls_back_to_executable_line_coverage() -> None:
    """Helpers absent from coverage's function map use their executable lines."""
    coverage = _file(executed_lines=[10, 11], missing_lines=[12])

    assert entries_from_reports(coverage, _function(lineno=10, endline=13)) == [
        CrapEntry("module.py", "f", 1, 2 / 3, 10)
    ]
    assert entries_from_reports(coverage, _function(lineno=20)) == [
        CrapEntry("module.py", "f", 1, 0.0, 20)
    ]


@pytest.mark.parametrize(
    ("coverage", "complexity", "message"),
    [
        ({"files": {}}, {}, "no function entries"),
        ({"files": {}}, _function(), r"coverage file module\.py"),
        (
            {"files": {"/a/module.py": {"functions": {}}, "/b/module.py": {"functions": {}}}},
            _function(),
            r"coverage file module\.py",
        ),
        (_file(), {"module.py": None}, "invalid file entry"),
        (_file(), {7: []}, "invalid file entry"),
        (_file(), {"module.py": [None]}, "must be an object"),
        (_file(), _function(name=None), "entry is malformed"),
        (_file(), _function(complexity=True), "entry is malformed"),
        (_file(), _function(classname=3), "classname is malformed"),
        (_file(), _function(lineno=True), "line is malformed"),
        (_file(), _function(lineno=4, endline=3), "endline is malformed"),
        (_file(executed_lines="1"), _function(), "executed_lines"),
        (_file(missing_lines=[1, False]), _function(), "missing_lines"),
        (
            _file(functions={"f": {"summary": {"percent_covered": "100"}}}),
            _function(),
            "no percentage",
        ),
    ],
)
def test_entries_from_reports_fails_closed_on_malformed_reports(
    coverage: dict[str, object], complexity: dict[object, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        entries_from_reports(coverage, complexity)  # type: ignore[arg-type]


def test_crap_cli_reports_pass_failure_and_unreadable_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    coverage_path = tmp_path / "coverage.json"
    complexity_path = tmp_path / "complexity.json"
    coverage_path.write_text(
        json.dumps(_file(functions={"f": {"summary": {"percent_covered": 100}}})),
        encoding="utf-8",
    )
    complexity_path.write_text(json.dumps(_function(lineno=4)), encoding="utf-8")
    arguments = ["--coverage", str(coverage_path), "--complexity", str(complexity_path)]

    assert main([*arguments, "--maximum", "1.0"]) == 1
    assert "CRAP threshold exceeded" in capsys.readouterr().out
    assert main([*arguments, "--maximum", "2.0"]) == 0
    assert "CRAP threshold passed" in capsys.readouterr().out
    complexity_path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read JSON report"):
        main(arguments)
