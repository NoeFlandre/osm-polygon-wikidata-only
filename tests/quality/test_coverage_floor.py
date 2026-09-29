"""Contracts for the per-file coverage floor check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.quality.coverage_floor import FileCoverage, below_floor, file_coverages, main


def _report(**files: tuple[int, int]) -> dict[str, object]:
    return {
        "files": {
            path.replace("__", "/"): {
                "summary": {"num_statements": statements, "covered_lines": covered}
            }
            for path, (statements, covered) in files.items()
        }
    }


def test_file_coverages_uses_line_counts_and_treats_empty_files_as_covered() -> None:
    assert file_coverages(_report(b__x_py=(10, 9), a__empty_py=(0, 0))) == [
        FileCoverage("a/empty_py", 100.0, 0),
        FileCoverage("b/x_py", 90.0, 10),
    ]


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"files": {"a.py": {}}},
        {"files": {"a.py": {"summary": {"num_statements": True, "covered_lines": 1}}}},
        {"files": {"a.py": {"summary": {"num_statements": 3}}}},
    ],
)
def test_malformed_reports_are_rejected(report: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        file_coverages(report)


def test_below_floor_is_strict_and_honours_exemptions() -> None:
    items = [
        FileCoverage("src/pkg/ok.py", 85.0, 20),
        FileCoverage("src/pkg/low.py", 84.9, 20),
        FileCoverage("scripts/subprocess_only.py", 0.0, 4),
    ]
    assert below_floor(items, minimum=85.0, exempt=["scripts/subprocess_only.py"]) == [items[1]]
    assert below_floor(items, minimum=0.0) == []


@pytest.mark.parametrize("minimum", [-1.0, 100.1, float("nan")])
def test_invalid_minimum_is_rejected(minimum: float) -> None:
    with pytest.raises(ValueError, match="minimum"):
        below_floor([], minimum=minimum)


def test_main_reports_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report(a__low_py=(10, 5), a__ok_py=(10, 10))), encoding="utf-8")

    assert main(["--coverage", str(path), "--minimum", "90"]) == 1
    captured = capsys.readouterr()
    assert "a/low_py" in captured.err
    assert "a/ok_py" not in captured.err

    assert main(["--coverage", str(path), "--minimum", "90", "--exempt", "a/low_py"]) == 0
    assert "floor passed" in capsys.readouterr().out
