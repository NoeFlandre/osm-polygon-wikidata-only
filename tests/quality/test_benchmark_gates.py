"""Behavior of the slow-test budget and benchmark regression gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.quality import bench_compare, slow_tests


def _junit(path: Path, cases: dict[str, float]) -> Path:
    body = "".join(
        f'<testcase classname="tests.mod" name="{name}" time="{seconds}"/>'
        for name, seconds in cases.items()
    )
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")
    return path


def test_slow_tests_orders_durations_and_flags_only_tests_over_budget(tmp_path: Path) -> None:
    report = _junit(tmp_path / "junit.xml", {"fast": 0.1, "exact": 3.0, "slow": 3.5})

    durations = slow_tests.test_durations(report)

    assert [name for _, name in durations] == [
        "tests.mod::slow",
        "tests.mod::exact",
        "tests.mod::fast",
    ]
    assert slow_tests.over_budget(durations, 3.0) == [(3.5, "tests.mod::slow")]


@pytest.mark.parametrize(("slowest", "expected"), [(2.9, 0), (3.1, 1)])
def test_slow_tests_exit_status_follows_the_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slowest: float, expected: int
) -> None:
    report = _junit(tmp_path / "junit.xml", {"a": 0.2, "b": slowest})

    assert slow_tests.main(["--junit-xml", str(report)]) == expected
    assert ("OVER BUDGET" in capsys.readouterr().err) is bool(expected)


def _bench(path: Path, medians: dict[str, float]) -> Path:
    entries = [{"fullname": name, "stats": {"median": value}} for name, value in medians.items()]
    path.write_text(json.dumps({"benchmarks": entries}), encoding="utf-8")
    return path


def test_regressions_report_only_medians_beyond_the_threshold() -> None:
    baseline = {"a": 1.0, "b": 1.0, "c": 1.0}
    current = {"a": 1.25, "b": 1.26, "c": 0.5, "new": 9.0}

    found = bench_compare.regressions(baseline, current, 0.25)

    assert [name for name, _ in found] == ["b"]
    assert found[0][1] == pytest.approx(0.26)


def test_medians_reject_reports_without_benchmarks() -> None:
    with pytest.raises(ValueError, match="benchmarks"):
        bench_compare.medians({})


@pytest.mark.parametrize(("mode", "expected"), [("warn", 0), ("enforce", 1)])
def test_compare_fails_only_in_enforce_mode(tmp_path: Path, mode: str, expected: int) -> None:
    baseline = _bench(tmp_path / "base.json", {"a": 1.0})
    current = _bench(tmp_path / "now.json", {"a": 2.0})

    status = bench_compare.main(
        ["--baseline", str(baseline), "--current", str(current), "--mode", mode]
    )

    assert status == expected


def test_compare_without_a_baseline_never_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = _bench(tmp_path / "now.json", {"a": 2.0})

    status = bench_compare.main(
        [
            "--baseline",
            str(tmp_path / "missing.json"),
            "--current",
            str(current),
            "--mode",
            "enforce",
        ]
    )

    assert status == 0
    assert "No baseline" in capsys.readouterr().out


def test_compare_lists_new_benchmarks_and_reports_a_clean_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _bench(tmp_path / "base.json", {"a": 1.0})
    current = _bench(tmp_path / "now.json", {"a": 1.1, "added": 5.0})

    assert bench_compare.main(["--baseline", str(baseline), "--current", str(current)]) == 0

    output = capsys.readouterr().out
    assert "NEW added" in output
    assert "No median regression above 25%" in output


def test_explicit_per_test_budgets_raise_only_the_named_test(tmp_path: Path) -> None:
    durations = [(5.0, "tests.mod::named"), (5.0, "tests.mod::other")]

    found = slow_tests.over_budget(durations, 3.0, {"tests.mod::named": 8.0})

    assert found == [(5.0, "tests.mod::other")]


def test_main_reads_overrides_from_the_given_file(tmp_path: Path) -> None:
    report = _junit(tmp_path / "junit.xml", {"named": 5.0})
    overrides = tmp_path / "budgets.json"
    overrides.write_text(json.dumps({"tests.mod::named": 8.0}), encoding="utf-8")

    assert slow_tests.main(["--junit-xml", str(report), "--overrides", str(overrides)]) == 0
    assert slow_tests.main(["--junit-xml", str(report), "--overrides", str(tmp_path / "none")]) == 1
