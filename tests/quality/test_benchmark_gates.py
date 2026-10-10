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


def test_missing_lists_baseline_benchmarks_absent_from_the_run() -> None:
    assert bench_compare.missing({"a": 1.0, "b": 1.0}, {"a": 1.0, "new": 1.0}) == ["b"]


@pytest.mark.parametrize(("mode", "expected"), [("warn", 0), ("enforce", 1)])
def test_compare_treats_a_dropped_benchmark_as_a_failure_only_in_enforce_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, expected: int
) -> None:
    baseline = _bench(tmp_path / "base.json", {"a": 1.0, "dropped": 1.0})
    current = _bench(tmp_path / "now.json", {"a": 1.0})

    status = bench_compare.main(
        ["--baseline", str(baseline), "--current", str(current), "--mode", mode]
    )

    assert status == expected
    assert "MISSING dropped" in capsys.readouterr().out


def test_slow_tests_defaults_missing_attributes_to_zero_seconds_and_an_empty_name(
    tmp_path: Path,
) -> None:
    report = tmp_path / "junit.xml"
    report.write_text("<testsuite><testcase/></testsuite>", encoding="utf-8")

    assert slow_tests.test_durations(report) == [(0.0, "::")]


def test_slow_tests_prints_each_duration_with_its_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _junit(tmp_path / "junit.xml", {"a": 1.5})

    slow_tests.main(["--junit-xml", str(report), "--overrides", str(tmp_path / "none")])

    assert capsys.readouterr().out == "   1.50s  tests.mod::a\n"


def test_slow_tests_parser_types_and_defaults() -> None:
    parser = slow_tests.build_parser()

    args = parser.parse_args(["--junit-xml", "j.xml", "--budget", "5"])

    help_text = " ".join(parser.format_help().split())
    assert "Fail when any single test exceeds the per-test time budget" in help_text
    assert args.junit_xml == Path("j.xml")
    assert args.budget == 5.0
    assert isinstance(args.budget, float)
    assert parser.parse_args(["--junit-xml", "j.xml"]).budget == slow_tests.DEFAULT_BUDGET_SECONDS
    assert parser.parse_args(["--junit-xml", "j.xml"]).overrides == slow_tests.DEFAULT_OVERRIDES
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_slow_tests_reads_a_utf8_overrides_file(tmp_path: Path) -> None:
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps({"tests.mod::é": 8.0}, ensure_ascii=False), encoding="utf-8")

    assert slow_tests._load_overrides(path) == {"tests.mod::é": 8.0}


def test_compare_parser_types_choices_and_defaults() -> None:
    parser = bench_compare.build_parser()

    args = parser.parse_args(["--baseline", "b.json", "--current", "c.json", "--threshold", "0.5"])

    help_text = " ".join(parser.format_help().split())
    assert "Compare a pytest-benchmark JSON run with a committed baseline" in help_text
    assert args.baseline == Path("b.json")
    assert args.current == Path("c.json")
    assert args.threshold == 0.5
    assert isinstance(args.threshold, float)
    assert args.mode == "warn"
    defaults = parser.parse_args(["--baseline", "b.json", "--current", "c.json"])
    assert defaults.threshold == bench_compare.DEFAULT_THRESHOLD
    for invalid in (["--current", "c.json"], ["--baseline", "b.json"]):
        with pytest.raises(SystemExit):
            parser.parse_args(invalid)
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline", "b", "--current", "c", "--mode", "WARN"])


def test_compare_measures_slowdown_as_a_ratio_of_the_baseline() -> None:
    found = bench_compare.regressions({"a": 2.0}, {"a": 3.0}, 0.25)

    assert found == [("a", pytest.approx(0.5))]


def test_medians_error_message_is_exact() -> None:
    with pytest.raises(ValueError, match=r"^benchmark report has no 'benchmarks' list$"):
        bench_compare.medians({})


def test_compare_prints_one_finding_per_line_and_no_all_clear_after_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _bench(tmp_path / "base.json", {"a": 1.0, "gone": 1.0})
    current = _bench(tmp_path / "now.json", {"a": 2.0, "new": 1.0})

    bench_compare.main(["--baseline", str(baseline), "--current", str(current)])

    assert capsys.readouterr().out.splitlines() == [
        "NEW new",
        "MISSING gone: in the baseline but not in this run",
        "REGRESSION a: median +100% vs baseline",
    ]


def test_compare_reads_a_utf8_report(tmp_path: Path) -> None:
    path = tmp_path / "now.json"
    entries = [{"fullname": "é", "stats": {"median": 1.0}}]
    path.write_text(json.dumps({"benchmarks": entries}, ensure_ascii=False), encoding="utf-8")

    assert bench_compare._load(path) == {"é": 1.0}


def test_a_small_slowdown_of_a_slow_benchmark_is_not_a_regression() -> None:
    assert bench_compare.regressions({"a": 2.0}, {"a": 2.2}, 0.25) == []
