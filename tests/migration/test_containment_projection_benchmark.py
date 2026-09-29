"""Tests for the bounded containment projection benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import benchmark_containment_projection as benchmark


def test_synthetic_batch_has_manifest_and_unused_columns() -> None:
    batch = benchmark._synthetic_batch(3, 6)

    assert batch.num_rows == 3
    assert len(batch.column_names) == len(benchmark._POLYGON_MANIFEST_COLUMNS) + 26
    assert batch.column("wikidata").to_pylist() == ["Q3", "Q4", "Q5"]
    assert batch.column("tag_keys").to_pylist() == [benchmark._TAG_KEYS] * 3
    assert batch.column("geometry").to_pylist()[0].startswith("POLYGON ((0 0, ")
    assert batch.column("unused_00").to_pylist() == [3, 4, 5]


def test_write_synthetic_parquet_writes_partial_final_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "_BATCH_ROWS", 2)
    parquet_path = tmp_path / "polygons.parquet"

    benchmark._write_synthetic_parquet(parquet_path, 5)

    parquet_file = pq.ParquetFile(parquet_path)
    assert parquet_file.metadata.num_rows == 5
    assert parquet_file.metadata.num_row_groups == 3
    assert pq.read_table(parquet_path).column("wikidata").to_pylist() == [
        "Q0",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]


@pytest.mark.parametrize(
    ("platform", "expected_mib"),
    [("darwin", 2.0), ("linux", 2048.0)],
)
def test_peak_rss_converts_platform_units(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected_mib: float
) -> None:
    monkeypatch.setattr(benchmark.sys, "platform", platform)
    monkeypatch.setattr(
        benchmark.resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_maxrss=2 * 1024 * 1024),
    )

    assert benchmark._peak_rss_mib() == expected_mib


@pytest.mark.parametrize("mode", ["all-columns", "projected"])
def test_worker_emits_stats_and_memory_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    parquet_path = tmp_path / "polygons.parquet"
    documents_path = tmp_path / "documents.parquet"
    benchmark._write_synthetic_parquet(parquet_path, 4)
    pq.write_table(pa.table({"language": ["en"], "article_length_chars": [42]}), documents_path)
    rss_values = iter((100.0, 105.0))
    monkeypatch.setattr(benchmark, "_peak_rss_mib", lambda: next(rss_values))

    benchmark._run_worker(mode, parquet_path, documents_path)

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == mode
    assert output["stats"]["polygon_count"] == 4
    assert output["baseline_peak_rss_mib"] == 100.0
    assert output["additional_peak_rss_mib"] == 5.0
    if mode == "projected":
        assert output["tracemalloc_peak_mib"] >= 0
    else:
        assert output["tracemalloc_peak_mib"] is None


def test_measure_runs_worker_and_decodes_its_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"mode": "projected", "stats": {"rows": 3}}
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(stdout=json.dumps(expected))

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    result = benchmark._measure(
        "projected", tmp_path / "polygons.parquet", tmp_path / "docs.parquet"
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == benchmark.sys.executable
    assert command[2:4] == ["--worker-mode", "projected"]
    assert captured["check"] is True
    assert captured["capture_output"] is True
    assert captured["text"] is True
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert cast(dict[str, str], environment)["PYTHONDONTWRITEBYTECODE"] == "1"
    assert result == expected


def test_run_benchmark_emits_matching_stats_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    measurements = {
        "stats": {"polygon_count": 4},
        "elapsed_seconds": 1.0,
        "baseline_peak_rss_mib": 100.0,
        "peak_rss_mib": 105.0,
        "additional_peak_rss_mib": 5.0,
        "tracemalloc_peak_mib": 2.0,
    }
    monkeypatch.setattr(benchmark, "_measure", lambda *args: measurements)

    benchmark._run_benchmark(4)

    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 4
    assert summary["stats_equal"] is True
    assert summary["all_columns"]["elapsed_seconds"] >= 0
    assert summary["projected"]["tracemalloc_peak_mib"] >= 0


@pytest.mark.parametrize(
    "measurements, message",
    [
        (
            [
                {"stats": {"rows": 2}, "elapsed_seconds": 1.0, "tracemalloc_peak_mib": 1.0},
                {"stats": {"rows": 3}, "elapsed_seconds": 1.0, "tracemalloc_peak_mib": 1.0},
            ],
            "statistics differ",
        ),
        (
            [
                {"stats": {"rows": 2}, "elapsed_seconds": 1.0, "tracemalloc_peak_mib": 1.0},
                {"stats": {"rows": 2}, "elapsed_seconds": 3.6, "tracemalloc_peak_mib": 1.0},
            ],
            "acceptance target",
        ),
        (
            [
                {"stats": {"rows": 2}, "elapsed_seconds": 1.0, "tracemalloc_peak_mib": 1.0},
                {"stats": {"rows": 2}, "elapsed_seconds": 1.0, "tracemalloc_peak_mib": 101.0},
            ],
            "acceptance target",
        ),
    ],
)
def test_run_benchmark_rejects_mismatch_or_budget_overrun(
    monkeypatch: pytest.MonkeyPatch,
    measurements: list[dict[str, object]],
    message: str,
) -> None:
    pending = iter(measurements)
    monkeypatch.setattr(benchmark, "_measure", lambda *args: next(pending))

    with pytest.raises(RuntimeError, match=message):
        benchmark._run_benchmark(2)


def test_main_accepts_explicit_arguments_and_runs_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[int] = []
    monkeypatch.setattr(benchmark, "_run_benchmark", received.append)

    benchmark.main(["--rows", "7"])

    assert received == [7]


def test_main_dispatches_worker_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[str, Path, Path]] = []
    monkeypatch.setattr(benchmark, "_run_worker", lambda *args: received.append(args))

    benchmark.main(
        [
            "--worker-mode",
            "projected",
            "--parquet",
            "polygons.parquet",
            "--documents",
            "documents.parquet",
        ]
    )

    assert received == [("projected", Path("polygons.parquet"), Path("documents.parquet"))]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--worker-mode", "projected", "--parquet", "polygons.parquet"],
        ["--parquet", "polygons.parquet"],
        ["--rows", "0"],
        ["--rows", "200001"],
    ],
)
def test_argument_parser_rejects_invalid_mode_combinations(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark._parse_arguments(arguments)
