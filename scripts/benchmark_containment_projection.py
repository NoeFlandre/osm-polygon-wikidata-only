"""Compare full-width and projected polygon manifest-stat reads at bounded scale.

Run from the issue worktree with:

    python scripts/benchmark_containment_projection.py

The benchmark writes a synthetic 31-column Parquet file under ``TMPDIR``, then
removes it on exit. The projected worker measures peak traced Python memory,
matching the issue's acceptance measurement, and reports process RSS separately.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.pipeline.containment_migration import (
    _POLYGON_MANIFEST_COLUMNS,
    StagedRule,
    _canonical_manifest_stats,
    _polygon_manifest_stats,
)

_DEFAULT_ROWS = 200_000
_MAX_ROWS = 200_000
_BATCH_ROWS = 10_000
_UNUSED_COLUMNS = 26
_TAG_KEYS = '["amenity", "name", "wikidata"]'
_GEOMETRY_COORDINATES = ", ".join(f"{point * 37} {point * 19}" for point in range(1, 45))
_GEOMETRY_PREFIX = f"POLYGON ((0 0, {_GEOMETRY_COORDINATES}, "


def _manifest_values(start: int, stop: int) -> dict[str, pa.Array]:
    indexes = range(start, stop)
    return {
        "wikidata": pa.array([f"Q{index}" for index in indexes], type=pa.string()),
        "has_wikipedia": pa.array([index % 3 != 0 for index in indexes], type=pa.bool_()),
        "text_available": pa.array([index % 5 != 0 for index in indexes], type=pa.bool_()),
        "area_bucket": pa.array(
            [("tiny", "small", "large", "huge")[index % 4] for index in indexes],
            type=pa.string(),
        ),
        "tag_keys": pa.array([_TAG_KEYS] * (stop - start), type=pa.string()),
    }


def _geometry_values(start: int, stop: int) -> pa.Array:
    return pa.array(
        [f"{_GEOMETRY_PREFIX}{index} 0, 0 0))" for index in range(start, stop)],
        type=pa.string(),
    )


def _unused_values(start: int, stop: int) -> dict[str, pa.Array]:
    return {
        f"unused_{column:02}": pa.array(
            [index % 16 for index in range(start, stop)],
            type=pa.int64(),
        )
        for column in range(_UNUSED_COLUMNS - 1)
    }


def _synthetic_batch(start: int, stop: int) -> pa.Table:
    values = _manifest_values(start, stop)
    values["geometry"] = _geometry_values(start, stop)
    return pa.table(values | _unused_values(start, stop))


def _write_synthetic_parquet(path: Path, row_count: int) -> None:
    first = _synthetic_batch(0, min(row_count, _BATCH_ROWS))
    with pq.ParquetWriter(path, first.schema, compression="snappy") as writer:
        writer.write_table(first, row_group_size=_BATCH_ROWS)
        for start in range(_BATCH_ROWS, row_count, _BATCH_ROWS):
            stop = min(start + _BATCH_ROWS, row_count)
            writer.write_table(_synthetic_batch(start, stop), row_group_size=_BATCH_ROWS)


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _worker_stats(mode: str, parquet_path: Path, documents_path: Path) -> dict[str, Any]:
    if mode == "all-columns":
        polygons = pq.read_table(parquet_path).to_pylist()
        return _polygon_manifest_stats(polygons)
    staged = StagedRule(
        "benchmark-latest",
        (),
        (
            ("polygons", parquet_path),
            ("wikipedia/documents", documents_path),
        ),
    )
    return _canonical_manifest_stats(staged)


def _run_worker(mode: str, parquet_path: Path, documents_path: Path) -> None:
    baseline_peak_rss_mib = _peak_rss_mib()
    if mode == "projected":
        tracemalloc.start()
    started = time.perf_counter()
    stats = _worker_stats(mode, parquet_path, documents_path)
    elapsed_seconds = time.perf_counter() - started
    tracemalloc_peak_mib = None
    if mode == "projected":
        _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc_peak_mib = tracemalloc_peak_bytes / (1024 * 1024)
    peak_rss_mib = _peak_rss_mib()
    print(
        json.dumps(
            {
                "mode": mode,
                "elapsed_seconds": elapsed_seconds,
                "baseline_peak_rss_mib": baseline_peak_rss_mib,
                "peak_rss_mib": peak_rss_mib,
                "additional_peak_rss_mib": max(0.0, peak_rss_mib - baseline_peak_rss_mib),
                "tracemalloc_peak_mib": tracemalloc_peak_mib,
                "stats": stats,
            },
            sort_keys=True,
        )
    )


def _measure(mode: str, parquet_path: Path, documents_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(  # noqa: S603 -- executable and argv are controlled by this script.
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-mode",
            mode,
            "--parquet",
            str(parquet_path),
            "--documents",
            str(documents_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument(
        "--worker-mode", choices=("all-columns", "projected"), help=argparse.SUPPRESS
    )
    parser.add_argument("--parquet", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--documents", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_worker_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.parquet is None or args.documents is None:
        parser.error("--parquet and --documents are required in worker mode")


def _validate_benchmark_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.parquet is not None or args.documents is not None:
        parser.error("--parquet and --documents are only valid in worker mode")
    if not 1 <= args.rows <= _MAX_ROWS:
        parser.error(f"--rows must be between 1 and {_MAX_ROWS}")


def _parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(arguments)
    if args.worker_mode is None:
        _validate_benchmark_arguments(args, parser)
    else:
        _validate_worker_arguments(args, parser)
    return args


def _assert_stats_match(all_columns: dict[str, Any], projected: dict[str, Any]) -> None:
    projected_polygon_stats = {key: projected["stats"][key] for key in all_columns["stats"]}
    if all_columns["stats"] != projected_polygon_stats:
        raise RuntimeError("Projected manifest statistics differ from all-column statistics")


def _assert_performance_target(projected: dict[str, Any]) -> None:
    if projected["elapsed_seconds"] > 3.5 or projected["tracemalloc_peak_mib"] > 100:
        raise RuntimeError(
            "Projected manifest stats exceeded the 3.5-second / 100-MiB acceptance target"
        )


def _measurement_metrics(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        key: measurement[key]
        for key in (
            "elapsed_seconds",
            "baseline_peak_rss_mib",
            "peak_rss_mib",
            "additional_peak_rss_mib",
            "tracemalloc_peak_mib",
        )
    }


def _benchmark_summary(
    row_count: int,
    parquet_path: Path,
    all_columns: dict[str, Any],
    projected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rows": row_count,
        "columns": len(_POLYGON_MANIFEST_COLUMNS) + _UNUSED_COLUMNS,
        "parquet_mib": parquet_path.stat().st_size / (1024 * 1024),
        "stats_equal": True,
        "all_columns": _measurement_metrics(all_columns),
        "projected": _measurement_metrics(projected),
    }


def _run_benchmark(row_count: int) -> None:
    with tempfile.TemporaryDirectory(prefix="containment-projection-") as raw_tmp:
        parquet_path = Path(raw_tmp) / "polygons.parquet"
        documents_path = Path(raw_tmp) / "documents.parquet"
        _write_synthetic_parquet(parquet_path, row_count)
        pq.write_table(
            pa.table({"language": ["en"], "article_length_chars": [42]}),
            documents_path,
        )
        all_columns = _measure("all-columns", parquet_path, documents_path)
        projected = _measure("projected", parquet_path, documents_path)
        _assert_stats_match(all_columns, projected)
        _assert_performance_target(projected)
        print(
            json.dumps(
                _benchmark_summary(row_count, parquet_path, all_columns, projected),
                indent=2,
                sort_keys=True,
            )
        )


def main(arguments: list[str] | None = None) -> None:
    args = _parse_arguments(arguments)
    if args.worker_mode is not None:
        _run_worker(args.worker_mode, args.parquet, args.documents)
        return
    _run_benchmark(args.rows)


if __name__ == "__main__":
    main()
