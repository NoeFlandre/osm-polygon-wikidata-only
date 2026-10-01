"""Workload benchmarks over the repository's expensive public paths.

Inputs are synthetic and seeded (see ``_workloads``). Run with ``just bench``;
a plain ``pytest`` run does not collect this directory. Memory-sensitive cases
also assert an Arrow memory-pool peak so a memory regression fails even when the
elapsed time barely moves.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import _workloads
from osm_polygon_wikidata_only.hf.coverage_map import generate_coverage_map
from osm_polygon_wikidata_only.hf.language_splits import partition_row_indices
from osm_polygon_wikidata_only.pipeline.containment_migration import (
    StagedRule,
    _canonical_manifest_stats,
)
from osm_polygon_wikidata_only.pipeline.link_migration import (
    apply_link_migration,
    plan_link_migration,
)
from scripts import benchmark_containment_projection as projection

pytest.importorskip("pytest_benchmark")

matplotlib.use("Agg")

_MIB = 1024 * 1024
_ROUNDS = 3


def _arrow_peak_mib(operation: str) -> float:
    """Run ``operation`` in a fresh interpreter and return Arrow's peak pool MiB.

    ``tracemalloc`` only sees Python allocations, not Arrow's native buffers, so
    the peak comes from the Arrow memory pool of an isolated process.
    """
    script = "import pyarrow as pa\n" + operation + "\nprint(pa.default_memory_pool().max_memory())"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return int(completed.stdout.strip().splitlines()[-1]) / _MIB


def test_coverage_map_with_land(benchmark, tmp_path: Path) -> None:
    lons, lats = _workloads.random_points(10_000)
    output = tmp_path / "coverage.png"

    def render() -> Path:
        return generate_coverage_map(
            lons, lats, output, land_geojson_path=_workloads.land_geojson()
        )

    result = benchmark.pedantic(render, rounds=_ROUNDS, iterations=1)

    assert result.stat().st_size > 0


def _staged_containment_inputs(directory: Path, rows: int) -> StagedRule:
    polygons = directory / "polygons.parquet"
    documents = directory / "documents.parquet"
    projection._write_synthetic_parquet(polygons, rows)
    pq.write_table(pa.table({"language": ["en"], "article_length_chars": [42]}), documents)
    return StagedRule(
        "benchmark-latest",
        (),
        (("polygons", polygons), ("wikipedia/documents", documents)),
    )


@pytest.fixture(scope="module")
def containment_inputs(tmp_path_factory: pytest.TempPathFactory) -> StagedRule:
    return _staged_containment_inputs(tmp_path_factory.mktemp("containment"), 200_000)


def test_containment_manifest_stats_200k_polygons(
    benchmark, containment_inputs: StagedRule
) -> None:
    stats = benchmark.pedantic(
        _canonical_manifest_stats, args=(containment_inputs,), rounds=_ROUNDS, iterations=1
    )

    assert stats["language_count"] == 1


def test_containment_manifest_stats_peak_memory() -> None:
    peak = _arrow_peak_mib(
        """
import tempfile
from pathlib import Path
from benchmarks.test_workloads import _staged_containment_inputs
from osm_polygon_wikidata_only.pipeline.containment_migration import _canonical_manifest_stats
with tempfile.TemporaryDirectory() as raw:
    staged = _staged_containment_inputs(Path(raw), 200_000)
    _canonical_manifest_stats(staged)
"""
    )

    assert peak < 50


def _partition_all(batches: list[pa.RecordBatch]) -> int:
    partitions = 0
    for batch in batches:
        grouped = partition_row_indices(batch, batch.schema.get_field_index("language"))
        partitions += len(grouped)
        for indices in grouped.values():
            batch.take(indices)
    return partitions


@pytest.fixture(scope="module")
def language_batches() -> list[pa.RecordBatch]:
    return list(_workloads.language_batches(1_000_000, 50, 65_536))


def test_language_partitioning_1m_rows_50_languages(
    benchmark, language_batches: list[pa.RecordBatch]
) -> None:
    partitions = benchmark.pedantic(
        _partition_all, args=(language_batches,), rounds=_ROUNDS, iterations=1
    )

    assert partitions == 50 * len(language_batches)


def test_language_partitioning_peak_memory() -> None:
    peak = _arrow_peak_mib(
        """
from benchmarks import _workloads
from benchmarks.test_workloads import _partition_all
_partition_all(list(_workloads.language_batches(1_000_000, 50, 65_536)))
"""
    )

    assert peak < 40


_LINKS = 100_000


@pytest.fixture(scope="module")
def legacy_links_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    processed = tmp_path_factory.mktemp("links")
    _workloads.write_legacy_link_stem(processed, "bench-latest", _LINKS)
    return processed


def test_link_migration_plan_100k_links(benchmark, legacy_links_template: Path) -> None:
    plan = benchmark.pedantic(
        plan_link_migration, args=(legacy_links_template,), rounds=_ROUNDS, iterations=1
    )

    assert plan.stems
    assert plan.is_safe_to_apply


def test_link_migration_apply_100k_links(
    benchmark, tmp_path_factory: pytest.TempPathFactory
) -> None:
    def prepare() -> tuple[tuple[Path], dict[str, object]]:
        processed = tmp_path_factory.mktemp("apply") / "processed"
        _workloads.write_legacy_link_stem(processed, "bench-latest", _LINKS)
        return (processed,), {}

    benchmark.pedantic(apply_link_migration, setup=prepare, rounds=_ROUNDS, iterations=1)


def test_cli_version_cold_start(benchmark) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; from osm_polygon_wikidata_only.cli.app import run; sys.argv[1:] = ['--version']; run()",
    ]

    def start() -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603

    completed = benchmark.pedantic(start, rounds=5, iterations=1)

    assert completed.stdout.strip()
