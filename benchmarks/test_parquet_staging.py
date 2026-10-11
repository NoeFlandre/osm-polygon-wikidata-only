"""Timed Parquet staging benchmark for one V2 region.

The timed call is ``_stage_region_tables``: it writes every region table to a
temporary Parquet sibling, validates each written schema, and atomically
replaces the final files. Seeded synthetic rows and fresh output directories
are prepared outside the timed region, so only the staging call is measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import _workloads
from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.v2 import storage as v2_storage
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)

pytest.importorskip("pytest_benchmark")

_ROUNDS = 3
_REGION = "benchmark-latest"
_ROWS = 50_000


def _stage(
    final_paths: tuple[Path, ...],
    rows: tuple[list[dict[str, Any]], ...],
    schemas: tuple[pa.Schema, ...],
) -> tuple[Path, ...]:
    v2_storage._stage_region_tables(final_paths, rows, schemas)
    return final_paths


@pytest.fixture(scope="module")
def region_rows() -> tuple[list[dict[str, Any]], ...]:
    polygons, documents, links = _workloads.v2_region_rows(_ROWS, _REGION)
    # Sections stay empty, matching the default ``write_v2_region`` call.
    return (polygons, documents, [], links)


def test_parquet_staging_v2_region_tables(
    benchmark,
    tmp_path_factory: pytest.TempPathFactory,
    region_rows: tuple[list[dict[str, Any]], ...],
) -> None:
    schemas = (
        polygon_v2_schema(),
        wikipedia_document_v2_schema(),
        section_schema(),
        polygon_document_link_v2_schema(),
    )

    def prepare() -> tuple[tuple[Any, ...], dict[str, object]]:
        processed = tmp_path_factory.mktemp("staging") / "processed_v2"
        return (v2_storage._region_paths(processed, _REGION), region_rows, schemas), {}

    final_paths = benchmark.pedantic(_stage, setup=prepare, rounds=_ROUNDS, iterations=1)

    expected_rows = (_ROWS, _ROWS, 0, _ROWS)
    for path, expected in zip(final_paths, expected_rows, strict=True):
        assert pq.read_metadata(path).num_rows == expected
