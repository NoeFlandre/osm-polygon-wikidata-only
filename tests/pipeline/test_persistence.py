"""Persistence-phase integrity enforcement over the committed Monaco fixture."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.integrity import INTEGRITY_CONTRACT_VERSION
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.models import Article, Polygon, PolygonArticleLink
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.persistence import run_persistence_phase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"


def _rows(table: str) -> list[dict]:
    return pq.read_table(FIXTURE / table / f"{STEM}.parquet").to_pylist()


def test_persistence_drops_mismatched_links_and_records_integrity_in_manifest(
    tmp_path: Path,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygons = [Polygon(**row) for row in _rows("polygons")]
    articles = [Article(**row) for row in _rows("articles")]
    links = [PolygonArticleLink(**row) for row in _rows("polygon_articles")]
    stale = dataclasses.replace(links[0], wikidata="Q999999999")

    outcome = run_persistence_phase(
        polygons,
        articles,
        [stale, *links[1:]],
        data_root=data_root,
        stem=STEM,
        source_pbf=f"{STEM}.osm.pbf",
    )

    assert outcome.integrity_result is not None
    assert outcome.integrity_result.rejected_row_count == 1
    assert pq.read_table(outcome.links_path).num_rows == len(links) - 1
    integrity = load_manifest(outcome.manifest_path)[f"{STEM}.osm.pbf"]["integrity"]
    assert integrity["contract_version"] == INTEGRITY_CONTRACT_VERSION
    assert integrity["rejected_row_count"] == 1
    assert integrity["rejections"][0]["wikidata"] == "Q999999999"
    assert outcome.manifest_entry["integrity"] == integrity
