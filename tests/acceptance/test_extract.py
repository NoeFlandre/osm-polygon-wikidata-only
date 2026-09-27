"""Step definitions for ``extract.feature`` (#115).

The PBF is written on the fly with osmium, so extraction exercises the real
reader and processor while enrichment uses in-memory clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import osmium
import osmium.osm
import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.wikidata_qids import qids_from_osm_tag
from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
from osm_polygon_wikidata_only.pipeline.processor import process_pbf

scenarios("extract.feature")

_REGION = "acceptance"
_TAGGED_WAY_ID = 100
_UNTAGGED_WAY_ID = 101
_RELATION_ID = 200
_RELATION_OUTER_WAY_ID = 102
_RELATION_SECOND_OUTER_WAY_ID = 103


@dataclass
class _State:
    pbf: Path | None = None
    relation_tag: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)


def _square(writer: Any, first_node_id: int, lon: float, lat: float) -> list[int]:
    corners = [(lon, lat), (lon + 0.01, lat), (lon + 0.01, lat + 0.01), (lon, lat + 0.01)]
    ids = []
    for offset, (x, y) in enumerate(corners):
        node_id = first_node_id + offset
        writer.add_node(osmium.osm.mutable.Node(id=node_id, location=(x, y), tags={}))
        ids.append(node_id)
    return [*ids, ids[0]]


def _write_pbf(path: Path, relation_tag: str) -> None:
    writer = osmium.SimpleWriter(str(path))
    try:
        tagged = _square(writer, 1, 7.40, 43.70)
        untagged = _square(writer, 10, 7.50, 43.70)
        outer = _square(writer, 20, 7.60, 43.70)
        second_outer = _square(writer, 30, 7.70, 43.70)
        writer.add_way(
            osmium.osm.mutable.Way(
                id=_TAGGED_WAY_ID,
                nodes=tagged,
                tags={"wikidata": "Q3", "name": "Tagged park", "leisure": "park"},
            )
        )
        writer.add_way(
            osmium.osm.mutable.Way(id=_UNTAGGED_WAY_ID, nodes=untagged, tags={"leisure": "park"})
        )
        writer.add_way(osmium.osm.mutable.Way(id=_RELATION_OUTER_WAY_ID, nodes=outer, tags={}))
        writer.add_way(
            osmium.osm.mutable.Way(
                id=_RELATION_SECOND_OUTER_WAY_ID,
                nodes=second_outer,
                tags={},
            )
        )
        writer.add_relation(
            osmium.osm.mutable.Relation(
                id=_RELATION_ID,
                members=[
                    ("w", _RELATION_OUTER_WAY_ID, "outer"),
                    ("w", _RELATION_SECOND_OUTER_WAY_ID, "outer"),
                ],
                tags={
                    "type": "multipolygon",
                    "wikidata": relation_tag,
                    "name": "Twin",
                    "leisure": "park",
                },
            )
        )
    finally:
        writer.close()


@pytest.fixture
def state() -> _State:
    return _State()


@given(parsers.parse('a tiny PBF with a tagged way and a multipolygon relation tagged "{tag}"'))
def tiny_pbf(state: _State, tmp_path: Path, tag: str) -> None:
    pbf = tmp_path / f"{_REGION}-latest.osm.pbf"
    _write_pbf(pbf, tag)
    state.pbf = pbf
    state.relation_tag = tag


@when("I run extraction with in-memory enrichment clients")
def run_extraction(state: _State, tmp_path: Path) -> None:
    assert state.pbf is not None
    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()
    result = process_pbf(
        state.pbf,
        data_root=data_root,
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        settings=Settings(),
    )
    state.rows = pq.read_table(result.polygons_path).to_pylist()


def _identities(state: _State) -> set[tuple[str, int]]:
    return {(str(row["osm_type"]), int(row["osm_id"])) for row in state.rows}


@then("the polygon Parquet has one row per tagged polygon")
def one_row_per_polygon(state: _State) -> None:
    assert len(state.rows) == 2
    assert _identities(state) == {("way", _TAGGED_WAY_ID), ("relation", _RELATION_ID)}


@then("every polygon row has a valid polygon_id and a positive area")
def valid_ids_and_area(state: _State) -> None:
    for row in state.rows:
        assert row["polygon_id"] == f"{_REGION}-latest:{row['osm_type']}:{row['osm_id']}"
        assert row["area_m2"] > 0


@then(parsers.parse('the relation row keeps both QIDs "{first}" and "{second}"'))
def relation_keeps_qids(state: _State, first: str, second: str) -> None:
    relation = next(row for row in state.rows if row["osm_type"] == "relation")
    assert set(qids_from_osm_tag(str(relation["wikidata"]))) == {first, second}


@then("the untagged closed way is not in the polygon Parquet")
def untagged_excluded(state: _State) -> None:
    identities = _identities(state)
    assert ("way", _UNTAGGED_WAY_ID) not in identities
    assert ("way", _RELATION_OUTER_WAY_ID) not in identities
    assert ("way", _RELATION_SECOND_OUTER_WAY_ID) not in identities
