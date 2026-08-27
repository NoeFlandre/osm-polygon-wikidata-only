from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2.comparison import (
    select_v2_added_wikipedia_tag_document_polygon_ids,
)
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def _row(schema: pa.Schema, **values: object) -> dict[str, object]:
    row = {field.name: None for field in schema}
    row.update(values)
    return row


def _write_v1_fixture(root: Path) -> None:
    polygon_path = root / "polygons" / "region-latest.parquet"
    polygon_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_row(polygon_schema(), polygon_id="region-latest:way:1")],
            schema=polygon_schema(),
        ),
        polygon_path,
    )

    document_path = root / "wikipedia" / "documents" / "region-latest.parquet"
    document_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_row(wikipedia_document_schema(), document_id="old-doc")],
            schema=wikipedia_document_schema(),
        ),
        document_path,
    )


def _v2_polygon(polygon_id: str, sources: str) -> dict[str, object]:
    row = empty_row(tuple(field.name for field in polygon_schema()))
    row.update(
        {
            "polygon_id": polygon_id,
            "discovery_sources": sources,
            "has_wikidata": "wikidata" in sources,
            "lon": 2.0,
            "lat": 48.0,
        }
    )
    return row


def test_selection_contains_only_v2_added_wikipedia_tag_document_polygons(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    _write_v1_fixture(v1)

    document = _row(wikipedia_document_v2_schema(), document_id="new-doc")
    write_v2_region(
        v2,
        "region-latest",
        polygons=[
            _v2_polygon("region-latest:way:1", '["wikipedia_tag"]'),
            _v2_polygon("region-latest:way:2", '["wikipedia_tag"]'),
            _v2_polygon("region-latest:way:3", '["wikidata","wikipedia_tag"]'),
            _v2_polygon("region-latest:way:4", '["wikipedia_tag"]'),
            _v2_polygon("region-latest:way:5", '["wikipedia_tag"]'),
        ],
        documents=[document],
        links=[
            {
                "polygon_id": "region-latest:way:1",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
            {
                "polygon_id": "region-latest:way:2",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
            {
                "polygon_id": "region-latest:way:3",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
            {
                "polygon_id": "region-latest:way:4",
                "document_id": "old-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
            {
                "polygon_id": "region-latest:way:5",
                "document_id": "new-doc",
                "project": "wikivoyage",
                "link_sources": '["osm_wikipedia_tag"]',
            },
        ],
    )

    selected = select_v2_added_wikipedia_tag_document_polygon_ids(v2, v1)

    assert selected == {"region-latest:way:2"}
