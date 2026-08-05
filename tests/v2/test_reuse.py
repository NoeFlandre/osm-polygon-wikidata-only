import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2.reuse import _rows, load_v1_region


def _write(path: Path, schema: pa.Schema, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)


def _empty(schema: pa.Schema) -> dict:
    return {
        field.name: (
            0
            if pa.types.is_integer(field.type)
            else False
            if pa.types.is_boolean(field.type)
            else 0.0
            if pa.types.is_floating(field.type)
            else ""
        )
        for field in schema
    }


def test_load_v1_region_adds_direct_tag_metadata_to_existing_polygons(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    polygon_schema_value = polygon_schema()
    polygon = empty_row(tuple(field.name for field in polygon_schema_value))
    polygon.update(
        {
            "polygon_id": "region-latest:way:1",
            "wikidata": "Q42",
            "tags": json.dumps({"wikipedia": "fr:Douglas Adams"}),
        }
    )
    _write(root.processed_polygons / "region-latest.parquet", polygon_schema_value, polygon)
    document_schema = wikipedia_document_schema()
    document = _empty(document_schema)
    document.update(
        {
            "document_id": "Q42:wikipedia:fr:1:2",
            "article_id": "Q42:fr:1:2",
            "wikidata": "Q42",
            "project": "wikipedia",
            "language": "fr",
            "page_id": 1,
            "revision_id": 2,
        }
    )
    _write(root.processed / "wikipedia/documents/region-latest.parquet", document_schema, document)
    link_schema = polygon_document_link_schema()
    link = _empty(link_schema)
    link.update(
        {
            "polygon_id": polygon["polygon_id"],
            "document_id": document["document_id"],
            "project": "wikipedia",
            "wikidata": "Q42",
            "language": "fr",
            "source_pbf": "region-latest.osm.pbf",
            "region": "region",
            "osm_type": "way",
            "osm_id": 1,
            "page_id": 1,
            "revision_id": 2,
        }
    )
    _write(root.processed_links / "region-latest.parquet", link_schema, link)

    result = load_v1_region(root, "region-latest")
    assert result.polygons[0]["discovery_sources"] == '["wikidata","wikipedia_tag"]'
    assert result.links[0]["link_sources"] == '["wikidata_sitelink"]'


def test_rows_closes_parquet_file_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated V1 shard loads must not retain a descriptor per read."""
    root = DataRoot(tmp_path)
    root.ensure()
    path = root.processed_polygons / "region-latest.parquet"
    schema = polygon_schema()
    _write(path, schema, _empty(schema))

    import osm_polygon_wikidata_only.v2.reuse as reuse

    original = reuse.pq.ParquetFile
    opened: list[object] = []
    closed: list[object] = []

    class TrackedParquetFile:
        def __init__(self, source: Path) -> None:
            self._inner = original(source)
            opened.append(self)

        def __enter__(self) -> "TrackedParquetFile":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def close(self) -> None:
            if self not in closed:
                self._inner.close()
                closed.append(self)

        def read(self):
            return self._inner.read()

    monkeypatch.setattr(reuse.pq, "ParquetFile", TrackedParquetFile)
    assert _rows(path)
    assert opened
    assert closed == opened
