from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2 import comparison
from osm_polygon_wikidata_only.v2.comparison import (
    _iter_direct_document_batches,
    _iter_polygon_source_batches,
    _record_direct_document,
    _record_polygon_source,
    _validated_source_list,
    _values_from_batch,
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


def test_unique_values_reuses_the_open_parquet_file_for_schema_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "values.parquet"
    pq.write_table(pa.table({"language": ["en", "fr"]}), path)

    def fail_if_opened_separately(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema inspection must reuse the row-reading handle")

    monkeypatch.setattr(comparison.pq, "read_schema", fail_if_opened_separately)

    assert comparison._unique_values((path,), "language") == {"en", "fr"}


def test_values_from_batch_returns_non_empty_string_values() -> None:
    batch = pa.record_batch({"value": ["en", None, "", "fr"]})

    assert _values_from_batch(batch) == {"en", "fr"}


def test_record_polygon_source_only_updates_requested_identity() -> None:
    values: dict[str, set[str]] = {}

    _record_polygon_source(
        values,
        "keep",
        '["wikipedia_tag"]',
        {"keep"},
        Path("polygons.parquet"),
    )
    _record_polygon_source(
        values,
        "drop",
        '["wikidata"]',
        {"keep"},
        Path("polygons.parquet"),
    )

    assert values == {"keep": {"wikipedia_tag"}}


def test_polygon_source_batches_skip_files_without_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "polygons.parquet"
    pq.write_table(pa.table({"polygon_id": ["polygon-1"]}), path)

    assert list(_iter_polygon_source_batches(path)) == []


def test_direct_document_batches_read_only_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "links.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["polygon-1"],
                "document_id": ["document-1"],
                "project": ["wikipedia"],
                "link_sources": ['["osm_wikipedia_tag"]'],
            }
        ),
        path,
    )

    batches = list(_iter_direct_document_batches(path))

    assert len(batches) == 1
    assert batches[0].schema.names == ["polygon_id", "document_id", "project", "link_sources"]


def test_record_direct_document_requires_wikipedia_tag_source() -> None:
    values: dict[str, set[str]] = {}

    _record_direct_document(
        values,
        "polygon-1",
        "document-1",
        "wikipedia",
        '["osm_wikipedia_tag"]',
        Path("links.parquet"),
    )
    _record_direct_document(
        values,
        "polygon-2",
        "document-2",
        "wikivoyage",
        '["osm_wikipedia_tag"]',
        Path("links.parquet"),
    )

    assert values == {"polygon-1": {"document-1"}}


def test_validated_source_list_rejects_non_string_members() -> None:
    with pytest.raises(ValueError, match="Invalid link_sources"):
        _validated_source_list(
            ["osm_wikipedia_tag", 7], "polygon-1", Path("links.parquet"), "link_sources"
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


def test_selection_contains_only_v2_added_wikipedia_tag_document_polygons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    _write_v1_fixture(v1)

    def fail_if_opened_separately(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema inspection must reuse the row-reading handle")

    monkeypatch.setattr(comparison.pq, "read_schema", fail_if_opened_separately)

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
