import pyarrow as pa

from osm_polygon_wikidata_only.domain.schema import polygon_schema
from osm_polygon_wikidata_only.v2.schema import (
    V2_LINK_COLUMNS,
    V2_POLYGON_COLUMNS,
    V2_WIKIPEDIA_DOCUMENT_COLUMNS,
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)


def test_v2_polygon_schema_is_explicit_union_contract() -> None:
    schema = polygon_v2_schema()
    assert tuple(schema.names) == V2_POLYGON_COLUMNS
    assert schema.field("wikidata").nullable
    assert schema.field("wikipedia_tag_refs").type == pa.string()
    assert schema.field("has_wikidata").type == pa.bool_()


def test_v2_documents_allow_missing_wikidata_without_changing_v1_schema() -> None:
    schema = wikipedia_document_v2_schema()
    assert tuple(schema.names) == V2_WIKIPEDIA_DOCUMENT_COLUMNS
    assert schema.field("wikidata").nullable
    assert schema.field("document_id").type == pa.string()
    assert tuple(polygon_schema().names) != V2_WIKIPEDIA_DOCUMENT_COLUMNS


def test_v2_links_record_provenance_and_nullable_qid() -> None:
    schema = polygon_document_link_v2_schema()
    assert tuple(schema.names) == V2_LINK_COLUMNS
    assert schema.field("wikidata").nullable
    assert schema.field("link_sources").type == pa.string()
    assert schema.field("project").type == pa.string()
