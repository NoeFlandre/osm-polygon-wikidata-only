"""Versioned Parquet contracts for the Wikipedia-tag dataset."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    WIKIPEDIA_DOCUMENT_DESCRIPTIONS,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS,
    CANONICAL_DESCRIPTIONS,
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
    polygon_schema,
)

V2_POLYGON_COLUMNS: tuple[str, ...] = (
    *POLYGON_COLUMNS,
    "wikipedia_tag_refs",
    "wikipedia_tag_rejections",
    "discovery_sources",
)
V2_WIKIPEDIA_DOCUMENT_COLUMNS = WIKIPEDIA_DOCUMENT_COLUMNS
V2_LINK_COLUMNS: tuple[str, ...] = (*CANONICAL_COLUMNS, "link_sources")

V2_POLYGON_DESCRIPTIONS = {
    **POLYGON_DESCRIPTIONS,
    "wikidata": "Nullable Wikidata QID value from the OSM wikidata=* tag.",
    "has_wikidata": "True when the polygon has a valid Wikidata tag.",
    "has_wikipedia": "True when at least one linked Wikipedia document exists.",
    "wikipedia_tag_refs": "Deterministic JSON list of normalized direct Wikipedia tag references.",
    "wikipedia_tag_rejections": "Deterministic JSON list of malformed Wikipedia tag records.",
    "discovery_sources": "Deterministic JSON list containing wikidata and/or wikipedia_tag.",
}
V2_DOCUMENT_DESCRIPTIONS = {
    **WIKIPEDIA_DOCUMENT_DESCRIPTIONS,
    "document_id": "Stable V2 identity based on Wikidata when available, otherwise the Wikipedia page.",
    "article_id": "Stable V2 article identity based on Wikidata when available, otherwise the Wikipedia page.",
    "wikidata": "Nullable Wikidata QID resolved from V1 reuse or the direct page.",
}
V2_LINK_DESCRIPTIONS = {
    **CANONICAL_DESCRIPTIONS,
    "document_id": "Stable V2 document identity.",
    "wikidata": "Nullable Wikidata QID for the linked document.",
    "link_sources": "Deterministic JSON list of discovery sources for this relationship.",
}


def _with_description(field: pa.Field, description: str) -> pa.Field:
    return pa.field(
        field.name,
        field.type,
        nullable=field.nullable,
        metadata={b"description": description.encode()},
    )


def polygon_v2_schema() -> pa.Schema:
    """Return the V2 polygon schema with direct-tag provenance fields."""
    base = polygon_schema()
    fields = [
        _with_description(base.field(name), V2_POLYGON_DESCRIPTIONS[name])
        for name in POLYGON_COLUMNS
    ]
    fields.extend(
        pa.field(
            name, pa.string(), metadata={b"description": V2_POLYGON_DESCRIPTIONS[name].encode()}
        )
        for name in V2_POLYGON_COLUMNS[len(POLYGON_COLUMNS) :]
    )
    return pa.schema(fields)


def wikipedia_document_v2_schema() -> pa.Schema:
    """Return the V2 document schema, allowing a missing Wikidata QID."""
    base = wikipedia_document_schema()
    return pa.schema(
        [_with_description(field, V2_DOCUMENT_DESCRIPTIONS[field.name]) for field in base]
    )


def polygon_document_link_v2_schema() -> pa.Schema:
    """Return the V2 link schema with deterministic source provenance."""
    base = polygon_document_link_schema()
    fields = [_with_description(field, V2_LINK_DESCRIPTIONS[field.name]) for field in base]
    fields.append(
        pa.field(
            "link_sources",
            pa.string(),
            metadata={b"description": V2_LINK_DESCRIPTIONS["link_sources"].encode()},
        )
    )
    return pa.schema(fields)


__all__ = [
    "V2_LINK_COLUMNS",
    "V2_POLYGON_COLUMNS",
    "V2_WIKIPEDIA_DOCUMENT_COLUMNS",
    "polygon_document_link_v2_schema",
    "polygon_v2_schema",
    "wikipedia_document_v2_schema",
]
