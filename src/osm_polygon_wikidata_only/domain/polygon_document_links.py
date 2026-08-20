"""Canonical polygon-to-document link schema, construction, and validation.

This module is the single source of truth for the 11-column canonical
``polygon_articles`` table. The builder joins polygons to documents by
QID and emits one row per ``(polygon_id, project, document_id)``
identity. The validator enforces strict identity invariants:

* ``project`` must be one of ``{"wikipedia", "wikivoyage"}``;
* the document_id's project slot must match the column;
* conflicting duplicates (same identity, different values) are rejected;
* byte-identical duplicates are collapsed.

Multi-QID OSM tags: a polygon's ``wikidata`` OSM-tag value may be a
single QID (``Q1``) or a semicolon-separated list of QIDs
(``Q1;Q2``). The canonical builder uses the project's
:func:`qids_from_osm_tag` parser -- no ad-hoc QID regex. A link is
emitted for every (polygon, document) pair whose QID is a member of
the polygon's parsed QID set.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pyarrow as pa

from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    is_valid_qid,
    qids_from_osm_tag,
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "polygon_id",
    "document_id",
    "project",
    "wikidata",
    "language",
    "source_pbf",
    "region",
    "osm_type",
    "osm_id",
    "page_id",
    "revision_id",
)
CANONICAL_DESCRIPTIONS: dict[str, str] = {
    "polygon_id": "FK to polygons polygon_id.",
    "document_id": (
        "FK to the project documents table. Format: "
        "<wikidata>:<project>:<language>:<page_id>:<revision_id>."
    ),
    "project": "Source project: wikipedia or wikivoyage.",
    "wikidata": "Wikidata QID (denormalized for fast filtering).",
    "language": "Document language code.",
    "source_pbf": "Source PBF filename (denormalized).",
    "region": "Geofabrik region slug (denormalized).",
    "osm_type": "OSM element type: way or relation.",
    "osm_id": "OSM numeric identifier.",
    "page_id": "MediaWiki page ID.",
    "revision_id": "MediaWiki revision ID.",
}

PROJECTS: frozenset[str] = frozenset({"wikipedia", "wikivoyage"})

LINK_CONTRACT_VERSION = "polygon-document-links-v1"


def polygon_document_link_schema() -> pa.Schema:
    """Return the canonical 11-column schema as a pyarrow Schema."""
    fields = [
        pa.field(
            "polygon_id",
            pa.string(),
            metadata={"description": CANONICAL_DESCRIPTIONS["polygon_id"]},
        ),
        pa.field(
            "document_id",
            pa.string(),
            metadata={"description": CANONICAL_DESCRIPTIONS["document_id"]},
        ),
        pa.field(
            "project", pa.string(), metadata={"description": CANONICAL_DESCRIPTIONS["project"]}
        ),
        pa.field(
            "wikidata", pa.string(), metadata={"description": CANONICAL_DESCRIPTIONS["wikidata"]}
        ),
        pa.field(
            "language", pa.string(), metadata={"description": CANONICAL_DESCRIPTIONS["language"]}
        ),
        pa.field(
            "source_pbf",
            pa.string(),
            metadata={"description": CANONICAL_DESCRIPTIONS["source_pbf"]},
        ),
        pa.field("region", pa.string(), metadata={"description": CANONICAL_DESCRIPTIONS["region"]}),
        pa.field(
            "osm_type", pa.string(), metadata={"description": CANONICAL_DESCRIPTIONS["osm_type"]}
        ),
        pa.field("osm_id", pa.int64(), metadata={"description": CANONICAL_DESCRIPTIONS["osm_id"]}),
        pa.field(
            "page_id", pa.int64(), metadata={"description": CANONICAL_DESCRIPTIONS["page_id"]}
        ),
        pa.field(
            "revision_id",
            pa.int64(),
            metadata={"description": CANONICAL_DESCRIPTIONS["revision_id"]},
        ),
    ]
    return pa.schema(fields)


def _document_id_project(document_id: str) -> str | None:
    """Return the project slot from a document_id, or None if not present."""
    parts = document_id.split(":")
    if len(parts) < 2:
        return None
    candidate = parts[1]
    if candidate in PROJECTS:
        return candidate
    return None


def _document_id_wikidata(document_id: str) -> str | None:
    """Return the wikidata slot from a document_id, or None if not present."""
    parts = document_id.split(":")
    if len(parts) < 1:
        return None
    candidate = parts[0]
    if is_valid_qid(candidate):
        return candidate
    return None


def _is_valid_stem(stem: str) -> bool:
    if not stem or stem in {".", ".."}:
        return False
    return "/" not in stem and "\\" not in stem


def _coerce_polygon_row(polygon: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical polygon fields used to populate a link row."""
    polygon_id = str(polygon.get("polygon_id", ""))
    if not _is_valid_stem(polygon_id):
        raise ValueError(f"Invalid polygon_id: {polygon_id!r}")
    raw_wikidata = str(polygon.get("wikidata", ""))
    qids = qids_from_osm_tag(raw_wikidata)
    if not qids:
        raise ValueError(
            f"Polygon polygon_id={polygon_id!r} has invalid wikidata: {raw_wikidata!r}"
        )
    return {
        "polygon_id": polygon_id,
        "wikidata_tag": raw_wikidata,
        "wikidata_qids": set(qids),
        "source_pbf": str(polygon.get("source_pbf", "")),
        "region": str(polygon.get("region", "")),
        "osm_type": str(polygon.get("osm_type", "")),
        "osm_id": int(polygon.get("osm_id", 0)),
    }


def _validate_document(doc: dict[str, Any]) -> None:
    document_id = str(doc.get("document_id", ""))
    if not document_id:
        raise ValueError("Document is missing document_id")
    project = str(doc.get("project", ""))
    if project not in PROJECTS:
        raise ValueError(f"Document document_id={document_id!r} has invalid project: {project!r}")
    declared_project = _document_id_project(document_id)
    if declared_project != project:
        raise ValueError(
            f"Document document_id={document_id!r} project slot disagrees with "
            f"project field ({declared_project!r} vs {project!r})"
        )
    wikidata = str(doc.get("wikidata", ""))
    if not is_valid_qid(wikidata):
        raise ValueError(f"Document document_id={document_id!r} has invalid wikidata: {wikidata!r}")


def _coerce_document_id(doc: dict[str, Any]) -> str:
    return str(doc["document_id"])


def _build_link_row(polygon_fields: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    document_id = _coerce_document_id(doc)
    return {
        "polygon_id": polygon_fields["polygon_id"],
        "document_id": document_id,
        "project": str(doc["project"]),
        "wikidata": str(doc["wikidata"]),
        "language": str(doc.get("language", "")),
        "source_pbf": polygon_fields["source_pbf"],
        "region": polygon_fields["region"],
        "osm_type": polygon_fields["osm_type"],
        "osm_id": polygon_fields["osm_id"],
        "page_id": int(doc.get("page_id", 0)),
        "revision_id": int(doc.get("revision_id", 0)),
    }


def build_polygon_document_links(
    polygons: Iterable[dict[str, Any]],
    *,
    wikipedia_documents: Iterable[dict[str, Any]] = (),
    wikivoyage_documents: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Join polygons to documents by QID and emit canonical link rows.

    The polygon's ``wikidata`` field is the OSM-tag value (possibly
    a semicolon-separated list). The canonical
    :func:`qids_from_osm_tag` parser splits it into the polygon's QID
    set. A document emits one row per polygon whose QID set contains
    the document's QID. The ``project`` for each link is read from
    the document, not the caller.
    """
    poly_list = list(polygons)
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for raw in poly_list:
        fields = _coerce_polygon_row(raw)
        for qid in fields["wikidata_qids"]:
            by_qid.setdefault(qid, []).append(fields)

    links: list[dict[str, Any]] = []
    for doc in list(wikipedia_documents) + list(wikivoyage_documents):
        _validate_document(doc)
        qid = str(doc["wikidata"])
        for polygon_fields in by_qid.get(qid, ()):
            links.append(_build_link_row(polygon_fields, doc))
    return validate_polygon_document_links(links)


def validate_polygon_document_links(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate, deduplicate, and sort a sequence of canonical link rows.

    * Byte-identical duplicates are collapsed to one.
    * Conflicting duplicates (same identity, different values) raise.
    * Output is sorted by ``(polygon_id, project, document_id)``.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        _merge_link_row(out, _validated_link_row(raw))
    return [out[k] for k in sorted(out.keys())]


def _validated_link_row(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"Each link row must be a dict, got {type(raw).__name__}")
    return _coerce_row(cast(dict[str, Any], raw))


def _merge_link_row(out: dict[tuple[str, str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["polygon_id"], row["project"], row["document_id"])
    existing = out.get(key)
    if existing is None:
        out[key] = row
        return
    if existing != row:
        raise ValueError(
            f"Conflicting duplicate link identity {key}: existing={existing} new={row}"
        )


def _copy_canonical_columns(row: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for col in CANONICAL_COLUMNS:
        if col not in row:
            raise ValueError(f"Link row is missing required column {col!r}")
        coerced[col] = row[col]
    return coerced


def _validated_polygon_id(value: Any) -> str:
    polygon_id = str(value)
    if not _is_valid_stem(polygon_id):
        raise ValueError(f"Invalid polygon_id: {polygon_id!r}")
    return polygon_id


def _validate_document_project(polygon_id: str, document_id: str, project: Any) -> None:
    if project not in PROJECTS:
        raise ValueError(f"polygon_id={polygon_id!r}: invalid project {project!r}")
    declared_project = _document_id_project(document_id)
    if declared_project != project:
        raise ValueError(
            f"polygon_id={polygon_id!r}: document_id={document_id!r} project slot "
            f"disagrees with project column ({declared_project!r} vs {project!r})"
        )


def _validate_document_language(polygon_id: str, document_id: str, language: Any) -> None:
    language_declared = _document_id_language(document_id)
    if language_declared is not None and language_declared != language:
        raise ValueError(
            f"polygon_id={polygon_id!r}: document_id={document_id!r} language slot "
            f"disagrees with language column ({language_declared!r} vs {language!r})"
        )


def _validate_document_wikidata(polygon_id: str, document_id: str, value: Any) -> None:
    wikidata = str(value)
    if not is_valid_qid(wikidata):
        raise ValueError(f"polygon_id={polygon_id!r}: invalid wikidata {wikidata!r}")
    doc_id_wikidata = _document_id_wikidata(document_id)
    if doc_id_wikidata is not None and doc_id_wikidata != wikidata:
        raise ValueError(
            f"polygon_id={polygon_id!r}: document_id={document_id!r} wikidata slot "
            f"disagrees with wikidata column ({doc_id_wikidata!r} vs {wikidata!r})"
        )


def _coerce_numeric_columns(coerced: dict[str, Any]) -> None:
    for column in ("osm_id", "page_id", "revision_id"):
        coerced[column] = int(coerced[column])


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced = _copy_canonical_columns(row)
    polygon_id = _validated_polygon_id(coerced["polygon_id"])
    document_id = str(coerced["document_id"])
    _validate_document_project(polygon_id, document_id, coerced["project"])
    _validate_document_language(polygon_id, document_id, coerced["language"])
    _validate_document_wikidata(polygon_id, document_id, coerced["wikidata"])
    _coerce_numeric_columns(coerced)
    return coerced


def _document_id_language(document_id: str) -> str | None:
    parts = document_id.split(":")
    if len(parts) < 4:
        return None
    return parts[2]


__all__ = [
    "CANONICAL_COLUMNS",
    "CANONICAL_DESCRIPTIONS",
    "LINK_CONTRACT_VERSION",
    "PROJECTS",
    "build_polygon_document_links",
    "polygon_document_link_schema",
    "validate_polygon_document_links",
]
