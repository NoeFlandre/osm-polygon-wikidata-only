"""Load and validate the tabular inputs used by the recovery audit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import fact_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag

from .audit_types import RegionRows, ScanError


def region_paths(data_root: DataRoot, stem: str) -> tuple[tuple[str, Path, bool], ...]:
    return (
        ("polygons", data_root.processed_polygons / f"{stem}.parquet", True),
        ("polygon_articles", data_root.processed_links / f"{stem}.parquet", True),
        (
            "wikipedia_documents",
            data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
            True,
        ),
        (
            "wikidata_facts",
            data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
            True,
        ),
    )


def load_region_rows(data_root: DataRoot, stem: str) -> RegionRows:
    paths = {label: path for label, path, _ in region_paths(data_root, stem)}
    require_schema(paths["polygons"], polygon_schema())
    link_schema = pq.read_schema(paths["polygon_articles"])
    canonical_links = link_schema.equals(polygon_document_link_schema(), check_metadata=True)
    if not canonical_links:
        require_schema(paths["polygon_articles"], polygon_article_schema())
    require_schema(paths["wikipedia_documents"], wikipedia_document_schema())
    require_schema(paths["wikidata_facts"], fact_schema())
    link_columns = (
        ["polygon_id", "document_id", "project", "wikidata"]
        if canonical_links
        else ["polygon_id", "article_id", "wikidata"]
    )
    return RegionRows(
        paths=paths,
        canonical_links=canonical_links,
        polygon_rows=read_rows(paths["polygons"], ["polygon_id", "wikidata"]),
        link_rows=read_rows(paths["polygon_articles"], link_columns),
        document_rows=read_rows(
            paths["wikipedia_documents"],
            ["article_id", "document_id", "wikidata"],
        ),
        fact_rows=read_rows(paths["wikidata_facts"], ["fact_id", "wikidata"]),
    )


def index_polygon_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[str]]]:
    polygons: dict[str, tuple[str, ...]] = {}
    polygon_ids_by_qid: dict[str, list[str]] = {}
    for row in rows:
        polygon_id = required_string(row, "polygon_id", "polygons")
        raw_qid = required_string(row, "wikidata", "polygons")
        qids = qids_from_osm_tag(raw_qid)
        if not qids:
            raise ScanError(f"polygons contains invalid Wikidata identifier {raw_qid!r}")
        if polygon_id in polygons:
            raise ScanError(f"polygons contains duplicate polygon_id {polygon_id!r}")
        polygons[polygon_id] = qids
        for qid in qids:
            polygon_ids_by_qid.setdefault(qid, []).append(polygon_id)
    return polygons, polygon_ids_by_qid


def index_document_rows(
    rows: list[dict[str, Any]],
    polygon_ids_by_qid: Mapping[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    documents_by_article: dict[str, dict[str, Any]] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    document_ids: set[str] = set()
    orphan_document_ids: list[str] = []
    for row in rows:
        article_id = required_string(row, "article_id", "wikipedia documents")
        document_id = required_string(row, "document_id", "wikipedia documents")
        qid = required_string(row, "wikidata", "wikipedia documents")
        if article_id in documents_by_article:
            raise ScanError(f"wikipedia documents contains duplicate article_id {article_id!r}")
        if document_id in document_ids:
            raise ScanError(f"wikipedia documents contains duplicate document_id {document_id!r}")
        if qid not in polygon_ids_by_qid:
            orphan_document_ids.append(document_id)
        documents_by_article[article_id] = row
        documents_by_id[document_id] = row
        document_ids.add(document_id)
    return documents_by_article, documents_by_id, orphan_document_ids


def linked_polygon_qids(
    rows: list[dict[str, Any]],
    *,
    canonical_links: bool,
    polygons: Mapping[str, tuple[str, ...]],
    documents_by_article: Mapping[str, dict[str, Any]],
    documents_by_id: Mapping[str, dict[str, Any]],
    orphan_document_ids: list[str],
) -> set[tuple[str, str]]:
    linked_polygon_qids: set[tuple[str, str]] = set()
    link_ids: set[tuple[str, str]] = set()
    for row in rows:
        link = validate_link_row(
            row,
            canonical_links=canonical_links,
            polygons=polygons,
            documents_by_article=documents_by_article,
            documents_by_id=documents_by_id,
            orphan_document_ids=orphan_document_ids,
            link_ids=link_ids,
        )
        if link is not None:
            linked_polygon_qids.add((link[0], link[2]))
    return linked_polygon_qids


def validate_link_row(
    row: Mapping[str, Any],
    *,
    canonical_links: bool,
    polygons: Mapping[str, tuple[str, ...]],
    documents_by_article: Mapping[str, dict[str, Any]],
    documents_by_id: Mapping[str, dict[str, Any]],
    orphan_document_ids: list[str],
    link_ids: set[tuple[str, str]],
) -> tuple[str, str, str] | None:
    identity = link_identity(row, canonical_links=canonical_links, link_ids=link_ids)
    if identity is None:
        return None
    return resolve_link_target(
        identity,
        canonical_links=canonical_links,
        polygons=polygons,
        documents_by_article=documents_by_article,
        documents_by_id=documents_by_id,
        orphan_document_ids=orphan_document_ids,
    )


def link_identity(
    row: Mapping[str, Any],
    *,
    canonical_links: bool,
    link_ids: set[tuple[str, str]],
) -> tuple[str, str, str, str] | None:
    if canonical_links and str(row.get("project") or "") != "wikipedia":
        return None
    polygon_id = required_string(row, "polygon_id", "polygon_articles")
    reference_key = reference_column(canonical_links)
    reference_id = required_string(row, reference_key, "polygon_articles")
    qid = required_string(row, "wikidata", "polygon_articles")
    identity = (polygon_id, reference_id)
    remember_link_identity(identity, link_ids)
    return polygon_id, reference_key, reference_id, qid


def reference_column(canonical_links: bool) -> str:
    return "document_id" if canonical_links else "article_id"


def remember_link_identity(identity: tuple[str, str], link_ids: set[tuple[str, str]]) -> None:
    if identity in link_ids:
        raise ScanError(f"polygon_articles contains duplicate identity {identity!r}")
    link_ids.add(identity)


def resolve_link_target(
    identity: tuple[str, str, str, str],
    *,
    canonical_links: bool,
    polygons: Mapping[str, tuple[str, ...]],
    documents_by_article: Mapping[str, dict[str, Any]],
    documents_by_id: Mapping[str, dict[str, Any]],
    orphan_document_ids: list[str],
) -> tuple[str, str, str] | None:
    polygon_id, reference_key, reference_id, qid = identity
    document = (
        documents_by_id.get(reference_id)
        if canonical_links
        else documents_by_article.get(reference_id)
    )
    if document is not None and str(document["document_id"]) in orphan_document_ids:
        return None
    validate_polygon_qid(polygon_id, qid, polygons)
    return finish_link_target(
        polygon_id,
        reference_key,
        reference_id,
        qid,
        document,
        canonical_links,
    )


def validate_polygon_qid(
    polygon_id: str,
    qid: str,
    polygons: Mapping[str, tuple[str, ...]],
) -> None:
    polygon_qids = polygons.get(polygon_id)
    if polygon_qids is None:
        raise ScanError(f"polygon_articles references absent polygon_id {polygon_id!r}")
    if qid not in polygon_qids:
        raise ScanError(
            f"polygon_articles QID {qid!r} disagrees with polygon {polygon_id!r} "
            f"QIDs {polygon_qids!r}"
        )


def finish_link_target(
    polygon_id: str,
    reference_key: str,
    reference_id: str,
    qid: str,
    document: dict[str, Any] | None,
    canonical_links: bool,
) -> tuple[str, str, str] | None:
    if document is None:
        return missing_link_target(reference_key, reference_id, canonical_links)
    if str(document.get("wikidata") or "") != qid:
        raise ScanError(f"document {reference_id!r} disagrees with link QID {qid!r}")
    return polygon_id, reference_id, qid


def missing_link_target(
    reference_key: str,
    reference_id: str,
    canonical_links: bool,
) -> None:
    if canonical_links:
        # A canonical link can outlive its document after an interrupted run.
        return
    raise ScanError(f"polygon_articles references absent {reference_key} {reference_id!r}")


def missing_polygon_links(
    polygon_ids_by_qid: Mapping[str, list[str]],
    linked_polygon_qids: set[tuple[str, str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    missing: list[tuple[str, tuple[str, ...]]] = []
    for qid, polygon_ids in sorted(polygon_ids_by_qid.items()):
        missing_ids = tuple(
            polygon_id
            for polygon_id in sorted(polygon_ids)
            if (polygon_id, qid) not in linked_polygon_qids
        )
        if missing_ids:
            missing.append((qid, missing_ids))
    return tuple(missing)


def orphan_fact_ids(rows: list[dict[str, Any]], valid_polygon_qids: set[str]) -> list[str]:
    orphan_ids: list[str] = []
    seen_fact_ids: set[str] = set()
    for row in rows:
        fact_id = required_string(row, "fact_id", "wikidata facts")
        qid = required_string(row, "wikidata", "wikidata facts")
        if fact_id in seen_fact_ids:
            raise ScanError(f"wikidata facts contains duplicate fact_id {fact_id!r}")
        seen_fact_ids.add(fact_id)
        if qid not in valid_polygon_qids:
            orphan_ids.append(fact_id)
    return orphan_ids


def require_schema(path: Path, expected: pa.Schema) -> None:
    actual: pa.Schema = pq.read_schema(path)
    if not actual.equals(expected, check_metadata=True):
        raise ScanError(f"Recovery input schema mismatch: {path}")


def read_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    table: pa.Table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def required_string(row: Mapping[str, Any], key: str, table: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ScanError(f"{table} contains an empty or non-string {key}")
    return value


__all__ = [
    "index_document_rows",
    "index_polygon_rows",
    "link_identity",
    "linked_polygon_qids",
    "load_region_rows",
    "missing_polygon_links",
    "orphan_fact_ids",
    "read_rows",
    "reference_column",
    "region_paths",
    "remember_link_identity",
    "require_schema",
    "resolve_link_target",
    "validate_link_row",
    "validate_polygon_qid",
]
