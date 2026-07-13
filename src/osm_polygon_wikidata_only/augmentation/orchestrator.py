"""Incremental publication of additive regional augmentation sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import Document, Section, WikidataFact, document_from_article_row
from .progress import AugmentationProgress
from .schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from .sections import parse_sections
from .wikimedia import FACT_PROPERTIES, discover_wikivoyage_sitelinks, normalize_facts

CONTRACT_VERSION = "text-sidecars-v1"


class AugmentationClient(Protocol):
    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]: ...
    def parse_html(self, project: str, language: str, revision_id: int) -> str: ...
    def wikivoyage_document(
        self, qid: str, language: str, site: str, title: str
    ) -> Document | None: ...


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    wikipedia_documents_path: Path
    wikipedia_sections_path: Path
    wikivoyage_documents_path: Path
    wikivoyage_sections_path: Path
    wikidata_facts_path: Path
    manifest_path: Path
    counts: dict[str, int]


def sidecar_paths(data_root: DataRoot, stem: str) -> tuple[Path, ...]:
    root = data_root.processed
    return (
        root / "wikipedia" / "documents" / f"{stem}.parquet",
        root / "wikipedia" / "sections" / f"{stem}.parquet",
        root / "wikivoyage" / "documents" / f"{stem}.parquet",
        root / "wikivoyage" / "sections" / f"{stem}.parquet",
        root / "wikidata" / "facts" / f"{stem}.parquet",
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(
    path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...], schema: pa.Schema
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = (
        pa.Table.from_pylist(normalized, schema=schema)
        if normalized
        else pa.Table.from_pylist([], schema=schema)
    )
    pq.write_table(table, temporary, compression="snappy")  # type: ignore[no-untyped-call]
    os.replace(temporary, path)


def _label_maps(entities: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for qid, entity in entities.items():
        labels = entity.get("labels") or {}
        out[qid] = {
            str(language): str(value.get("value", ""))
            for language, value in labels.items()
            if isinstance(value, dict) and value.get("value")
        }
    return out


def augment_region(
    data_root: DataRoot,
    stem: str,
    client: AugmentationClient,
    *,
    progress: AugmentationProgress | None = None,
) -> AugmentationResult:
    progress = progress or AugmentationProgress()
    articles_path = data_root.processed_articles / f"{stem}.parquet"
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not articles_path.exists() or not polygons_path.exists():
        raise FileNotFoundError(f"Core region is incomplete: {stem}")
    core_hashes = {str(path): _hash(path) for path in (articles_path, polygons_path)}
    article_rows = pq.read_table(articles_path).to_pylist()  # type: ignore[no-untyped-call]
    polygon_rows = pq.read_table(polygons_path, columns=["wikidata"]).to_pylist()  # type: ignore[no-untyped-call]
    wikipedia_documents = sorted(
        (document_from_article_row(row) for row in article_rows), key=lambda row: row.document_id
    )
    qids = sorted({str(row["wikidata"]) for row in polygon_rows if row.get("wikidata")})
    progress.start("Wikidata entities", total=len(qids))
    entities = client.entities(qids, props="sitelinks|claims")
    progress.complete()

    voyage_links = [
        (qid, language, site, title)
        for qid, entity in sorted(entities.items())
        for language, site, title in discover_wikivoyage_sitelinks(entity)
    ]
    progress.start("Wikivoyage documents", total=len(voyage_links))

    def fetch_voyage(item: tuple[str, str, str, str]) -> Document | None:
        try:
            return client.wikivoyage_document(*item)
        finally:
            progress.advance()

    with ThreadPoolExecutor(max_workers=8) as executor:
        voyage_documents = [
            document
            for document in executor.map(fetch_voyage, voyage_links)
            if document is not None
        ]
    voyage_documents.sort(key=lambda row: row.document_id)

    all_documents = wikipedia_documents + voyage_documents
    progress.start("Article sections", total=len(all_documents))

    def fetch_html(document: Document) -> str:
        try:
            return client.parse_html(document.project, document.language, document.revision_id)
        finally:
            progress.advance()

    with ThreadPoolExecutor(max_workers=8) as executor:
        html_pages = list(executor.map(fetch_html, all_documents))
    sections_by_project: dict[str, list[Section]] = {"wikipedia": [], "wikivoyage": []}
    for document, html in zip(all_documents, html_pages, strict=True):
        sections_by_project[document.project].extend(parse_sections(document, html))
    for rows in sections_by_project.values():
        rows.sort(key=lambda row: (row.document_id, row.section_index))

    label_ids = set(FACT_PROPERTIES)
    for entity in entities.values():
        for property_id, claims in (entity.get("claims") or {}).items():
            if property_id not in FACT_PROPERTIES:
                continue
            for claim in claims:
                value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(value, dict) and value.get("id"):
                    label_ids.add(str(value["id"]))
    labels = _label_maps(client.entities(label_ids, props="labels"))
    progress.start("Wikidata facts", total=len(entities))
    facts: list[WikidataFact] = []
    for entity in entities.values():
        facts.extend(normalize_facts(entity, labels))
        progress.advance()
    facts.sort(key=lambda row: row.fact_id)

    paths = sidecar_paths(data_root, stem)
    progress.start("Writing sidecars", total=len(paths))
    _write(
        paths[0],
        [row.to_dict() for row in wikipedia_documents],
        DOCUMENT_COLUMNS,
        document_schema(),
    )
    progress.advance()
    _write(
        paths[1],
        [row.to_dict() for row in sections_by_project["wikipedia"]],
        SECTION_COLUMNS,
        section_schema(),
    )
    progress.advance()
    _write(
        paths[2], [row.to_dict() for row in voyage_documents], DOCUMENT_COLUMNS, document_schema()
    )
    progress.advance()
    _write(
        paths[3],
        [row.to_dict() for row in sections_by_project["wikivoyage"]],
        SECTION_COLUMNS,
        section_schema(),
    )
    progress.advance()
    _write(paths[4], [row.to_dict() for row in facts], FACT_COLUMNS, fact_schema())
    progress.advance()
    if core_hashes != {str(path): _hash(path) for path in (articles_path, polygons_path)}:
        raise RuntimeError("Core artifacts changed during augmentation")

    counts = {
        "wikipedia_documents": len(wikipedia_documents),
        "wikipedia_sections": len(sections_by_project["wikipedia"]),
        "wikivoyage_documents": len(voyage_documents),
        "wikivoyage_sections": len(sections_by_project["wikivoyage"]),
        "wikidata_facts": len(facts),
    }
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[stem] = {
        "contract_version": CONTRACT_VERSION,
        "core_hashes": core_hashes,
        "paths": [str(path.relative_to(data_root.processed)) for path in paths],
        "counts": counts,
        "completed_at": utc_now_iso(),
    }
    atomic_write_text(manifest_path, dumps(manifest) + "\n")
    return AugmentationResult(
        paths[0], paths[1], paths[2], paths[3], paths[4], manifest_path, counts
    )


def completed_region_stems(data_root: DataRoot) -> list[str]:
    article_stems = {path.stem for path in data_root.processed_articles.glob("*.parquet")}
    polygon_stems = {path.stem for path in data_root.processed_polygons.glob("*.parquet")}
    return sorted(article_stems & polygon_stems)


def augmentation_is_current(data_root: DataRoot, stem: str) -> bool:
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not manifest_path.exists() or not all(
        path.exists() for path in sidecar_paths(data_root, stem)
    ):
        return False
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get(stem, {})
    if entry.get("contract_version") != CONTRACT_VERSION:
        return False
    core_paths = (
        data_root.processed_articles / f"{stem}.parquet",
        data_root.processed_polygons / f"{stem}.parquet",
    )
    if not all(path.exists() for path in core_paths):
        return False
    expected = entry.get("core_hashes")
    current = {str(path): _hash(path) for path in core_paths}
    return bool(expected == current)


__all__ = [
    "AugmentationResult",
    "augment_region",
    "augmentation_is_current",
    "completed_region_stems",
    "sidecar_paths",
]
