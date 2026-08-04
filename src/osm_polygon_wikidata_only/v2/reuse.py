"""Lossless V1 reuse and V2 relationship assembly.

The V2 build starts from finalized V1 shards.  Existing Wikipedia,
Wikivoyage, Wikidata, and polygon-link rows are copied unchanged where
possible.  Only direct Wikipedia-tag relationships not already represented
by a V1 document are fetched.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import shutil
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import Document
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import CANONICAL_COLUMNS
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    DirectEnrichmentResult,
    _cached_client,
    enrich_wikipedia_refs,
)
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf
from osm_polygon_wikidata_only.v2.storage import write_v2_region
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef, parse_wikipedia_tags

LOGGER = logging.getLogger(__name__)

SIDECAR_SUBDIRS: tuple[str, ...] = (
    "wikipedia/sections",
    "wikivoyage/documents",
    "wikivoyage/sections",
    "wikidata/facts",
)


@dataclass(frozen=True, slots=True)
class V1RegionData:
    """Rows and sidecar source paths loaded from one finalized V1 shard."""

    polygons: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    sidecars: tuple[Path, ...]


class SectionClient(Protocol):
    """Minimal exact-revision HTML client needed by the V2 section builder."""

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        """Return rendered HTML for one immutable Wikimedia revision."""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pq.read_table(path).to_pylist()


def load_v1_region(data_root: DataRoot, stem: str) -> V1RegionData:
    """Load V1 rows while accepting both pre- and post-migration links."""
    polygons = _rows(data_root.processed_polygons / f"{stem}.parquet")
    documents_path = data_root.processed / "wikipedia/documents" / f"{stem}.parquet"
    if documents_path.is_file():
        documents = _rows(documents_path)
    else:
        articles_path = data_root.processed_articles / f"{stem}.parquet"
        documents = [
            wikipedia_document_from_article_row(row).to_dict() for row in _rows(articles_path)
        ]
    links = _rows(data_root.processed_links / f"{stem}.parquet")
    by_article = {str(row.get("article_id")): row for row in documents}
    normalized_links = [_normalize_link(row, by_article) for row in links]
    sidecars = tuple(
        source / f"{stem}.parquet"
        for source in (data_root.processed / subdir for subdir in SIDECAR_SUBDIRS)
        if (source / f"{stem}.parquet").is_file()
    )
    return V1RegionData(
        polygons=tuple(_v2_polygon_row(row) for row in polygons),
        documents=tuple(documents),
        links=tuple(normalized_links),
        sidecars=sidecars,
    )


def _v2_polygon_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    tags: dict[str, str]
    try:
        parsed = json.loads(str(row.get("tags", "{}")))
        tags = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        tags = {}
    refs, rejections = parse_wikipedia_tags(tags)
    result["wikipedia_tag_refs"] = json.dumps(
        [
            {
                "language": ref.language,
                "title": ref.title,
                "raw_key": ref.raw_key,
                "raw_value": ref.raw_value,
            }
            for ref in refs
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    result["wikipedia_tag_rejections"] = json.dumps(
        [
            {"raw_key": item.raw_key, "raw_value": item.raw_value, "reason": item.reason}
            for item in rejections
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sources = ["wikidata"] if row.get("wikidata") else []
    if refs:
        sources.append("wikipedia_tag")
    result["discovery_sources"] = json.dumps(sources, separators=(",", ":"))
    return result


def _normalize_link(row: dict[str, Any], by_article: dict[str, dict[str, Any]]) -> dict[str, Any]:
    document_id = row.get("document_id")
    if not document_id:
        document = by_article.get(str(row.get("article_id", "")))
        document_id = document.get("document_id") if document else None
    if not document_id:
        raise ValueError(f"V1 link has no resolvable document identity: {row}")
    return {
        **{column: row.get(column) for column in CANONICAL_COLUMNS},
        "document_id": str(document_id),
        "project": row.get("project", "wikipedia"),
        "wikidata": row.get("wikidata"),
        "link_sources": json.dumps(["wikidata_sitelink"], separators=(",", ":")),
    }


def copy_v1_sidecars(data_root: DataRoot, stem: str, destination: Path) -> tuple[Path, ...]:
    """Copy finalized V1 sidecars into isolated V2 storage idempotently."""
    copied: list[Path] = []
    for source_root in (data_root.processed / subdir for subdir in SIDECAR_SUBDIRS):
        source = source_root / f"{stem}.parquet"
        if not source.is_file():
            continue
        target = destination / source_root.relative_to(data_root.processed) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or _sha256(source) != _sha256(target):
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
        copied.append(target)
    return tuple(copied)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _section_document(row: dict[str, Any]) -> Document:
    """Adapt a V2 document row to the existing V1 section parser model."""
    return Document(
        document_id=str(row["document_id"]),
        article_id=str(row.get("article_id") or ""),
        wikidata=str(row.get("wikidata") or ""),
        project=str(row.get("project") or "wikipedia"),
        language=str(row.get("language") or ""),
        site=str(row.get("site") or ""),
        title=str(row.get("title") or ""),
        url=str(row.get("url") or ""),
        page_id=int(row.get("page_id") or 0),
        revision_id=int(row.get("revision_id") or 0),
        revision_timestamp=str(row.get("revision_timestamp") or ""),
        retrieved_at=str(row.get("retrieved_at") or ""),
        full_text=str(row.get("full_text") or ""),
        full_text_format=str(row.get("full_text_format") or ""),
        article_length_chars=int(row.get("article_length_chars") or 0),
        article_length_words=int(row.get("article_length_words") or 0),
        article_length_tokens_estimate=int(row.get("article_length_tokens_estimate") or 0),
        license=str(row.get("license") or ""),
        attribution=str(row.get("attribution") or ""),
        source_api=str(row.get("source_api") or ""),
        fetch_status=str(row.get("fetch_status") or ""),
        fetch_error=str(row.get("fetch_error") or ""),
        content_hash=str(row.get("content_hash") or ""),
    )


def _build_missing_sections(
    documents: list[dict[str, Any]],
    existing_sections: list[dict[str, Any]],
    *,
    section_client: SectionClient | None,
    section_workers: int,
) -> list[dict[str, Any]]:
    """Fetch and parse only Wikipedia documents without persisted sections."""
    covered = {str(row.get("document_id")) for row in existing_sections if row.get("document_id")}
    missing = [
        row
        for row in sorted(documents, key=lambda item: str(item.get("document_id", "")))
        if row.get("project") == "wikipedia" and str(row.get("document_id")) not in covered
    ]
    if not missing:
        return existing_sections
    if section_client is None:
        raise ValueError("V2 section client is required to build missing Wikipedia sections")

    def fetch_one(row: dict[str, Any]) -> list[dict[str, Any]]:
        document = _section_document(row)
        html = section_client.parse_html(
            document.project,
            document.language,
            document.revision_id,
        )
        return [section.to_dict() for section in parse_sections(document, html)]

    with ThreadPoolExecutor(max_workers=max(1, section_workers)) as executor:
        fetched = executor.map(fetch_one, missing)
        new_sections = [row for group in fetched for row in group]
    by_id = {str(row["section_id"]): row for row in existing_sections}
    by_id.update({str(row["section_id"]): row for row in new_sections})
    return sorted(
        by_id.values(),
        key=lambda row: (
            str(row.get("document_id", "")),
            int(row.get("section_index", 0)),
            str(row.get("section_id", "")),
        ),
    )


def merge_v2_region(
    data_root: DataRoot,
    extracted: V2ExtractedPbf,
    *,
    index: Any,
    wikipedia_client: Any,
    cache: Any = None,
    fetch_full_text: bool = True,
    section_client: SectionClient | None = None,
    section_workers: int = 8,
    direct_workers: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Merge V1 rows with V2 discoveries and persist one canonical region."""
    stem = extracted.stem.stem
    v1 = load_v1_region(data_root, stem)
    polygons = {str(row["polygon_id"]): dict(row) for row in v1.polygons}
    for discovered in extracted.polygons:
        key = str(discovered["polygon_id"])
        existing = polygons.get(key)
        if existing is not None:
            if (
                existing.get("wikidata")
                and discovered.get("wikidata")
                and existing["wikidata"] != discovered["wikidata"]
            ):
                raise ValueError(f"Conflicting Wikidata values for polygon {key}")
            existing.update(
                {
                    name: discovered[name]
                    for name in (
                        "tags",
                        "tag_keys",
                        "tag_count",
                        "wikipedia_tag_refs",
                        "wikipedia_tag_rejections",
                        "discovery_sources",
                    )
                }
            )
        else:
            polygons[key] = dict(discovered)
    documents = {str(row["document_id"]): dict(row) for row in v1.documents}
    links = {(_link_key(row)): dict(row) for row in v1.links}
    direct_client = _cached_client(wikipedia_client, cache)
    direct_inputs: list[tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]]] = []
    for polygon_id, polygon in sorted(polygons.items()):
        raw_refs = json.loads(str(polygon.get("wikipedia_tag_refs", "[]")))
        refs = tuple(
            parse_wikipedia_tags({str(item.get("raw_key", "")): str(item.get("raw_value", ""))})[0]
            for item in raw_refs
        )
        refs = tuple(ref for group in refs for ref in group)
        if not refs:
            continue
        direct_inputs.append((polygon_id, polygon, refs))

    def enrich_one(
        item: tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]],
    ) -> DirectEnrichmentResult:
        polygon_id, polygon, refs = item
        return enrich_wikipedia_refs(
            polygon_id,
            refs,
            index=index,
            wikipedia_client=direct_client,
            polygon_context=polygon,
            cache=None,
            fetch_full_text=fetch_full_text,
        )

    if direct_inputs and direct_workers > 1:
        LOGGER.info(
            "V2 direct Wikipedia enrichment: %d polygon(s) with up to %d workers",
            len(direct_inputs),
            direct_workers,
        )
        workers = min(direct_workers, len(direct_inputs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque[Future[DirectEnrichmentResult]] = deque()
            inputs = iter(direct_inputs)
            for _ in range(workers):
                try:
                    pending.append(executor.submit(enrich_one, next(inputs)))
                except StopIteration:
                    break
            enriched = []
            while pending:
                enriched.append(pending.popleft().result())
                with contextlib.suppress(StopIteration):
                    pending.append(executor.submit(enrich_one, next(inputs)))
    else:
        enriched = [enrich_one(item) for item in direct_inputs]

    for direct in enriched:
        for document in direct.documents:
            documents[str(document["document_id"])] = dict(document)
        for link in direct.links:
            key = _link_key(link)
            if key in links:
                sources = set(json.loads(links[key].get("link_sources", "[]")))
                sources.update(json.loads(link.get("link_sources", "[]")))
                links[key]["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))
            else:
                links[key] = dict(link)

    links_by_polygon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in links.values():
        links_by_polygon[str(row["polygon_id"])].append(row)

    for polygon_id, polygon in polygons.items():
        rows = links_by_polygon.get(polygon_id, [])
        wikipedia_rows = [row for row in rows if row.get("project") == "wikipedia"]
        languages = sorted(
            {str(row.get("language", "")) for row in wikipedia_rows if row.get("language")}
        )
        # Keep the V1 field semantics: Wikivoyage relationships are part of
        # the unified link table, but do not make a polygon look like it has
        # a Wikipedia document.
        polygon["has_wikipedia"] = bool(wikipedia_rows)
        polygon["wikipedia_language_count"] = len(languages)
        polygon["wikipedia_languages"] = json.dumps(languages, separators=(",", ":"))
        polygon["wikipedia_article_count"] = len(wikipedia_rows)
        polygon["has_english_wikipedia"] = "en" in languages
        polygon["has_french_wikipedia"] = "fr" in languages
        polygon["text_available"] = any(
            bool(documents.get(str(row["document_id"]), {}).get("full_text"))
            for row in wikipedia_rows
        )
        if languages and not polygon.get("best_language"):
            polygon["best_language"] = languages[0]

    copy_v1_sidecars(data_root, stem, data_root.processed_v2)
    sections_path = data_root.processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet"
    sections = _rows(sections_path)
    sections = _build_missing_sections(
        list(documents.values()),
        sections,
        section_client=section_client,
        section_workers=section_workers,
    )
    write_v2_region(
        data_root.processed_v2,
        stem,
        polygons=sorted(polygons.values(), key=lambda row: str(row["polygon_id"])),
        documents=sorted(documents.values(), key=lambda row: str(row["document_id"])),
        links=sorted(links.values(), key=lambda row: _link_key(row)),
        sections=sections,
    )
    return tuple(polygons.values())


def _link_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("polygon_id", "")),
        str(row.get("project", "")),
        str(row.get("document_id", "")),
    )


__all__ = [
    "SIDECAR_SUBDIRS",
    "SectionClient",
    "V1RegionData",
    "copy_v1_sidecars",
    "load_v1_region",
    "merge_v2_region",
]
