"""Build deterministic Wikipedia section rows for the isolated V2 workflow."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

from osm_polygon_wikidata_only.augmentation.models import Document
from osm_polygon_wikidata_only.augmentation.sections import parse_sections


class SectionClient(Protocol):
    """Minimal exact-revision HTML client needed by the V2 section builder."""

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        """Return rendered HTML for one immutable Wikimedia revision."""


def _section_document(row: dict[str, Any]) -> Document:
    """Adapt a V2 document row to the shared section parser model."""
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


def build_missing_sections(
    documents: list[dict[str, Any]],
    existing_sections: list[dict[str, Any]],
    *,
    section_client: SectionClient | None,
    section_workers: int,
    on_document: Callable[[str, list[dict[str, Any]]], None] | None = None,
    completed_document_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch and parse only Wikipedia documents without persisted sections."""
    covered = {str(row.get("document_id")) for row in existing_sections if row.get("document_id")}
    covered.update(completed_document_ids or ())
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
        futures = {executor.submit(fetch_one, row): row for row in missing}
        new_sections: list[dict[str, Any]] = []
        first_error: Exception | None = None
        for future in as_completed(futures):
            try:
                fetched = future.result()
            except Exception as error:
                first_error = first_error or error
                continue
            if on_document is not None:
                on_document(str(futures[future]["document_id"]), fetched)
            new_sections.extend(fetched)
        if first_error is not None:
            raise first_error
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


__all__ = ["SectionClient", "build_missing_sections"]
