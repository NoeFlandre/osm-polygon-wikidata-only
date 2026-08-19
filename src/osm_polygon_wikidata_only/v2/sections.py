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
        article_id=_string_field(row, "article_id"),
        wikidata=_string_field(row, "wikidata"),
        project=_string_field(row, "project", "wikipedia"),
        language=_string_field(row, "language"),
        site=_string_field(row, "site"),
        title=_string_field(row, "title"),
        url=_string_field(row, "url"),
        page_id=_int_field(row, "page_id"),
        revision_id=_int_field(row, "revision_id"),
        revision_timestamp=_string_field(row, "revision_timestamp"),
        retrieved_at=_string_field(row, "retrieved_at"),
        full_text=_string_field(row, "full_text"),
        full_text_format=_string_field(row, "full_text_format"),
        article_length_chars=_int_field(row, "article_length_chars"),
        article_length_words=_int_field(row, "article_length_words"),
        article_length_tokens_estimate=_int_field(row, "article_length_tokens_estimate"),
        license=_string_field(row, "license"),
        attribution=_string_field(row, "attribution"),
        source_api=_string_field(row, "source_api"),
        fetch_status=_string_field(row, "fetch_status"),
        fetch_error=_string_field(row, "fetch_error"),
        content_hash=_string_field(row, "content_hash"),
    )


def _string_field(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key)
    return str(value) if value else default


def _int_field(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    return int(value) if value else default


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
    missing = _missing_documents(documents, existing_sections, completed_document_ids)
    if not missing:
        return existing_sections
    if section_client is None:
        raise ValueError("V2 section client is required to build missing Wikipedia sections")
    new_sections = _fetch_missing_sections(
        missing,
        section_client=section_client,
        section_workers=section_workers,
        on_document=on_document,
    )
    return _merge_section_rows(existing_sections, new_sections)


def _missing_documents(
    documents: list[dict[str, Any]],
    existing_sections: list[dict[str, Any]],
    completed_document_ids: set[str] | None,
) -> list[dict[str, Any]]:
    covered = _covered_document_ids(existing_sections, completed_document_ids)
    missing: list[dict[str, Any]] = []
    for row in sorted(documents, key=lambda item: str(item.get("document_id", ""))):
        if _is_missing_document(row, covered):
            missing.append(row)
    return missing


def _covered_document_ids(
    existing_sections: list[dict[str, Any]],
    completed_document_ids: set[str] | None,
) -> set[str]:
    covered = {str(row.get("document_id")) for row in existing_sections if row.get("document_id")}
    covered.update(completed_document_ids or ())
    return covered


def _is_missing_document(row: dict[str, Any], covered: set[str]) -> bool:
    return bool(row.get("project") == "wikipedia" and str(row.get("document_id")) not in covered)


def _fetch_missing_sections(
    missing: list[dict[str, Any]],
    *,
    section_client: SectionClient,
    section_workers: int,
    on_document: Callable[[str, list[dict[str, Any]]], None] | None,
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=max(1, section_workers)) as executor:
        futures = {executor.submit(_fetch_one, row, section_client): row for row in missing}
        new_sections: list[dict[str, Any]] = []
        errors: list[Exception] = []
        for future in as_completed(futures):
            error = _record_section_future(
                future,
                futures[future],
                on_document,
                new_sections,
            )
            if error is not None:
                errors.append(error)
        if errors:
            raise errors[0]
    return new_sections


def _record_section_future(
    future: Any,
    row: dict[str, Any],
    on_document: Callable[[str, list[dict[str, Any]]], None] | None,
    new_sections: list[dict[str, Any]],
) -> Exception | None:
    result = _future_result(future)
    if isinstance(result, Exception):
        return result
    if on_document is not None:
        on_document(str(row["document_id"]), result)
    new_sections.extend(result)
    return None


def _fetch_one(row: dict[str, Any], section_client: SectionClient) -> list[dict[str, Any]]:
    document = _section_document(row)
    html = section_client.parse_html(
        document.project,
        document.language,
        document.revision_id,
    )
    return [section.to_dict() for section in parse_sections(document, html)]


def _future_result(future: Any) -> list[dict[str, Any]] | Exception:
    try:
        return future.result()
    except Exception as error:
        return error


def _merge_section_rows(
    existing_sections: list[dict[str, Any]],
    new_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
