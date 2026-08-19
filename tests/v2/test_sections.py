from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.v2 import reuse
from osm_polygon_wikidata_only.v2.sections import (
    SectionClient,
    _string_field,
    build_missing_sections,
)


def _document(document_id: str, *, project: str = "wikipedia") -> dict[str, Any]:
    return {
        "document_id": document_id,
        "article_id": document_id,
        "wikidata": "Q42",
        "project": project,
        "language": "en",
        "site": "enwiki",
        "title": document_id,
        "url": "https://en.wikipedia.org/wiki/Example",
        "page_id": 1,
        "revision_id": 2,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "full_text": "Example text",
        "full_text_format": "plain",
        "article_length_chars": 12,
        "article_length_words": 2,
        "article_length_tokens_estimate": 3,
        "license": "CC BY-SA",
        "attribution": "Wikimedia contributors",
        "source_api": "test",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "hash",
    }


def test_section_builder_preserves_existing_rows_and_checkpoint_order() -> None:
    assert hasattr(SectionClient, "parse_html")
    existing = {
        "section_id": "existing",
        "document_id": "already",
        "section_index": 0,
    }
    saved: list[tuple[str, list[dict[str, Any]]]] = []

    class Client:
        def parse_html(self, project: str, language: str, revision_id: int) -> str:
            assert (project, language, revision_id) == ("wikipedia", "en", 2)
            return "<p>Fetched text</p><h2>Details</h2><p>More text</p>"

    result = build_missing_sections(
        [_document("already"), _document("missing")],
        [existing],
        section_client=Client(),
        section_workers=1,
        on_document=lambda document_id, rows: saved.append((document_id, rows)),
    )

    assert result[0]["section_id"] == "existing"
    assert [row["document_id"] for row in result[1:]] == ["missing", "missing"]
    assert [row["section_index"] for row in result[1:]] == [0, 1]
    assert saved and saved[0][0] == "missing"


def test_section_builder_does_not_fetch_non_wikipedia_documents() -> None:
    class Client:
        def parse_html(self, *_args: object) -> str:
            raise AssertionError("Wikivoyage rows must not use the Wikipedia section client")

    result = build_missing_sections(
        [_document("voyage", project="wikivoyage")],
        [],
        section_client=Client(),
        section_workers=1,
    )

    assert result == []


def test_reuse_module_preserves_section_compatibility_exports() -> None:
    assert reuse.SectionClient is SectionClient
    assert reuse._build_missing_sections is build_missing_sections


def test_string_field_preserves_empty_value_defaults() -> None:
    assert _string_field({"value": None}, "value") == ""
    assert _string_field({"value": 0}, "value") == ""
    assert _string_field({"value": "kept"}, "value") == "kept"
