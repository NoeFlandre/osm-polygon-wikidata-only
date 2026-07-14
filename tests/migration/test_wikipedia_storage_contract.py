"""Contract tests freezing schemas, columns, types, and mappings for unification."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row, document_id
from osm_polygon_wikidata_only.augmentation.schema import DOCUMENT_COLUMNS, document_schema
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS, article_schema

EXPECTED_ARTICLE_COLUMNS = (
    "article_id",
    "wikidata",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "retrieved_at",
    "wikidata_label",
    "wikidata_description",
    "wikidata_aliases",
    "lead_text",
    "extract",
    "full_text",
    "full_text_format",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
    "thumbnail_url",
    "thumbnail_width",
    "thumbnail_height",
    "categories",
    "license",
    "attribution",
    "source_api",
    "fetch_status",
    "fetch_error",
    "content_hash",
)

EXPECTED_DOCUMENT_COLUMNS = (
    "document_id",
    "article_id",
    "wikidata",
    "project",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "retrieved_at",
    "full_text",
    "full_text_format",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
    "license",
    "attribution",
    "source_api",
    "fetch_status",
    "fetch_error",
    "content_hash",
)

EXPECTED_SHARED_COLUMNS = {
    "article_id",
    "wikidata",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "retrieved_at",
    "full_text",
    "full_text_format",
    "article_length_chars",
    "article_length_words",
    "article_length_tokens_estimate",
    "license",
    "attribution",
    "source_api",
    "fetch_status",
    "fetch_error",
    "content_hash",
}

EXPECTED_ARTICLE_ONLY = {
    "wikidata_label",
    "wikidata_description",
    "wikidata_aliases",
    "lead_text",
    "extract",
    "thumbnail_url",
    "thumbnail_width",
    "thumbnail_height",
    "categories",
}

EXPECTED_DOCUMENT_ONLY = {
    "document_id",
    "project",
}


def test_article_columns_contract() -> None:
    assert ARTICLE_COLUMNS == EXPECTED_ARTICLE_COLUMNS


def test_document_columns_contract() -> None:
    assert DOCUMENT_COLUMNS == EXPECTED_DOCUMENT_COLUMNS


def test_shared_columns_set() -> None:
    shared = set(ARTICLE_COLUMNS) & set(DOCUMENT_COLUMNS)
    assert shared == EXPECTED_SHARED_COLUMNS


def test_article_only_columns_set() -> None:
    article_only = set(ARTICLE_COLUMNS) - set(DOCUMENT_COLUMNS)
    assert article_only == EXPECTED_ARTICLE_ONLY


def test_canonical_upgrade_columns_contract() -> None:
    from tests.migration.audit import CANONICAL_UPGRADE_COLUMNS

    actual_upgrade = tuple(column for column in ARTICLE_COLUMNS if column not in DOCUMENT_COLUMNS)
    assert actual_upgrade == CANONICAL_UPGRADE_COLUMNS


def test_document_only_columns_set() -> None:
    document_only = set(DOCUMENT_COLUMNS) - set(ARTICLE_COLUMNS)
    assert document_only == EXPECTED_DOCUMENT_ONLY


def test_shared_pyarrow_types() -> None:
    art_s = article_schema()
    doc_s = document_schema()

    for col in EXPECTED_SHARED_COLUMNS:
        art_type = art_s.field(col).type
        doc_type = doc_s.field(col).type
        assert art_type == doc_type, f"Type mismatch for '{col}': {art_type} vs {doc_type}"

    # Specifically check important types
    assert art_s.field("page_id").type == pa.int64()
    assert art_s.field("revision_id").type == pa.int64()
    assert art_s.field("full_text").type == pa.string()


def test_document_from_article_row_mapping() -> None:
    # A dummy article row containing distinct values for each field to verify exact copying
    dummy_row = {
        "article_id": "Q123:en:456:789",
        "wikidata": "Q123",
        "language": "en",
        "site": "enwiki",
        "title": "Test Title",
        "url": "https://en.wikipedia.org/wiki/Test_Title",
        "page_id": 456,
        "revision_id": 789,
        "revision_timestamp": "2026-07-14T00:00:00Z",
        "retrieved_at": "2026-07-14T12:00:00Z",
        "wikidata_label": "Label Q123",
        "wikidata_description": "Description Q123",
        "wikidata_aliases": '["Alias1"]',
        "lead_text": "Lead text content.",
        "extract": "Extract content.",
        "full_text": "Full text content of the article.",
        "full_text_format": "plain_text",
        "article_length_chars": 33,
        "article_length_words": 6,
        "article_length_tokens_estimate": 8,
        "thumbnail_url": "https://example.com/thumb.jpg",
        "thumbnail_width": 200,
        "thumbnail_height": 150,
        "categories": '["Category1"]',
        "license": "CC-BY-SA",
        "attribution": "Contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "dummyhash123",
    }

    doc = document_from_article_row(dummy_row)

    # Check document_id derivation
    assert doc.document_id == "Q123:wikipedia:en:456:789"
    assert doc.document_id == document_id("Q123", "wikipedia", "en", 456, 789)

    # Check project column
    assert doc.project == "wikipedia"

    # Check article_id preservation
    assert doc.article_id == "Q123:en:456:789"

    # Check shared columns
    for col in EXPECTED_SHARED_COLUMNS:
        val_in_row = dummy_row[col]
        val_in_doc = getattr(doc, col)
        assert val_in_doc == val_in_row, (
            f"Shared column '{col}' value mismatch: {val_in_doc} vs {val_in_row}"
        )
