"""Comprehensive tests for the canonical Wikipedia document contract.

Covers:
- Schema structure, metadata preservation, and legacy compatibility
- Strict type validation (no coercion)
- Identity validation (QID, language whitespace checks, positive IDs)
- Extra-key rejection
- Table-level validation (missing/extra/wrong/reordered/duplicate columns, metadata validation)
- Phase 1 audit compatibility
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.models import (
    Document,
    document_from_article_row,
    document_id,
)
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    document_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    WikipediaDocumentConversionError,
    build_wikipedia_document_table,
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    article_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article_row(
    wikidata: str = "Q235",
    language: str = "en",
    page_id: int = 3649,
    revision_id: int = 1234567,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a complete article row with deterministic defaults."""
    article_id = f"{wikidata}:{language}:{page_id}:{revision_id}"
    row: dict[str, Any] = {
        "article_id": article_id,
        "wikidata": wikidata,
        "language": language,
        "site": f"{language}wiki",
        "title": "Monaco",
        "url": f"https://{language}.wikipedia.org/wiki/Monaco",
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": "2026-01-15T10:30:00Z",
        "retrieved_at": "2026-07-14T12:00:00Z",
        "wikidata_label": "Monaco",
        "wikidata_description": "Sovereign city-state and microstate",
        "wikidata_aliases": '["Principality of Monaco", "MCO"]',
        "lead_text": "Monaco, officially the Principality of Monaco...",
        "extract": "Monaco is a sovereign city-state.",
        "full_text": "Monaco, officially the Principality of Monaco, is a sovereign city-state.",
        "full_text_format": "plain_text",
        "article_length_chars": 71,
        "article_length_words": 11,
        "article_length_tokens_estimate": 17,
        "thumbnail_url": "https://upload.wikimedia.org/thumb.jpg",
        "thumbnail_width": 300,
        "thumbnail_height": 200,
        "categories": '["Microstates", "City-states"]',
        "license": "CC BY-SA",
        "attribution": "Wikipedia contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "abc123def456",
    }
    row.update(overrides)
    return row


def _make_article_table(rows: list[dict[str, Any]] | None = None) -> pa.Table:
    """Build a PyArrow Table from article rows."""
    if rows is None:
        rows = [_make_article_row()]
    return pa.Table.from_pylist(rows, schema=article_schema())


# ===========================================================================
# Schema Tests
# ===========================================================================


class TestCanonicalSchema:
    """Tests for the canonical 32-column schema definition."""

    def test_exact_32_column_ordered_schema(self) -> None:
        assert len(WIKIPEDIA_DOCUMENT_COLUMNS) == 32
        assert WIKIPEDIA_DOCUMENT_COLUMNS == (
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

    def test_removing_document_id_and_project_recovers_article_columns(self) -> None:
        filtered = tuple(
            col for col in WIKIPEDIA_DOCUMENT_COLUMNS if col not in ("document_id", "project")
        )
        assert filtered == ARTICLE_COLUMNS

    def test_every_inherited_field_type_and_metadata_matches_article(self) -> None:
        """Each inherited field must preserve both type AND metadata."""
        art_schema = article_schema()
        wp_schema = wikipedia_document_schema()
        for col in ARTICLE_COLUMNS:
            art_field = art_schema.field(col)
            wp_field = wp_schema.field(col)
            assert art_field.type == wp_field.type, (
                f"Type mismatch for '{col}': article={art_field.type}, "
                f"wikipedia_doc={wp_field.type}"
            )
            assert art_field.metadata == wp_field.metadata, (
                f"Metadata mismatch for '{col}': article={art_field.metadata}, "
                f"wikipedia_doc={wp_field.metadata}"
            )

    def test_document_id_and_project_are_strings(self) -> None:
        wp_schema = wikipedia_document_schema()
        assert wp_schema.field("document_id").type == pa.string()
        assert wp_schema.field("project").type == pa.string()

    def test_document_id_has_description_metadata(self) -> None:
        wp_schema = wikipedia_document_schema()
        meta = wp_schema.field("document_id").metadata
        assert meta is not None
        assert b"description" in meta
        desc = meta[b"description"].decode()
        assert "wikidata" in desc.lower() or "document" in desc.lower()

    def test_project_has_description_metadata(self) -> None:
        wp_schema = wikipedia_document_schema()
        meta = wp_schema.field("project").metadata
        assert meta is not None
        assert b"description" in meta
        desc = meta[b"description"].decode()
        assert "wikipedia" in desc.lower()

    def test_schema_has_32_fields(self) -> None:
        wp_schema = wikipedia_document_schema()
        assert len(wp_schema) == 32

    def test_schema_field_order_matches_columns(self) -> None:
        wp_schema = wikipedia_document_schema()
        assert tuple(wp_schema.names) == WIKIPEDIA_DOCUMENT_COLUMNS

    def test_existing_document_columns_unchanged(self) -> None:
        assert DOCUMENT_COLUMNS == (
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

    def test_existing_document_schema_unchanged(self) -> None:
        ds = document_schema()
        assert len(ds) == 23
        assert tuple(ds.names) == DOCUMENT_COLUMNS

    def test_parquet_round_trip_preserves_complete_schema(self, tmp_path: Path) -> None:
        """Parquet round-trip must preserve field types AND metadata."""
        table = _make_article_table()
        result = build_wikipedia_document_table(table)
        path = tmp_path / "schema_test.parquet"
        pq.write_table(result, path)
        loaded = pq.read_table(path)
        expected = wikipedia_document_schema()
        assert loaded.schema.equals(expected), (
            f"Schema mismatch after round-trip:\n"
            f"  expected: {expected}\n"
            f"  got:      {loaded.schema}"
        )


# ===========================================================================
# Row Conversion Tests — Value Preservation
# ===========================================================================


class TestRowConversion:
    """Tests for wikipedia_document_from_article_row()."""

    def test_all_30_article_values_preserved(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        for col in ARTICLE_COLUMNS:
            actual = getattr(doc, col)
            expected = row[col]
            assert actual is expected or actual == expected, (
                f"Field '{col}' not preserved: {actual!r} != {expected!r}"
            )

    def test_nine_previously_omitted_fields_preserved(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        assert doc.wikidata_label == "Monaco"
        assert doc.wikidata_description == "Sovereign city-state and microstate"
        assert doc.wikidata_aliases == '["Principality of Monaco", "MCO"]'
        assert doc.lead_text == "Monaco, officially the Principality of Monaco..."
        assert doc.extract == "Monaco is a sovereign city-state."
        assert doc.thumbnail_url == "https://upload.wikimedia.org/thumb.jpg"
        assert doc.thumbnail_width == 300
        assert doc.thumbnail_height == 200
        assert doc.categories == '["Microstates", "City-states"]'

    def test_none_thumbnail_dimensions_remain_none(self) -> None:
        row = _make_article_row(thumbnail_width=None, thumbnail_height=None)
        doc = wikipedia_document_from_article_row(row)
        assert doc.thumbnail_width is None
        assert doc.thumbnail_height is None

    def test_empty_strings_remain_empty(self) -> None:
        row = _make_article_row(
            fetch_error="",
            thumbnail_url="",
            wikidata_label="",
            lead_text="",
        )
        doc = wikipedia_document_from_article_row(row)
        assert doc.fetch_error == ""
        assert doc.thumbnail_url == ""
        assert doc.wikidata_label == ""
        assert doc.lead_text == ""

    def test_whitespace_remains_unchanged(self) -> None:
        row = _make_article_row(
            title="  Monaco  ",
            full_text="Line 1\n\tLine 2  \n",
        )
        doc = wikipedia_document_from_article_row(row)
        assert doc.title == "  Monaco  "
        assert doc.full_text == "Line 1\n\tLine 2  \n"

    def test_unicode_and_multilingual_text(self) -> None:
        row = _make_article_row(
            language="ja",
            title="モナコ",
            wikidata_label="モナコ公国",
            full_text="モナコ公国（モナコこうこく）は、西ヨーロッパの立憲君主制国家。",  # noqa: RUF001
            wikidata_description="ヨーロッパの都市国家",
            lead_text="モナコ公国は...",
        )
        doc = wikipedia_document_from_article_row(row)
        assert doc.title == "モナコ"
        assert doc.wikidata_label == "モナコ公国"
        assert doc.full_text == "モナコ公国（モナコこうこく）は、西ヨーロッパの立憲君主制国家。"  # noqa: RUF001

    def test_article_id_unchanged(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        assert doc.article_id == row["article_id"]

    def test_correct_document_id(self) -> None:
        row = _make_article_row(wikidata="Q235", language="en", page_id=3649, revision_id=1234567)
        doc = wikipedia_document_from_article_row(row)
        assert doc.document_id == "Q235:wikipedia:en:3649:1234567"

    def test_constant_wikipedia_project(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        assert doc.project == "wikipedia"

    def test_input_mapping_unchanged_after_conversion(self) -> None:
        row = _make_article_row()
        original = copy.deepcopy(row)
        wikipedia_document_from_article_row(row)
        assert row == original

    def test_frozen_mapping_input_accepted(self) -> None:
        """Conversion must accept any Mapping, not just dict."""
        from types import MappingProxyType

        row = _make_article_row()
        frozen = MappingProxyType(row)
        doc = wikipedia_document_from_article_row(frozen)
        assert doc.project == "wikipedia"

    def test_to_dict_canonical_order(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        d = doc.to_dict()
        assert tuple(d.keys()) == WIKIPEDIA_DOCUMENT_COLUMNS

    def test_model_is_frozen(self) -> None:
        row = _make_article_row()
        doc = wikipedia_document_from_article_row(row)
        with pytest.raises(AttributeError):
            doc.title = "changed"  # type: ignore[misc]


# ===========================================================================
# Row Conversion Tests — Strict Type Rejection (No Coercion)
# ===========================================================================


class TestStrictTypeRejection:
    """Prove that numeric strings, booleans, floats, and arbitrary objects are rejected."""

    def test_string_page_id_rejected(self) -> None:
        row = _make_article_row(page_id="3649")
        with pytest.raises(WikipediaDocumentConversionError, match="page_id"):
            wikipedia_document_from_article_row(row)

    def test_string_revision_id_rejected(self) -> None:
        row = _make_article_row(revision_id="1234567")
        with pytest.raises(WikipediaDocumentConversionError, match="revision_id"):
            wikipedia_document_from_article_row(row)

    def test_string_thumbnail_width_rejected(self) -> None:
        row = _make_article_row(thumbnail_width="300")
        with pytest.raises(WikipediaDocumentConversionError, match="thumbnail_width"):
            wikipedia_document_from_article_row(row)

    def test_string_thumbnail_height_rejected(self) -> None:
        row = _make_article_row(thumbnail_height="200")
        with pytest.raises(WikipediaDocumentConversionError, match="thumbnail_height"):
            wikipedia_document_from_article_row(row)

    def test_invalid_thumbnail_text_rejected(self) -> None:
        row = _make_article_row(thumbnail_width="not_a_number")
        with pytest.raises(WikipediaDocumentConversionError, match="thumbnail_width"):
            wikipedia_document_from_article_row(row)

    def test_float_integer_field_rejected(self) -> None:
        row = _make_article_row(page_id=3649.0)
        with pytest.raises(WikipediaDocumentConversionError, match="page_id"):
            wikipedia_document_from_article_row(row)

    def test_float_article_length_rejected(self) -> None:
        row = _make_article_row(article_length_chars=71.0)
        with pytest.raises(WikipediaDocumentConversionError, match="article_length_chars"):
            wikipedia_document_from_article_row(row)

    def test_boolean_integer_field_rejected(self) -> None:
        row = _make_article_row(page_id=True)
        with pytest.raises(WikipediaDocumentConversionError, match="page_id"):
            wikipedia_document_from_article_row(row)

    def test_boolean_thumbnail_dimension_rejected(self) -> None:
        row = _make_article_row(thumbnail_width=True)
        with pytest.raises(WikipediaDocumentConversionError, match="thumbnail_width"):
            wikipedia_document_from_article_row(row)

    def test_integer_for_string_field_rejected(self) -> None:
        row = _make_article_row(title=12345)
        with pytest.raises(WikipediaDocumentConversionError, match="title"):
            wikipedia_document_from_article_row(row)

    def test_arbitrary_object_for_string_field_rejected(self) -> None:
        row = _make_article_row(title=["a", "list"])
        with pytest.raises(WikipediaDocumentConversionError, match="title"):
            wikipedia_document_from_article_row(row)

    def test_none_for_required_string_rejected(self) -> None:
        row = _make_article_row(title=None)
        with pytest.raises(WikipediaDocumentConversionError, match="title"):
            wikipedia_document_from_article_row(row)

    def test_none_for_required_int_rejected(self) -> None:
        row = _make_article_row(article_length_chars=None)
        with pytest.raises(WikipediaDocumentConversionError, match="article_length_chars"):
            wikipedia_document_from_article_row(row)


# ===========================================================================
# Row Conversion Tests — Identity Validation
# ===========================================================================


class TestIdentityValidation:
    """Prove strict identity validation for QID, language, and positive IDs."""

    @pytest.mark.parametrize("qid", ["Q1", "Q42", "Q9999999"])
    def test_shared_qid_validator_accepts_valid(self, qid: str) -> None:
        row = _make_article_row(wikidata=qid)
        doc = wikipedia_document_from_article_row(row)
        assert doc.wikidata == qid

    @pytest.mark.parametrize(
        "qid",
        ["", "Q0", "Q-1", "q42", "Q42a", "Q 42", "P42", "X1", "Q"],
    )
    def test_shared_qid_validator_rejects_invalid(self, qid: str) -> None:
        row = _make_article_row(wikidata=qid)
        with pytest.raises(WikipediaDocumentConversionError):
            wikipedia_document_from_article_row(row)

    def test_empty_wikidata_rejected(self) -> None:
        row = _make_article_row(wikidata="")
        with pytest.raises(WikipediaDocumentConversionError, match="wikidata"):
            wikipedia_document_from_article_row(row)

    def test_q0_rejected(self) -> None:
        row = _make_article_row(wikidata="Q0")
        with pytest.raises(WikipediaDocumentConversionError, match="Q0"):
            wikipedia_document_from_article_row(row)

    def test_lowercase_qid_rejected(self) -> None:
        row = _make_article_row(wikidata="q123")
        with pytest.raises(WikipediaDocumentConversionError, match="q123"):
            wikipedia_document_from_article_row(row)

    def test_arbitrary_wikidata_text_rejected(self) -> None:
        row = _make_article_row(wikidata="not_a_qid")
        with pytest.raises(WikipediaDocumentConversionError, match="not_a_qid"):
            wikipedia_document_from_article_row(row)

    def test_empty_language_rejected(self) -> None:
        row = _make_article_row(language="")
        with pytest.raises(WikipediaDocumentConversionError, match="language"):
            wikipedia_document_from_article_row(row)

    def test_language_with_colon_rejected(self) -> None:
        row = _make_article_row(language="en:fr")
        with pytest.raises(WikipediaDocumentConversionError, match="language"):
            wikipedia_document_from_article_row(row)

    @pytest.mark.parametrize(
        "lang",
        ["", " ", " en", "en ", "en\n", "en\t", " en "],
    )
    def test_whitespace_language_rejected(self, lang: str) -> None:
        row = _make_article_row(language=lang)
        with pytest.raises(WikipediaDocumentConversionError, match="language"):
            wikipedia_document_from_article_row(row)

    def test_hyphenated_language_accepted(self) -> None:
        """zh-min-nan and similar hyphenated language codes must be valid."""
        row = _make_article_row(language="zh-min-nan")
        doc = wikipedia_document_from_article_row(row)
        assert doc.language == "zh-min-nan"

    def test_non_string_language_rejected(self) -> None:
        row = _make_article_row(language=123)
        with pytest.raises(WikipediaDocumentConversionError, match="language"):
            wikipedia_document_from_article_row(row)

    def test_zero_page_id_rejected(self) -> None:
        row = _make_article_row(page_id=0)
        with pytest.raises(WikipediaDocumentConversionError, match="page_id"):
            wikipedia_document_from_article_row(row)

    def test_negative_page_id_rejected(self) -> None:
        row = _make_article_row(page_id=-1)
        with pytest.raises(WikipediaDocumentConversionError, match="page_id"):
            wikipedia_document_from_article_row(row)

    def test_zero_revision_id_rejected(self) -> None:
        row = _make_article_row(revision_id=0)
        with pytest.raises(WikipediaDocumentConversionError, match="revision_id"):
            wikipedia_document_from_article_row(row)

    def test_negative_revision_id_rejected(self) -> None:
        row = _make_article_row(revision_id=-1)
        with pytest.raises(WikipediaDocumentConversionError, match="revision_id"):
            wikipedia_document_from_article_row(row)

    def test_inconsistent_article_id_rejected(self) -> None:
        row = _make_article_row()
        row["article_id"] = "WRONG:id:0:0"
        with pytest.raises(WikipediaDocumentConversionError, match="article_id"):
            wikipedia_document_from_article_row(row)

    def test_null_wikidata_rejected(self) -> None:
        row = _make_article_row(wikidata=None)
        with pytest.raises(WikipediaDocumentConversionError, match="wikidata"):
            wikipedia_document_from_article_row(row)

    def test_missing_required_field_fails(self) -> None:
        row = _make_article_row()
        del row["wikidata"]
        with pytest.raises(WikipediaDocumentConversionError, match="wikidata"):
            wikipedia_document_from_article_row(row)


# ===========================================================================
# Row Conversion Tests — Extra Keys
# ===========================================================================


class TestExtraKeyRejection:
    """Prove unknown extra keys are rejected for migration safety."""

    def test_extra_key_rejected(self) -> None:
        row = _make_article_row()
        row["unknown_field"] = "surprise"
        with pytest.raises(WikipediaDocumentConversionError, match="unknown_field"):
            wikipedia_document_from_article_row(row)

    def test_multiple_extra_keys_reported(self) -> None:
        row = _make_article_row()
        row["extra_a"] = 1
        row["extra_b"] = 2
        with pytest.raises(WikipediaDocumentConversionError, match="extra"):
            wikipedia_document_from_article_row(row)


# ===========================================================================
# Table Conversion Tests
# ===========================================================================


class TestTableConversion:
    """Tests for build_wikipedia_document_table()."""

    def test_multiple_rows_converted_correctly(self) -> None:
        rows = [
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
            _make_article_row(wikidata="Q3", page_id=3, revision_id=300),
        ]
        table = _make_article_table(rows)
        result = build_wikipedia_document_table(table)
        assert result.num_rows == 3

    def test_output_sorted_by_document_id(self) -> None:
        rows = [
            _make_article_row(wikidata="Q3", page_id=3, revision_id=300),
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
        ]
        table = _make_article_table(rows)
        result = build_wikipedia_document_table(table)
        doc_ids = result.column("document_id").to_pylist()
        assert doc_ids == sorted(doc_ids)

    def test_input_ordering_does_not_affect_output(self) -> None:
        rows_a = [
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
        ]
        rows_b = [
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
        ]
        result_a = build_wikipedia_document_table(_make_article_table(rows_a))
        result_b = build_wikipedia_document_table(_make_article_table(rows_b))
        assert result_a.equals(result_b)

    def test_duplicate_article_id_rejected(self) -> None:
        rows = [
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
        ]
        table = _make_article_table(rows)
        with pytest.raises(WikipediaDocumentConversionError, match=r"[Dd]uplicate.*article_id"):
            build_wikipedia_document_table(table)

    def test_multiple_duplicate_article_ids_deterministic(self) -> None:
        """Several duplicate IDs must produce deterministic diagnostics."""
        rows = [
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
            _make_article_row(wikidata="Q2", page_id=2, revision_id=200),
            _make_article_row(wikidata="Q3", page_id=3, revision_id=300),
        ]
        table = _make_article_table(rows)
        with pytest.raises(WikipediaDocumentConversionError) as exc_info:
            build_wikipedia_document_table(table)
        msg = str(exc_info.value)
        # Both duplicate IDs must be mentioned, in sorted order
        assert "Q1:en:1:100" in msg
        assert "Q2:en:2:200" in msg

    def test_empty_typed_table_supported(self) -> None:
        table = pa.Table.from_pylist([], schema=article_schema())
        result = build_wikipedia_document_table(table)
        assert result.num_rows == 0
        assert result.schema.equals(wikipedia_document_schema())

    def test_missing_column_rejected(self) -> None:
        schema = pa.schema([f for f in article_schema() if f.name != "wikidata"])
        rows = [_make_article_row()]
        del rows[0]["wikidata"]
        table = pa.Table.from_pylist(rows, schema=schema)
        with pytest.raises(WikipediaDocumentConversionError, match="wikidata"):
            build_wikipedia_document_table(table)

    def test_wrong_type_rejected(self) -> None:
        wrong_schema = pa.schema(
            [
                pa.field("page_id", pa.string()) if f.name == "page_id" else f
                for f in article_schema()
            ]
        )
        row = _make_article_row()
        row["page_id"] = "not_a_number"
        table = pa.Table.from_pylist([row], schema=wrong_schema)
        with pytest.raises(WikipediaDocumentConversionError):
            build_wikipedia_document_table(table)

    def test_duplicate_column_names_rejected(self) -> None:
        """Duplicate column names in input table must be rejected."""
        wrong_schema = pa.schema(
            [
                pa.field("article_id", pa.string()),
                pa.field("article_id", pa.string()),
            ]
        )
        table = pa.Table.from_arrays(
            [pa.array(["Q1:en:1:100"]), pa.array(["Q1:en:1:100"])], schema=wrong_schema
        )
        with pytest.raises(WikipediaDocumentConversionError, match="Duplicate column name"):
            build_wikipedia_document_table(table)

    def test_unknown_extra_column_rejected(self) -> None:
        """Extra columns in the input table must be rejected."""
        art = article_schema()
        extended = pa.schema([*art, pa.field("surprise_col", pa.string())])
        row = _make_article_row()
        row["surprise_col"] = "oops"
        table = pa.Table.from_pylist([row], schema=extended)
        with pytest.raises(WikipediaDocumentConversionError, match="surprise_col"):
            build_wikipedia_document_table(table)

    def test_wrong_field_ordering_rejected(self) -> None:
        """Input table with reordered columns must be rejected."""
        art = article_schema()
        fields = list(art)
        # Swap first two fields
        fields[0], fields[1] = fields[1], fields[0]
        reordered = pa.schema(fields)
        row = _make_article_row()
        table = pa.Table.from_pylist([row], schema=reordered)
        with pytest.raises(WikipediaDocumentConversionError, match="order"):
            build_wikipedia_document_table(table)

    def test_input_table_missing_metadata_rejected(self) -> None:
        """If one field in the input table is missing expected metadata, it must be rejected."""
        art = article_schema()
        fields = list(art)
        idx = next(i for i, f in enumerate(fields) if f.name == "page_id")
        fields[idx] = pa.field("page_id", pa.int64())  # no metadata
        wrong_schema = pa.schema(fields)
        row = _make_article_row()
        table = pa.Table.from_pylist([row], schema=wrong_schema)
        with pytest.raises(WikipediaDocumentConversionError, match=r"Metadata mismatch.*page_id"):
            build_wikipedia_document_table(table)

    def test_input_table_incorrect_metadata_rejected(self) -> None:
        """If one field in the input table has incorrect description metadata, it must be rejected."""
        art = article_schema()
        fields = list(art)
        idx = next(i for i, f in enumerate(fields) if f.name == "title")
        fields[idx] = pa.field("title", pa.string(), metadata={"description": "wrong desc"})
        wrong_schema = pa.schema(fields)
        row = _make_article_row()
        table = pa.Table.from_pylist([row], schema=wrong_schema)
        with pytest.raises(WikipediaDocumentConversionError, match=r"Metadata mismatch.*title"):
            build_wikipedia_document_table(table)

    def test_input_table_extra_metadata_rejected(self) -> None:
        """If one field in the input table has extra metadata keys, it must be rejected."""
        art = article_schema()
        fields = list(art)
        idx = next(i for i, f in enumerate(fields) if f.name == "title")
        meta = dict(art.field(idx).metadata) if art.field(idx).metadata else {}
        meta[b"extra_key"] = b"some_val"
        fields[idx] = pa.field("title", pa.string(), metadata=meta)
        wrong_schema = pa.schema(fields)
        row = _make_article_row()
        table = pa.Table.from_pylist([row], schema=wrong_schema)
        with pytest.raises(WikipediaDocumentConversionError, match=r"Metadata mismatch.*title"):
            build_wikipedia_document_table(table)

    def test_input_table_correct_metadata_passes(self) -> None:
        """If metadata exactly matches, table conversion must succeed."""
        table = _make_article_table()
        # Should succeed without error
        result = build_wikipedia_document_table(table)
        assert result.num_rows == 1

    def test_exact_output_schema(self) -> None:
        table = _make_article_table()
        result = build_wikipedia_document_table(table)
        assert result.schema.equals(wikipedia_document_schema())

    def test_parquet_round_trip(self, tmp_path: Path) -> None:
        table = _make_article_table()
        result = build_wikipedia_document_table(table)
        path = tmp_path / "test.parquet"
        pq.write_table(result, path)
        loaded = pq.read_table(path)
        assert loaded.equals(result)
        assert loaded.schema.equals(wikipedia_document_schema())

    def test_two_conversions_of_identical_input_are_equal(self) -> None:
        table = _make_article_table()
        result_1 = build_wikipedia_document_table(table)
        result_2 = build_wikipedia_document_table(table)
        assert result_1.equals(result_2)

    def test_phase1_audit_classifies_as_both_equivalent(self, tmp_path: Path) -> None:
        """Resulting table must be classified as both_equivalent by Phase 1 audit."""
        from tests.migration.audit import run_audit

        # Setup directories
        processed = tmp_path / "processed"
        art_dir = processed / "articles"
        doc_dir = processed / "wikipedia" / "documents"
        art_dir.mkdir(parents=True)
        doc_dir.mkdir(parents=True)

        # Write article table
        rows = [_make_article_row(wikidata="Q1", page_id=1, revision_id=100)]
        art_table = _make_article_table(rows)
        pq.write_table(art_table, art_dir / "test-stem.parquet")

        # Convert and write canonical document table
        doc_table = build_wikipedia_document_table(art_table)
        pq.write_table(doc_table, doc_dir / "test-stem.parquet")

        # Run audit
        report = run_audit(tmp_path)
        stem = report["per_stem"]["test-stem"]
        assert stem["state"] == "both_equivalent", (
            f"Expected both_equivalent, got {stem['state']}. Discrepancies: {stem['discrepancies']}"
        )

    def test_values_preserved_in_output_table(self) -> None:
        row = _make_article_row(
            thumbnail_width=None,
            thumbnail_height=None,
            wikidata_label="Label",
            lead_text="  whitespace  ",
        )
        table = _make_article_table([row])
        result = build_wikipedia_document_table(table)
        out_row = result.to_pylist()[0]
        assert out_row["thumbnail_width"] is None
        assert out_row["thumbnail_height"] is None
        assert out_row["wikidata_label"] == "Label"
        assert out_row["lead_text"] == "  whitespace  "
        assert out_row["project"] == "wikipedia"
        assert out_row["document_id"] == "Q235:wikipedia:en:3649:1234567"


# ===========================================================================
# Duplicate Document ID Invariant
# ===========================================================================


class TestDuplicateDocumentIdInvariant:
    """The duplicate-document_id check is a defensive invariant.

    With strict identity validation (valid QID without ':', non-empty language
    without ':', positive page_id and revision_id), two valid unique
    article_ids cannot generate the same document_id.

    article_id = f"{wikidata}:{language}:{page_id}:{revision_id}"
    document_id = f"{wikidata}:wikipedia:{language}:{page_id}:{revision_id}"

    Since none of the components can contain ':', the mapping is bijective.
    The duplicate-document_id branch is therefore unreachable after the
    duplicate-article_id check, but is retained as a fail-safe.
    """

    def test_duplicate_article_id_caught_first(self) -> None:
        """Identical rows are caught by the article_id check, not document_id."""
        rows = [
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
            _make_article_row(wikidata="Q1", page_id=1, revision_id=100),
        ]
        table = _make_article_table(rows)
        with pytest.raises(WikipediaDocumentConversionError, match=r"article_id"):
            build_wikipedia_document_table(table)

    def test_bijection_argument(self) -> None:
        """Demonstrate that distinct valid article_ids produce distinct document_ids."""
        pairs = [
            ("Q1", "en", 1, 100),
            ("Q1", "en", 1, 200),
            ("Q1", "en", 2, 100),
            ("Q1", "fr", 1, 100),
            ("Q2", "en", 1, 100),
        ]
        doc_ids = set()
        for wikidata, lang, pid, rid in pairs:
            row = _make_article_row(wikidata=wikidata, language=lang, page_id=pid, revision_id=rid)
            doc = wikipedia_document_from_article_row(row)
            assert doc.document_id not in doc_ids, f"Collision: {doc.document_id}"
            doc_ids.add(doc.document_id)
        assert len(doc_ids) == len(pairs)


# ===========================================================================
# Compatibility Tests
# ===========================================================================


class TestLegacyCompatibility:
    """Verify existing legacy contracts remain unchanged."""

    def test_legacy_document_from_article_row_unchanged(self) -> None:
        row = _make_article_row()
        doc = document_from_article_row(row)
        assert isinstance(doc, Document)
        assert doc.document_id == document_id("Q235", "wikipedia", "en", 3649, 1234567)
        assert doc.project == "wikipedia"
        assert doc.article_id == row["article_id"]

    def test_legacy_document_shape_unchanged(self) -> None:
        import dataclasses

        fields = [f.name for f in dataclasses.fields(Document)]
        assert tuple(fields) == DOCUMENT_COLUMNS

    def test_legacy_document_columns_frozen(self) -> None:
        assert len(DOCUMENT_COLUMNS) == 23

    def test_legacy_document_schema_frozen(self) -> None:
        ds = document_schema()
        assert len(ds) == 23
        assert tuple(ds.names) == DOCUMENT_COLUMNS

    def test_no_public_import_identity_changes(self) -> None:
        """All original public imports from augmentation.models still work."""
        from osm_polygon_wikidata_only.augmentation.models import (
            Document as D,
        )
        from osm_polygon_wikidata_only.augmentation.models import (
            document_from_article_row as dfar,
        )
        from osm_polygon_wikidata_only.augmentation.models import (
            document_id as did,
        )

        assert D is Document
        assert dfar is document_from_article_row
        assert did is document_id
