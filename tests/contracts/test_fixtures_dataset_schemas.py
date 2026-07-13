"""Frozen dataset schemas verified against checked-in Parquet fixtures.

Every contract test in this module reads
``tests/fixtures/processed/*.parquet`` (committed to the repo) and
asserts that the actual schema and column ordering match the
documented contract. Each test also reconciles the live
``*_schema()`` factories against the checked-in JSON snapshots in
``tests/fixtures/schemas/`` so that any drift in *name, type,
nullability, or metadata description* fails the build.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    article_schema,
    polygon_article_schema,
    polygon_schema,
)

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures"
PROCESSED = FIXTURE_ROOT / "processed"
SCHEMAS = FIXTURE_ROOT / "schemas"


def _read_schema(path: Path) -> pa.Schema:
    return pq.read_metadata(path).schema.to_arrow_schema()


def _field_metadata(metadata: dict | None) -> dict[str, str]:
    """Decode a pyarrow ``Field.metadata`` mapping into JSON-friendly form."""
    if not metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in metadata.items():
        decoded[key.decode() if isinstance(key, bytes) else str(key)] = (
            value.decode() if isinstance(value, bytes) else str(value)
        )
    return decoded


def _schema_from_snapshot(pa_field: pa.Field) -> dict[str, object]:
    """Project a real pyarrow field into the JSON snapshot shape."""
    return {
        "name": pa_field.name,
        "type": str(pa_field.type),
        "nullable": bool(pa_field.nullable),
        "metadata": _field_metadata(pa_field.metadata),
    }


def _assert_schema_matches_snapshot(
    parquet_path: Path,
    expected_columns: tuple[str, ...],
    snapshot_path: Path,
    schema_factory: Callable[[], pa.Schema],
) -> None:
    """Assert every facet of the parquet + JSON snapshot reconciles with the live ``*_schema()`` factory.

    Compares field *name, type, nullability, and description metadata*
    for every documented column. Any drift on any axis fails the test.
    """
    parquet_schema = _read_schema(parquet_path)
    assert [field.name for field in parquet_schema] == list(expected_columns), (
        f"{parquet_path}: column order/contents drifted"
    )

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["columns"] == list(expected_columns), (
        f"{snapshot_path}: snapshot column list drifted"
    )

    schema_from_factory = schema_factory()
    factory_fields = [_schema_from_snapshot(field) for field in schema_from_factory]
    assert factory_fields == snapshot["fields"], (
        f"{snapshot_path}: field-by-field drift. "
        f"Got {factory_fields!r}, want {snapshot['fields']!r}"
    )

    parquet_fields = [_schema_from_snapshot(field) for field in parquet_schema]
    assert parquet_fields == snapshot["fields"], (
        f"{parquet_path}: parquet field-by-field drift. "
        f"Got {parquet_fields!r}, want {snapshot['fields']!r}"
    )


def test_polygon_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "polygons" / "monaco-latest.parquet",
        POLYGON_COLUMNS,
        SCHEMAS / "polygon.json",
        polygon_schema,
    )


def test_article_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "articles" / "monaco-latest.parquet",
        ARTICLE_COLUMNS,
        SCHEMAS / "article.json",
        article_schema,
    )


def test_polygon_article_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "polygon_articles" / "monaco-latest.parquet",
        POLYGON_ARTICLE_COLUMNS,
        SCHEMAS / "polygon_article.json",
        polygon_article_schema,
    )


def test_document_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "wikipedia" / "documents" / "monaco-latest.parquet",
        DOCUMENT_COLUMNS,
        SCHEMAS / "document.json",
        document_schema,
    )


def test_wikivoyage_documents_parquet_is_empty_with_correct_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "wikivoyage" / "documents" / "monaco-latest.parquet",
        DOCUMENT_COLUMNS,
        SCHEMAS / "document.json",
        document_schema,
    )
    table = pq.read_table(PROCESSED / "wikivoyage" / "documents" / "monaco-latest.parquet")
    assert table.num_rows == 0


def test_wikivoyage_sections_parquet_is_empty_with_correct_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "wikivoyage" / "sections" / "monaco-latest.parquet",
        SECTION_COLUMNS,
        SCHEMAS / "section.json",
        section_schema,
    )
    table = pq.read_table(PROCESSED / "wikivoyage" / "sections" / "monaco-latest.parquet")
    assert table.num_rows == 0


def test_section_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "wikipedia" / "sections" / "monaco-latest.parquet",
        SECTION_COLUMNS,
        SCHEMAS / "section.json",
        section_schema,
    )


def test_fact_parquet_matches_documented_schema() -> None:
    _assert_schema_matches_snapshot(
        PROCESSED / "wikidata" / "facts" / "monaco-latest.parquet",
        FACT_COLUMNS,
        SCHEMAS / "fact.json",
        fact_schema,
    )


def test_polygon_fixture_has_at_least_one_row() -> None:
    parquet_path = PROCESSED / "polygons" / "monaco-latest.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows >= 1
    wikidata = table.column("wikidata").to_pylist()
    assert "Q235" in wikidata


def test_manifest_fixture_is_well_formed() -> None:
    """The per-PBF manifest JSON must contain every documented key."""
    manifest_path = PROCESSED / "manifests" / "processed_pbfs.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "monaco-latest.osm.pbf" in payload
    entry = payload["monaco-latest.osm.pbf"]
    expected = {
        "source_pbf",
        "region",
        "polygons_path",
        "articles_path",
        "polygon_articles_path",
        "extraction_version",
        "processed_at",
        "polygon_count",
        "unique_wikidata_count",
        "article_count",
    }
    assert expected <= set(entry.keys())


def test_required_polygon_columns_carry_doc_descriptions() -> None:
    """Documented key polygon columns must carry the human-readable description.

    The Arrow nullability for every column is documented in the
    snapshot JSON; here we only assert the semantic contract: that
    the four columns documented as the polygon's primary identifier
    each have a non-empty ``description`` metadata entry sourced from
    :data:`POLYGON_DESCRIPTIONS`.
    """
    schema = polygon_schema()
    fields_by_name = {field.name: field for field in schema}
    expected_descriptions = {
        "polygon_id",
        "wikidata",
        "region",
        "source_pbf",
    }
    for column in expected_descriptions:
        assert column in fields_by_name, f"{column} missing from polygon schema"
        metadata = fields_by_name[column].metadata or {}
        raw = metadata.get(b"description", b"")
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        assert text, f"{column} must carry a non-empty description metadata entry"


def test_fixture_parquet_files_round_trip() -> None:
    """The committed parquet files are readable and produce stable row counts."""
    expected_counts = {
        PROCESSED / "polygons" / "monaco-latest.parquet": 1,
        PROCESSED / "articles" / "monaco-latest.parquet": 1,
        PROCESSED / "polygon_articles" / "monaco-latest.parquet": 1,
        PROCESSED / "wikipedia" / "documents" / "monaco-latest.parquet": 1,
        PROCESSED / "wikipedia" / "sections" / "monaco-latest.parquet": 1,
        PROCESSED / "wikivoyage" / "documents" / "monaco-latest.parquet": 0,
        PROCESSED / "wikivoyage" / "sections" / "monaco-latest.parquet": 0,
        PROCESSED / "wikidata" / "facts" / "monaco-latest.parquet": 1,
    }
    for path, expected_rows in expected_counts.items():
        assert path.exists(), f"missing fixture: {path}"
        assert pq.read_table(path).num_rows == expected_rows, f"{path}: row count drift"
