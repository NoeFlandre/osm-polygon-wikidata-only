"""Deterministic tests for the shared V1/V2 language-split contract."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import POLYGON_COLUMNS, empty_row, polygon_schema
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageDisposition,
    LanguageInventoryError,
    LanguageTable,
    build_language_inventory,
    language_config_name,
    language_split_name,
    language_table_specs,
    normalize_language,
)
from osm_polygon_wikidata_only.hf.repo_layout import REMOTE_LINKS_DIR, REMOTE_POLYGONS_DIR
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)


@pytest.mark.parametrize(
    ("raw", "canonical", "disposition"),
    [
        (" en ", "en", LanguageDisposition.CANONICAL),
        ("EN_us", "en-us", LanguageDisposition.CANONICAL),
        ("be_x_old", "be-tarask", LanguageDisposition.LEGACY_ALIAS),
        ("zh_min_nan", "zh-min-nan", LanguageDisposition.CANONICAL),
    ],
)
def test_normalize_language_uses_the_existing_contract(
    raw: object,
    canonical: str,
    disposition: LanguageDisposition,
) -> None:
    result = normalize_language(raw)

    assert result.canonical == canonical
    assert result.partition == canonical
    assert result.disposition is disposition


@pytest.mark.parametrize(
    ("raw", "disposition"),
    [
        (None, LanguageDisposition.MISSING),
        (" \t\n", LanguageDisposition.BLANK),
        ("en/fr", LanguageDisposition.MALFORMED),
        (42, LanguageDisposition.MALFORMED),
        ("simple", LanguageDisposition.LEGACY_UNUSABLE),
        ("abstract", LanguageDisposition.LEGACY_UNUSABLE),
    ],
)
def test_unusable_language_values_are_explicitly_unknown(
    raw: object,
    disposition: LanguageDisposition,
) -> None:
    result = normalize_language(raw)

    assert result.canonical is None
    assert result.partition == "unknown"
    assert result.disposition is disposition
    assert language_split_name(raw) == "lang-unknown"


def test_contract_uses_actual_textual_schemas_and_keeps_versions_separate() -> None:
    v1 = {spec.table: spec for spec in language_table_specs(DatasetContract.V1)}
    v2 = {spec.table: spec for spec in language_table_specs(DatasetContract.V2)}

    assert tuple(v1) == (
        LanguageTable.POLYGON_ARTICLES,
        LanguageTable.WIKIPEDIA_DOCUMENTS,
        LanguageTable.WIKIPEDIA_SECTIONS,
        LanguageTable.WIKIVOYAGE_DOCUMENTS,
        LanguageTable.WIKIVOYAGE_SECTIONS,
    )
    assert tuple(v2) == (
        LanguageTable.POLYGON_DOCUMENT_LINKS,
        LanguageTable.WIKIPEDIA_DOCUMENTS,
        LanguageTable.WIKIPEDIA_SECTIONS,
    )
    assert v1[LanguageTable.WIKIPEDIA_DOCUMENTS].schema_factory() == wikipedia_document_schema()
    assert v2[LanguageTable.WIKIPEDIA_DOCUMENTS].schema_factory() == wikipedia_document_v2_schema()
    assert v1[LanguageTable.POLYGON_ARTICLES].schema_factory() == polygon_document_link_schema()
    assert (
        v2[LanguageTable.POLYGON_DOCUMENT_LINKS].schema_factory()
        == polygon_document_link_v2_schema()
    )
    assert "polygons" not in v1
    assert "polygons" not in v2


def test_hugging_face_names_are_additive_and_address_one_language() -> None:
    assert REMOTE_POLYGONS_DIR == "polygons"
    assert REMOTE_LINKS_DIR == "polygon_articles"
    assert (
        language_config_name(LanguageTable.WIKIPEDIA_DOCUMENTS) == "wikipedia_documents_by_language"
    )
    assert language_config_name(LanguageTable.POLYGON_ARTICLES) == "polygon_articles_by_language"
    assert (
        language_config_name(LanguageTable.POLYGON_DOCUMENT_LINKS)
        == "polygon_document_links_by_language"
    )
    assert language_split_name(" FR ") == "lang-fr"
    assert language_split_name("be_x_old") == "lang-be-tarask"


def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row_for_schema(schema: pa.Schema, **values: object) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        elif pa.types.is_boolean(field.type):
            row[field.name] = False
        else:
            row[field.name] = ""
    row.update(values)
    return row


def _polygon_row(polygon_id: str, best_language: object) -> dict[str, object]:
    row = empty_row(POLYGON_COLUMNS)
    row.update(
        {
            "polygon_id": polygon_id,
            "region": "fixture",
            "source_pbf": "fixture.osm.pbf",
            "osm_type": "way",
            "osm_id": int(polygon_id),
            "wikidata": "Q1",
            "best_language": best_language,
        }
    )
    return row


def _write_v1_manifest(processed: Path) -> None:
    manifest = {
        "fixture.osm.pbf": {
            "source_pbf": "fixture.osm.pbf",
            "region": "fixture",
            "polygons_path": "polygons/fixture.parquet",
            "polygon_articles_path": "polygon_articles/fixture.parquet",
        }
    }
    path = processed / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _write_v2_manifest(processed: Path) -> None:
    manifest = {
        "contract_version": "wikipedia-tags-v2",
        "regions": {
            "fixture-latest": {
                "source_pbf": "fixture-latest.osm.pbf",
                "region": "fixture",
                "polygons_path": "polygons/fixture-latest.parquet",
                "documents_path": "wikipedia/documents/fixture-latest.parquet",
                "sections_path": "wikipedia/sections/fixture-latest.parquet",
                "links_path": "polygon_document_links/fixture-latest.parquet",
            }
        },
    }
    path = processed / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_v1_inventory_uses_each_textual_language_row_without_polygon_deduplication(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    _write_table(
        processed / "polygons/fixture.parquet",
        [_polygon_row("1", "en")],
        polygon_schema(),
    )
    _write_table(
        processed / "wikipedia/documents/fixture.parquet",
        [
            _row_for_schema(wikipedia_document_schema(), document_id="en-doc", language="en"),
            _row_for_schema(wikipedia_document_schema(), document_id="fr-doc", language="fr"),
        ],
        wikipedia_document_schema(),
    )
    _write_table(
        processed / "polygon_articles/fixture.parquet",
        [
            _row_for_schema(
                polygon_document_link_schema(),
                polygon_id="1",
                document_id="en-doc",
                project="wikipedia",
                language="en",
                wikidata="Q1",
                osm_type="way",
                osm_id=1,
            ),
            _row_for_schema(
                polygon_document_link_schema(),
                polygon_id="1",
                document_id="fr-doc",
                project="wikipedia",
                language="fr",
                wikidata="Q1",
                osm_type="way",
                osm_id=1,
            ),
        ],
        polygon_document_link_schema(),
    )
    _write_v1_manifest(processed)

    inventory = build_language_inventory(processed, DatasetContract.V1)

    assert inventory.languages == ("en", "fr")
    assert inventory.table("polygon_articles").row_count == 2
    assert inventory.table("polygon_articles").bucket("en").row_count == 1
    assert inventory.table("polygon_articles").bucket("fr").row_count == 1
    assert inventory.table("polygons", allow_non_language=True).row_count == 1


def test_v1_inventory_routes_missing_blank_malformed_and_legacy_values_to_unknown(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    _write_table(
        processed / "polygons/fixture.parquet",
        [_polygon_row("1", "en")],
        polygon_schema(),
    )
    _write_table(
        processed / "polygon_articles/fixture.parquet",
        [
            _row_for_schema(polygon_document_link_schema(), polygon_id="1", language=None),
            _row_for_schema(polygon_document_link_schema(), polygon_id="2", language="  "),
            _row_for_schema(polygon_document_link_schema(), polygon_id="3", language="en/fr"),
            _row_for_schema(polygon_document_link_schema(), polygon_id="4", language="simple"),
            _row_for_schema(polygon_document_link_schema(), polygon_id="5", language="be_x_old"),
        ],
        polygon_document_link_schema(),
    )
    _write_v1_manifest(processed)

    inventory = build_language_inventory(processed, DatasetContract.V1)
    unknown = inventory.table("polygon_articles").bucket("unknown")

    assert inventory.languages == ("be-tarask",)
    assert unknown.row_count == 4
    assert unknown.missing_rows == 1
    assert unknown.blank_rows == 1
    assert unknown.malformed_rows == 1
    assert unknown.legacy_unusable_rows == 1
    assert inventory.table("polygon_articles").bucket("be-tarask").legacy_alias_rows == 1


def test_v2_inventory_is_separate_and_uses_v2_schemas(tmp_path: Path) -> None:
    processed_v2 = tmp_path / "processed_v2"
    stem = "fixture-latest"
    _write_table(
        processed_v2 / f"polygons/{stem}.parquet",
        [_row_for_schema(polygon_v2_schema(), best_language="en")],
        polygon_v2_schema(),
    )
    _write_table(
        processed_v2 / f"wikipedia/documents/{stem}.parquet",
        [_row_for_schema(wikipedia_document_v2_schema(), document_id="v2-fr", language="fr")],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        processed_v2 / f"wikipedia/sections/{stem}.parquet",
        [_row_for_schema(section_schema(), section_id="v2-fr-section", language="fr")],
        section_schema(),
    )
    _write_table(
        processed_v2 / f"polygon_document_links/{stem}.parquet",
        [
            _row_for_schema(
                polygon_document_link_v2_schema(),
                polygon_id="1",
                document_id="v2-fr",
                language="fr",
                project="wikipedia",
            )
        ],
        polygon_document_link_v2_schema(),
    )
    _write_v2_manifest(processed_v2)

    inventory = build_language_inventory(processed_v2, DatasetContract.V2)

    assert inventory.dataset_id == "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    assert inventory.languages == ("fr",)
    assert tuple(table.table for table in inventory.tables) == (
        "polygon_document_links",
        "wikipedia_documents",
        "wikipedia_sections",
    )
    assert inventory.table("polygon_document_links").bucket("fr").row_count == 1


def test_inventory_rejects_an_unvalidated_textual_schema(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_table(
        processed / "polygons/fixture.parquet",
        [_polygon_row("1", "en")],
        polygon_schema(),
    )
    _write_table(
        processed / "polygon_articles/fixture.parquet",
        [_row_for_schema(polygon_document_link_schema(), polygon_id="1", language="en")],
        polygon_document_link_schema(),
    )
    _write_table(
        processed / "wikipedia/documents/fixture.parquet",
        [{"language": "en"}],
        pa.schema([pa.field("language", pa.string())]),
    )
    _write_v1_manifest(processed)

    with pytest.raises(LanguageInventoryError, match="schema mismatch"):
        build_language_inventory(processed, DatasetContract.V1)


def test_inventory_is_deterministic_for_the_same_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_table(
        processed / "polygons/fixture.parquet",
        [_polygon_row("1", "en")],
        polygon_schema(),
    )
    _write_table(
        processed / "polygon_articles/fixture.parquet",
        [_row_for_schema(polygon_document_link_schema(), polygon_id="1", language="en")],
        polygon_document_link_schema(),
    )
    _write_v1_manifest(processed)

    first = build_language_inventory(processed, DatasetContract.V1).to_dict()
    second = build_language_inventory(processed, DatasetContract.V1).to_dict()

    assert first == second
