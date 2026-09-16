"""Acceptance tests for the V2-only row-level language split generator."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import load_dataset

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    build_language_inventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.v2.language_splits import (
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    build_v2_language_splits,
    main,
)
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)


def _row_for_schema(schema: pa.Schema, **values: object) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_boolean(field.type):
            row[field.name] = False
        elif pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        else:
            row[field.name] = ""
    row.update(values)
    return row


def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="snappy",
        row_group_size=1,
    )


def _polygon(polygon_id: str, *, best_language: str) -> dict[str, object]:
    return _row_for_schema(
        polygon_v2_schema(),
        polygon_id=polygon_id,
        region="fixture",
        source_pbf=f"{polygon_id}.osm.pbf",
        osm_type="way",
        osm_id=int(polygon_id.removeprefix("polygon-")),
        wikidata="Q1",
        best_language=best_language,
    )


def _document(document_id: str, language: object, title: str) -> dict[str, object]:
    return _row_for_schema(
        wikipedia_document_v2_schema(),
        document_id=document_id,
        article_id=f"article-{document_id}",
        wikidata="Q1",
        project="wikipedia",
        language=language,
        site=f"{language or 'unknown'}wiki",
        title=title,
        page_id=1,
        revision_id=1,
        full_text=title,
        fetch_status="ok",
    )


def _section(section_id: str, language: object) -> dict[str, object]:
    return _row_for_schema(
        section_schema(),
        section_id=section_id,
        document_id=f"document-{section_id}",
        article_id=f"article-{section_id}",
        project="wikipedia",
        language=language,
        page_id=1,
        revision_id=1,
        section_index=0,
        text=f"text-{section_id}",
    )


def _link(
    document_id: str, language: object, *, polygon_id: str = "polygon-1"
) -> dict[str, object]:
    return _row_for_schema(
        polygon_document_link_v2_schema(),
        polygon_id=polygon_id,
        document_id=document_id,
        project="wikipedia",
        wikidata="Q1",
        language=language,
        source_pbf="a-latest.osm.pbf",
        region="fixture",
        osm_type="way",
        osm_id=1,
        page_id=1,
        revision_id=1,
        link_sources='["wikidata"]',
    )


def _write_v2_manifest(root: Path, stems: tuple[str, ...]) -> None:
    regions = {
        stem: {
            "source_pbf": f"{stem}.osm.pbf",
            "region": "fixture",
            "polygons_path": f"polygons/{stem}.parquet",
            "documents_path": f"wikipedia/documents/{stem}.parquet",
            "sections_path": f"wikipedia/sections/{stem}.parquet",
            "links_path": f"polygon_document_links/{stem}.parquet",
        }
        for stem in stems
    }
    manifest = {"contract_version": "wikipedia-tags-v2", "regions": regions}
    path = root / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _write_v2_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "processed_v2"
    stems = ("z-latest", "a-latest")

    _write_table(
        root / "polygons/a-latest.parquet",
        [_polygon("polygon-1", best_language="en")],
        polygon_v2_schema(),
    )
    _write_table(
        root / "polygons/z-latest.parquet",
        [_polygon("polygon-2", best_language="de")],
        polygon_v2_schema(),
    )
    _write_table(
        root / "wikipedia/documents/a-latest.parquet",
        [
            _document("doc-fr-a", "fr", "French A"),
            _document("doc-de-a", "de", "German A"),
            _document("doc-unknown-a", "en/fr", "Unknown A"),
        ],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        root / "wikipedia/documents/z-latest.parquet",
        [_document("doc-fr-z", " FR ", "French Z")],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        root / "wikipedia/sections/a-latest.parquet",
        [
            _section("section-fr-a", "fr"),
            _section("section-de-a", "de"),
            _section("section-unknown-a", None),
        ],
        section_schema(),
    )
    _write_table(
        root / "wikipedia/sections/z-latest.parquet",
        [_section("section-fr-z", "fr")],
        section_schema(),
    )
    _write_table(
        root / "polygon_document_links/a-latest.parquet",
        [
            _link("doc-fr-a", "fr"),
            _link("doc-de-a", "de"),
            _link("doc-missing-a", None),
            _link("doc-blank-a", "  "),
            _link("doc-malformed-a", "en/fr"),
            _link("doc-simple-a", "simple"),
            _link("doc-alias-a", "be_x_old"),
        ],
        polygon_document_link_v2_schema(),
    )
    _write_table(
        root / "polygon_document_links/z-latest.parquet",
        [_link("doc-fr-z", "fr", polygon_id="polygon-2")],
        polygon_document_link_v2_schema(),
    )
    _write_v2_manifest(root, stems)
    return root


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))


def _shard(root: Path, table: str, language: str, stem: str) -> Path:
    configuration = f"{table}_by_language"
    return root / "language_splits" / configuration / f"lang-{language}" / f"{stem}.parquet"


def _rows(root: Path, table: str, language: str) -> list[dict[str, object]]:
    configuration = f"{table}_by_language"
    paths = sorted(
        (root / "language_splits" / configuration / f"lang-{language}").glob("*.parquet")
    )
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


def test_v2_split_keeps_each_multilingual_row_in_its_own_partition(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)

    french_documents = _rows(root, "wikipedia_documents", "fr")
    german_documents = _rows(root, "wikipedia_documents", "de")
    french_links = _rows(root, "polygon_document_links", "fr")
    german_links = _rows(root, "polygon_document_links", "de")

    assert [row["document_id"] for row in french_documents] == ["doc-fr-a", "doc-fr-z"]
    assert [row["document_id"] for row in german_documents] == ["doc-de-a"]
    assert [(row["polygon_id"], row["document_id"]) for row in french_links] == [
        ("polygon-1", "doc-fr-a"),
        ("polygon-2", "doc-fr-z"),
    ]
    assert [(row["polygon_id"], row["document_id"]) for row in german_links] == [
        ("polygon-1", "doc-de-a")
    ]
    assert (root / "language_splits" / "polygons").exists() is False


def test_v2_split_routes_unusable_values_to_lang_unknown(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)

    unknown = _rows(root, "polygon_document_links", "unknown")
    aliases = _rows(root, "polygon_document_links", "be-tarask")

    assert [row["document_id"] for row in unknown] == [
        "doc-missing-a",
        "doc-blank-a",
        "doc-malformed-a",
        "doc-simple-a",
    ]
    assert [row["language"] for row in unknown] == [None, "  ", "en/fr", "simple"]
    assert [row["language"] for row in aliases] == ["be_x_old"]


def test_v2_split_conserves_rows_and_preserves_schema(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    inventory = build_language_inventory(root, DatasetContract.V2)

    table_specs = {spec.table: spec for spec in language_table_specs(DatasetContract.V2)}
    for table_inventory in inventory.tables:
        spec = table_specs[table_inventory.table]
        observed = 0
        for bucket in table_inventory.buckets:
            paths = sorted(
                (root / "language_splits" / spec.configuration / bucket.split).glob("*.parquet")
            )
            observed += sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
            for path in paths:
                assert pq.ParquetFile(path).schema_arrow.equals(
                    spec.schema_factory(), check_metadata=True
                )
        assert observed == table_inventory.row_count

    manifest = _manifest(root)
    assert manifest["dataset_contract"] == "v2"
    assert manifest["v2_contract_version"] == "wikipedia-tags-v2"
    assert manifest["source_manifest"] == "manifests/processed_pbfs.json"


def test_v2_split_preserves_sorted_source_and_row_order(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root, batch_size=1)

    assert [
        path.name
        for path in sorted(
            (root / "language_splits/wikipedia_documents_by_language/lang-fr").glob("*.parquet")
        )
    ] == [
        "a-latest.parquet",
        "z-latest.parquet",
    ]
    assert [row["document_id"] for row in _rows(root, "wikipedia_documents", "fr")] == [
        "doc-fr-a",
        "doc-fr-z",
    ]
    assert [row["document_id"] for row in _rows(root, "wikipedia_documents", "de")] == ["doc-de-a"]


def test_v2_split_is_byte_stable_on_repeated_runs(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    first = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "language_splits").rglob("*.parquet")
    }
    first_manifest = (root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_bytes()

    build_v2_language_splits(root)
    second = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "language_splits").rglob("*.parquet")
    }

    assert second == first
    assert (root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_bytes() == first_manifest
    assert not list((root / "language_splits").rglob("*.tmp"))


def test_v2_split_does_not_route_or_copy_the_polygon_table(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    source = (root / "polygons/a-latest.parquet").read_bytes()

    build_v2_language_splits(root)

    assert (root / "polygons/a-latest.parquet").read_bytes() == source
    assert not list((root / "language_splits").rglob("*polygon*.parquet"))
    assert all(
        "polygons" not in path.parts for path in (root / "language_splits").rglob("*.parquet")
    )


def test_v2_split_rejects_a_v1_processed_root_without_writing_output(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    manifest = root / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "fixture.osm.pbf": {
                    "polygons_path": "polygons/fixture.parquet",
                    "polygon_articles_path": "polygon_articles/fixture.parquet",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="V2 manifest contract version mismatch"):
        build_v2_language_splits(root)

    assert not (root / "language_splits").exists()


def test_v2_split_can_be_loaded_with_standard_datasets(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    french_files = sorted(
        (root / "language_splits/wikipedia_documents_by_language/lang-fr").glob("*.parquet")
    )
    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in french_files],
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
    )

    assert dataset.num_rows == 2
    assert dataset.column_names == [field.name for field in wikipedia_document_v2_schema()]
    assert dataset["document_id"] == ["doc-fr-a", "doc-fr-z"]


def test_v2_split_module_runs_as_a_local_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_v2_fixture(tmp_path)

    assert main([str(root), "--batch-size", "1"]) == 0

    captured = capsys.readouterr()
    assert str(root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH) in captured.out
