"""Integration tests for the shared V1/V2 language-split release path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.cli.commands import main
from osm_polygon_wikidata_only.cli.parser import build_parser
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.hf.language_split_release import (
    LanguageSplitReleaseError,
    plan_language_split_release,
    run_language_split_release,
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
        elif pa.types.is_list(field.type):
            row[field.name] = []
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


def _write_v1_manifest(root: Path) -> None:
    path = root / "manifests/processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "a.osm.pbf": {
                    "source_pbf": "a.osm.pbf",
                    "region": "fixture",
                    "polygons_path": "polygons/a.parquet",
                    "polygon_articles_path": "polygon_articles/a.parquet",
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_v2_manifest(root: Path) -> None:
    path = root / "manifests/processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "contract_version": "wikipedia-tags-v2",
                "regions": {
                    "a-latest": {
                        "source_pbf": "a-latest.osm.pbf",
                        "region": "fixture",
                        "polygons_path": "polygons/a-latest.parquet",
                        "documents_path": "wikipedia/documents/a-latest.parquet",
                        "sections_path": "wikipedia/sections/a-latest.parquet",
                        "links_path": "polygon_document_links/a-latest.parquet",
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_v1_fixture(data_root: Path) -> Path:
    root = data_root / "processed"
    polygons = empty_row(tuple(field.name for field in polygon_schema()))
    polygons.update(
        {
            "polygon_id": "polygon-1",
            "region": "fixture",
            "source_pbf": "a.osm.pbf",
            "osm_type": "way",
            "osm_id": 1,
            "wikidata": "Q1",
            "best_language": "de",
        }
    )
    _write_table(root / "polygons/a.parquet", [polygons], polygon_schema())

    link_schema = polygon_document_link_schema()
    rows = [
        _row_for_schema(
            link_schema,
            polygon_id="polygon-1",
            document_id="doc-fr",
            project="wikipedia",
            language="fr",
            source_pbf="a.osm.pbf",
            region="fixture",
            osm_type="way",
            osm_id=1,
        ),
        _row_for_schema(
            link_schema,
            polygon_id="polygon-1",
            document_id="doc-en",
            project="wikipedia",
            language=" EN ",
            source_pbf="a.osm.pbf",
            region="fixture",
            osm_type="way",
            osm_id=1,
        ),
        _row_for_schema(
            link_schema,
            polygon_id="polygon-1",
            document_id="doc-unknown",
            project="wikipedia",
            language="en/fr",
            source_pbf="a.osm.pbf",
            region="fixture",
            osm_type="way",
            osm_id=1,
        ),
    ]
    _write_table(root / "polygon_articles/a.parquet", rows, link_schema)
    _write_v1_manifest(root)
    return root


def _write_v2_fixture(data_root: Path) -> Path:
    root = data_root / "processed_v2"
    polygons = _row_for_schema(
        polygon_v2_schema(),
        polygon_id="polygon-1",
        region="fixture",
        source_pbf="a-latest.osm.pbf",
        osm_type="way",
        osm_id=1,
        wikidata="Q1",
        best_language="de",
    )
    _write_table(root / "polygons/a-latest.parquet", [polygons], polygon_v2_schema())

    document_schema = wikipedia_document_v2_schema()
    documents = [
        _row_for_schema(
            document_schema,
            document_id="doc-fr",
            article_id="article-fr",
            wikidata="Q1",
            project="wikipedia",
            language="fr",
            site="frwiki",
            title="French",
            page_id=1,
            revision_id=1,
            full_text="French text",
            fetch_status="ok",
        ),
        _row_for_schema(
            document_schema,
            document_id="doc-unknown",
            article_id="article-unknown",
            wikidata="Q1",
            project="wikipedia",
            language=None,
            site="unknownwiki",
            title="Unknown",
            page_id=2,
            revision_id=2,
            full_text="Unknown text",
            fetch_status="ok",
        ),
    ]
    _write_table(root / "wikipedia/documents/a-latest.parquet", documents, document_schema)
    _write_table(
        root / "wikipedia/sections/a-latest.parquet",
        [
            _row_for_schema(
                section_schema(),
                section_id="section-fr",
                document_id="doc-fr",
                article_id="article-fr",
                project="wikipedia",
                language="fr",
                text="French section",
            )
        ],
        section_schema(),
    )
    link_schema = polygon_document_link_v2_schema()
    _write_table(
        root / "polygon_document_links/a-latest.parquet",
        [
            _row_for_schema(
                link_schema,
                polygon_id="polygon-1",
                document_id="doc-fr",
                project="wikipedia",
                wikidata="Q1",
                language="fr",
                source_pbf="a-latest.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=1,
                page_id=1,
                revision_id=1,
                link_sources='["wikidata"]',
            ),
            _row_for_schema(
                link_schema,
                polygon_id="polygon-1",
                document_id="doc-unknown",
                project="wikipedia",
                wikidata="Q1",
                language="be_x_old",
                source_pbf="a-latest.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=1,
                page_id=2,
                revision_id=2,
                link_sources='["tag"]',
            ),
        ],
        link_schema,
    )
    _write_v2_manifest(root)
    return root


def _write_both_fixture(data_root: Path) -> None:
    _write_v1_fixture(data_root)
    _write_v2_fixture(data_root)


def _v1_partition(data_root: Path, table: str, language: str) -> Path:
    return (
        data_root
        / "processed/language_splits/data"
        / f"{table}_by_language"
        / f"lang-{language}-00000-of-00001.parquet"
    )


def _v2_partition(data_root: Path, table: str, language: str) -> Path:
    return (
        data_root
        / "processed_v2/language_splits"
        / f"{table}_by_language"
        / f"lang-{language}"
        / "a-latest.parquet"
    )


def test_language_split_parser_defaults_to_both_and_accepts_each_selector() -> None:
    parser = build_parser()

    assert parser.parse_args(["language-splits"]).dataset_version == "both"
    for selector in ("v1", "v2", "both"):
        args = parser.parse_args(["language-splits", "--dataset-version", selector])
        assert args.dataset_version == selector


def test_language_split_cli_emits_one_deterministic_dry_run_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_v1_fixture(tmp_path)

    assert (
        main(
            [
                "language-splits",
                "--data-root",
                str(tmp_path),
                "--dataset-version",
                "v1",
                "--batch-size",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    payload = json.loads(output[0])
    assert payload["command"] == "language-splits"
    assert payload["dataset_version"] == "v1"
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert not (tmp_path / "processed/language_splits").exists()


def test_plan_selects_only_the_requested_contract_and_keeps_roots_isolated(
    tmp_path: Path,
) -> None:
    _write_both_fixture(tmp_path)
    data_root = DataRoot(tmp_path)

    v1 = plan_language_split_release(data_root, dataset_version="v1", batch_size=1)
    v2 = plan_language_split_release(data_root, dataset_version="v2", batch_size=1)

    assert v1.versions == ("v1",)
    assert v2.versions == ("v2",)
    v1_release = cast(list[dict[str, object]], v1.to_payload()["releases"])[0]
    v2_release = cast(list[dict[str, object]], v2.to_payload()["releases"])[0]
    assert v1_release["processed_root"] == "processed"
    assert v1_release["output_root"] == "processed/language_splits"
    assert v2_release["processed_root"] == "processed_v2"
    assert v2_release["output_root"] == "processed_v2/language_splits"


def test_dry_run_is_deterministic_and_writes_no_language_outputs(tmp_path: Path) -> None:
    _write_both_fixture(tmp_path)
    data_root = DataRoot(tmp_path)

    first = run_language_split_release(
        data_root, dataset_version="both", batch_size=1, dry_run=True
    )
    second = run_language_split_release(
        data_root, dataset_version="both", batch_size=1, dry_run=True
    )

    assert first.to_payload() == second.to_payload()
    assert first.to_payload()["dry_run"] is True
    assert not (tmp_path / "processed/language_splits").exists()
    assert not (tmp_path / "processed_v2/language_splits").exists()


def test_v1_generation_routes_multilingual_rows_and_unknown_values_by_row_language(
    tmp_path: Path,
) -> None:
    _write_v1_fixture(tmp_path)

    result = run_language_split_release(DataRoot(tmp_path), dataset_version="v1", batch_size=1)

    french = pq.read_table(_v1_partition(tmp_path, "polygon_articles", "fr")).to_pylist()
    english = pq.read_table(_v1_partition(tmp_path, "polygon_articles", "en")).to_pylist()
    unknown = pq.read_table(_v1_partition(tmp_path, "polygon_articles", "unknown")).to_pylist()
    payload = result.to_payload()

    assert [row["document_id"] for row in french] == ["doc-fr"]
    assert [row["document_id"] for row in english] == ["doc-en"]
    assert [row["document_id"] for row in unknown] == ["doc-unknown"]
    release = cast(list[dict[str, object]], payload["releases"])[0]
    assert release["manifest_path"] == (
        "processed/language_splits/manifests/language_splits_v1.json"
    )
    assert (
        sum(
            cast(int, item["row_count"])
            for item in cast(list[dict[str, object]], release["files"])
            if item["table"] == "polygon_articles"
        )
        == 3
    )


def test_v2_generation_preserves_rows_and_uses_hugging_face_compatible_names(
    tmp_path: Path,
) -> None:
    _write_v2_fixture(tmp_path)

    result = run_language_split_release(DataRoot(tmp_path), dataset_version="v2", batch_size=1)

    french = pq.read_table(_v2_partition(tmp_path, "wikipedia_documents", "fr")).to_pylist()
    unknown = pq.read_table(
        _v2_partition(tmp_path, "polygon_document_links", "be-tarask")
    ).to_pylist()
    payload = result.to_payload()

    assert [row["document_id"] for row in french] == ["doc-fr"]
    assert [row["document_id"] for row in unknown] == ["doc-unknown"]
    release = cast(list[dict[str, object]], payload["releases"])[0]
    assert release["manifest_path"] == ("processed_v2/manifests/language_splits.json")
    assert all(
        "_" not in cast(str, item["split"]) and cast(str, item["split"]).startswith("lang-")
        for item in cast(list[dict[str, object]], release["files"])
    )
    assert not (tmp_path / "processed_v2/language_splits/polygons").exists()


def test_both_generation_conserves_rows_and_writes_both_manifests(tmp_path: Path) -> None:
    _write_both_fixture(tmp_path)

    result = run_language_split_release(DataRoot(tmp_path), dataset_version="both", batch_size=1)

    payload = result.to_payload()
    releases = cast(list[dict[str, object]], payload["releases"])
    assert [release["dataset_version"] for release in releases] == ["v1", "v2"]
    assert (tmp_path / "processed/language_splits/manifests/language_splits_v1.json").is_file()
    assert (tmp_path / "processed_v2/manifests/language_splits.json").is_file()
    for release in releases:
        files = cast(list[dict[str, object]], release["files"])
        tables = cast(list[dict[str, object]], release["tables"])
        files_by_table = {
            table: sum(cast(int, item["row_count"]) for item in files if item["table"] == table)
            for table in {item["table"] for item in files}
        }
        assert all(files_by_table.get(table["table"], 0) == table["row_count"] for table in tables)


def test_both_selection_validates_every_inventory_before_writing(tmp_path: Path) -> None:
    _write_v1_fixture(tmp_path)
    broken_v2 = tmp_path / "processed_v2/manifests/processed_pbfs.json"
    broken_v2.parent.mkdir(parents=True, exist_ok=True)
    broken_v2.write_text(
        json.dumps(
            {
                "contract_version": "wikipedia-tags-v2",
                "regions": {
                    "a-latest": {
                        "source_pbf": "a-latest.osm.pbf",
                        "region": "fixture",
                        "polygons_path": "polygons/a-latest.parquet",
                        "documents_path": "wikipedia/documents/a-latest.parquet",
                        "sections_path": "wikipedia/sections/a-latest.parquet",
                        "links_path": "polygon_document_links/missing.parquet",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LanguageSplitReleaseError, match="missing artifact"):
        run_language_split_release(DataRoot(tmp_path), dataset_version="both", dry_run=False)

    assert not (tmp_path / "processed/language_splits").exists()


def test_language_split_result_loads_with_standard_datasets_loader(tmp_path: Path) -> None:
    _write_v1_fixture(tmp_path)

    run_language_split_release(DataRoot(tmp_path), dataset_version="v1", batch_size=1)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import json",
                    "import sys",
                    "from datasets import load_dataset",
                    "dataset = load_dataset(",
                    '    "parquet",',
                    '    data_files={"train": sys.argv[1]},',
                    '    split="train",',
                    "    cache_dir=sys.argv[2],",
                    ")",
                    'print(json.dumps({"num_rows": dataset.num_rows, "language": list(dataset["language"])}))',
                )
            ),
            str(_v1_partition(tmp_path, "polygon_articles", "fr")),
            str(tmp_path / "hf-cache"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"num_rows": 1, "language": ["fr"]}
