"""Integration tests for the shared V1/V2 language-split release path."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
from osm_polygon_wikidata_only.hf import language_split_release
from osm_polygon_wikidata_only.hf.language_split_release import (
    LanguageSplitReleaseError,
    LanguageSplitReleasePlan,
    LanguageSplitVersion,
    _expected_file_sort_key,
    _expected_files,
    _generate_version,
    _generated_file_sort_key,
    _generated_release_payload,
    _recompute_inventory,
    _source_language_counts,
    _validate_generated_inventory,
    plan_language_split_release,
    run_language_split_release,
)
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageInventory,
    LanguageTableInventory,
)
from osm_polygon_wikidata_only.v2.language_splits import V2LanguageSplitResult
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


def _write_v2_manifest(root: Path, *, include_z: bool = False) -> None:
    path = root / "manifests/processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    regions: dict[str, dict[str, str]] = {
        "a-latest": {
            "source_pbf": "a-latest.osm.pbf",
            "region": "fixture",
            "polygons_path": "polygons/a-latest.parquet",
            "documents_path": "wikipedia/documents/a-latest.parquet",
            "sections_path": "wikipedia/sections/a-latest.parquet",
            "links_path": "polygon_document_links/a-latest.parquet",
        }
    }
    if include_z:
        regions["z-latest"] = {
            "source_pbf": "z-latest.osm.pbf",
            "region": "fixture",
            "polygons_path": "polygons/z-latest.parquet",
            "documents_path": "wikipedia/documents/z-latest.parquet",
            "sections_path": "wikipedia/sections/z-latest.parquet",
            "links_path": "polygon_document_links/z-latest.parquet",
        }
    path.write_text(
        json.dumps(
            {
                "contract_version": "wikipedia-tags-v2",
                "regions": regions,
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
        / "part-00000-of-00001.parquet"
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
    assert v1.releases[0].manifest_path == (
        tmp_path / "processed/language_splits/manifests/language_splits_v1.json"
    )
    assert v2.releases[0].manifest_path == (
        tmp_path / "processed_v2/manifests/language_splits.json"
    )


def test_release_defaults_and_expected_file_payloads_are_explicit(tmp_path: Path) -> None:
    _write_both_fixture(tmp_path)

    plan = plan_language_split_release(tmp_path, batch_size=1)
    assert plan.versions == ("v1", "v2")
    assert plan.data_root == tmp_path.resolve()

    releases = cast(list[dict[str, object]], plan.to_payload()["releases"])
    v1_expected = cast(list[dict[str, object]], releases[0]["expected_files"])
    v2_expected = cast(list[dict[str, object]], releases[1]["expected_files"])
    assert v1_expected
    assert v2_expected
    assert all(
        set(record) == {"table", "configuration", "language", "split", "path", "row_count"}
        for record in v1_expected
    )
    assert all(
        cast(str, record["path"]).startswith("processed/language_splits/data/")
        for record in v1_expected
    )
    assert all(
        set(record)
        == {
            "table",
            "configuration",
            "language",
            "split",
            "path",
            "row_count",
            "status",
        }
        for record in v2_expected
    )
    assert all(
        cast(str, record["path"]).startswith("processed_v2/language_splits/")
        for record in v2_expected
    )
    assert all(cast(str, record["path"]).endswith(".parquet") for record in v2_expected)

    with pytest.raises(LanguageSplitReleaseError, match="batch_size must be positive"):
        plan_language_split_release(tmp_path, dataset_version="v1", batch_size=0)


def test_v2_dry_run_plans_bounded_shards_from_inventory_counts(tmp_path: Path) -> None:
    _write_v2_fixture(tmp_path)

    plan = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1)
    release = cast(list[dict[str, object]], plan.to_payload()["releases"])[0]
    documents = [
        record
        for record in cast(list[dict[str, object]], release["expected_files"])
        if record["table"] == "wikipedia_documents"
    ]

    assert {(record["language"], record["row_count"], record["path"]) for record in documents} == {
        (
            "fr",
            1,
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-fr/part-00000-of-00001.parquet",
        ),
        (
            "unknown",
            1,
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-unknown/part-00000-of-00001.parquet",
        ),
    }
    assert all("source_file" not in record for record in documents)


def test_v2_dry_run_calculates_multiple_shards_without_source_scans(tmp_path: Path) -> None:
    _write_v2_fixture(tmp_path)
    planned = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1).releases[0]
    original = cast(LanguageTableInventory, planned.inventory.table("wikipedia_documents"))
    original_fr = original.bucket("fr")
    large_fr = replace(original_fr, row_count=200_001)
    replacement = replace(
        original,
        row_count=original.row_count - original_fr.row_count + large_fr.row_count,
        buckets=tuple(
            large_fr if bucket.language == "fr" else bucket for bucket in original.buckets
        ),
    )
    inventory = replace(
        planned.inventory,
        tables=tuple(
            replacement if table.table.value == "wikipedia_documents" else table
            for table in planned.inventory.tables
        ),
    )

    documents = [
        record
        for record in _expected_files(planned, inventory)
        if record["table"] == "wikipedia_documents" and record["language"] == "fr"
    ]

    assert [(record["path"], record["row_count"]) for record in documents] == [
        (
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-fr/part-00000-of-00003.parquet",
            100_000,
        ),
        (
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-fr/part-00001-of-00003.parquet",
            100_000,
        ),
        (
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-fr/part-00002-of-00003.parquet",
            1,
        ),
    ]


def test_v2_dry_run_keeps_a_boundary_sized_bucket_in_one_shard(tmp_path: Path) -> None:
    _write_v2_fixture(tmp_path)
    planned = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1).releases[0]
    original = cast(LanguageTableInventory, planned.inventory.table("wikipedia_documents"))
    original_fr = original.bucket("fr")
    boundary_fr = replace(original_fr, row_count=100_000)
    inventory = replace(
        planned.inventory,
        tables=tuple(
            replace(
                table,
                row_count=table.row_count - original_fr.row_count + boundary_fr.row_count,
                buckets=tuple(
                    boundary_fr if bucket.language == "fr" else bucket for bucket in table.buckets
                ),
            )
            if table.table.value == "wikipedia_documents"
            else table
            for table in planned.inventory.tables
        ),
    )

    documents = [
        record
        for record in _expected_files(planned, inventory)
        if record["table"] == "wikipedia_documents" and record["language"] == "fr"
    ]

    assert [(record["path"], record["row_count"]) for record in documents] == [
        (
            "processed_v2/language_splits/wikipedia_documents_by_language/"
            "lang-fr/part-00000-of-00001.parquet",
            100_000,
        )
    ]


def test_v2_dry_run_reports_bounded_candidate_shards_and_rows(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    _write_table(
        root / "polygons/z-latest.parquet",
        [
            _row_for_schema(
                polygon_v2_schema(),
                polygon_id="polygon-2",
                region="fixture",
                source_pbf="z-latest.osm.pbf",
                osm_type="way",
                osm_id=2,
                wikidata="Q2",
                best_language="en",
            )
        ],
        polygon_v2_schema(),
    )
    _write_table(
        root / "wikipedia/documents/z-latest.parquet",
        [
            _row_for_schema(
                wikipedia_document_v2_schema(),
                document_id="doc-en-z",
                article_id="article-doc-en-z",
                wikidata="Q2",
                project="wikipedia",
                language="en",
                site="enwiki",
                title="English Z",
                page_id=2,
                revision_id=2,
                full_text="English Z",
                fetch_status="ok",
            )
        ],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        root / "wikipedia/sections/z-latest.parquet",
        [
            _row_for_schema(
                section_schema(),
                section_id="section-en-z",
                document_id="doc-en-z",
                article_id="article-doc-en-z",
                project="wikipedia",
                language="en",
                page_id=2,
                revision_id=2,
                section_index=0,
                text="English Z",
            )
        ],
        section_schema(),
    )
    _write_table(
        root / "polygon_document_links/z-latest.parquet",
        [
            _row_for_schema(
                polygon_document_link_v2_schema(),
                polygon_id="polygon-2",
                document_id="doc-en-z",
                project="wikipedia",
                wikidata="Q2",
                language="en",
                source_pbf="z-latest.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
                page_id=2,
                revision_id=2,
                link_sources='["wikidata"]',
            )
        ],
        polygon_document_link_v2_schema(),
    )
    _write_v2_manifest(root, include_z=True)

    plan = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1)
    release = cast(list[dict[str, object]], plan.to_payload()["releases"])[0]
    documents = [
        record
        for record in cast(list[dict[str, object]], release["expected_files"])
        if record["table"] == "wikipedia_documents"
    ]

    assert sorted(
        (record["language"], record["split"], record["path"], record["row_count"], record["status"])
        for record in documents
    ) == sorted(
        [
            (
                "en",
                "lang-en",
                "processed_v2/language_splits/wikipedia_documents_by_language/"
                "lang-en/part-00000-of-00001.parquet",
                1,
                "candidate",
            ),
            (
                "fr",
                "lang-fr",
                "processed_v2/language_splits/wikipedia_documents_by_language/"
                "lang-fr/part-00000-of-00001.parquet",
                1,
                "candidate",
            ),
            (
                "unknown",
                "lang-unknown",
                "processed_v2/language_splits/wikipedia_documents_by_language/"
                "lang-unknown/part-00000-of-00001.parquet",
                1,
                "candidate",
            ),
        ]
    )
    assert all("path" in record for record in documents)
    assert all("path_template" not in record for record in documents)
    assert all("source_file" not in record for record in documents)


@pytest.mark.parametrize(
    ("dataset_version", "table_name"),
    [
        (LanguageSplitVersion.V1, "polygon_articles"),
        (LanguageSplitVersion.V2, "wikipedia_documents"),
    ],
)
def test_expected_files_honor_explicit_inventory_override(
    tmp_path: Path,
    dataset_version: LanguageSplitVersion,
    table_name: str,
) -> None:
    _write_both_fixture(tmp_path)
    plan = plan_language_split_release(
        tmp_path, dataset_version=dataset_version.value, batch_size=1
    ).releases[0]
    original = cast(LanguageTableInventory, plan.inventory.table(table_name))
    if dataset_version is LanguageSplitVersion.V1:
        replacement = replace(original, buckets=(), row_count=0)
    else:
        replacement = replace(original, source_files=(), row_count=0, buckets=())
    alternate_inventory = replace(
        plan.inventory,
        tables=tuple(
            replacement if table.table.value == table_name else table
            for table in plan.inventory.tables
        ),
    )

    expected = _expected_files(plan, alternate_inventory)

    assert not any(record["table"] == table_name for record in expected)


def test_source_language_counts_are_bounded_exact_and_stably_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.parquet"
    schema = wikipedia_document_v2_schema()
    _write_table(
        source,
        [
            _row_for_schema(schema, document_id="missing", language=None),
            _row_for_schema(schema, document_id="english-1", language="en"),
            _row_for_schema(schema, document_id="english-2", language="en"),
            _row_for_schema(schema, document_id="zz", language="zz"),
        ],
        schema,
    )
    original_parquet_file = language_split_release.open_parquet
    batch_calls: list[dict[str, object]] = []

    class RecordingParquetFile:
        def __init__(self, path: Path) -> None:
            self._inner = original_parquet_file(path)

        def __enter__(self) -> RecordingParquetFile:
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            self._inner.__exit__(exc_type, exc_value, traceback)

        def iter_batches(self, **kwargs: object):
            batch_calls.append(kwargs)
            return self._inner.iter_batches(**kwargs)

    monkeypatch.setattr(language_split_release, "open_parquet", RecordingParquetFile)

    counts = _source_language_counts(source, "language")
    assert counts == {
        "en": 2,
        "zz": 1,
        "unknown": 1,
    }
    assert list(counts) == ["en", "zz", "unknown"]
    # The scan stays column-pruned and bounded, and reads on the calling
    # thread so Arrow's prefetch pool cannot stall it.
    assert batch_calls == [
        {
            "batch_size": language_split_release.DEFAULT_BATCH_SIZE,
            "use_threads": False,
            "columns": ["language"],
        }
    ]


def test_release_file_sort_keys_distinguish_table_language_and_path() -> None:
    expected_records: list[dict[str, object]] = [
        {"table": "a", "language": "en", "path": "a"},
        {"table": "a", "language": "en", "path": "z"},
        {"table": "a", "language": "unknown", "path": "a"},
        {"table": "b", "language": "en", "path": "a"},
    ]
    assert sorted(reversed(expected_records), key=_expected_file_sort_key) == expected_records
    assert _expected_file_sort_key({"table": "a", "language": "unknown", "path": "a"}) == (
        "a",
        "True",
        "unknown",
        "a",
    )
    assert _expected_file_sort_key({"table": "a", "language": "en"}) == (
        "a",
        "False",
        "en",
        "",
    )
    assert (
        _expected_file_sort_key({"table": "a", "language": "en", "path_template": "template"})[-1]
        == "template"
    )

    generated_records: list[dict[str, object]] = [
        {"table": "a", "language": "en", "path": "a"},
        {"table": "a", "language": "en", "path": "z"},
        {"table": "a", "language": "unknown", "path": "a"},
        {"table": "b", "language": "en", "path": "a"},
    ]
    assert sorted(reversed(generated_records), key=_generated_file_sort_key) == generated_records
    assert _generated_file_sort_key({"table": "a", "language": "unknown", "path": "a"}) == (
        "a",
        "True",
        "unknown",
        "a",
    )
    assert _generated_file_sort_key({"table": "a", "path": "a"})[1:3] == ("False", "")

    calls: list[tuple[object, object]] = []

    class RecordingRecord(dict[str, object]):
        def get(self, key: object, default: object = None) -> object:
            calls.append((key, default))
            return super().get(key, default)

    assert _generated_file_sort_key(RecordingRecord(table="a", language="en", path="a")) == (
        "a",
        "False",
        "en",
        "a",
    )
    assert calls == [("language", ""), ("language", "")]


def test_run_release_uses_default_selector_and_forwards_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_plan(
        data_root: DataRoot | Path,
        *,
        dataset_version: str,
        batch_size: int,
    ) -> LanguageSplitReleasePlan:
        calls.update(
            data_root=data_root,
            dataset_version=dataset_version,
            batch_size=batch_size,
        )
        return LanguageSplitReleasePlan(
            data_root=tmp_path,
            dataset_version=dataset_version,
            batch_size=batch_size,
            releases=(),
        )

    monkeypatch.setattr(language_split_release, "plan_language_split_release", fake_plan)

    result = run_language_split_release(tmp_path, batch_size=7)

    assert calls == {"data_root": tmp_path, "dataset_version": "both", "batch_size": 7}
    assert result.plan.dataset_version == "both"
    assert result.plan.batch_size == 7


@pytest.mark.parametrize("version", [LanguageSplitVersion.V1, LanguageSplitVersion.V2])
def test_generate_version_forwards_paths_and_batch_size(
    tmp_path: Path, version: LanguageSplitVersion, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_root = tmp_path / "processed"
    output_root = processed_root / "language_splits"
    plan = SimpleNamespace(
        version=version,
        processed_root=processed_root,
        output_root=output_root,
    )
    calls: list[tuple[object, ...]] = []

    if version is LanguageSplitVersion.V1:
        from osm_polygon_wikidata_only.hf import v1_language_splits

        def fake_v1(root: Path, destination: Path, *, batch_size: int) -> object:
            calls.append((root, destination, batch_size))
            return "v1-generated"

        monkeypatch.setattr(v1_language_splits, "generate_v1_language_splits", fake_v1)
    else:
        from osm_polygon_wikidata_only.v2 import language_splits as v2_language_splits

        def fake_v2(root: Path, *, output_root: Path, batch_size: int) -> object:
            calls.append((root, output_root, batch_size))
            return "v2-generated"

        monkeypatch.setattr(v2_language_splits, "build_v2_language_splits", fake_v2)

    assert _generate_version(plan, batch_size=7) == f"{version.value}-generated"
    assert calls == [(processed_root, output_root, 7)]


def test_generated_payload_prefers_the_published_manifest_path(tmp_path: Path) -> None:
    plan = SimpleNamespace(
        version=LanguageSplitVersion.V2,
        to_dict=lambda data_root: {
            "manifest_path": "planned/manifest.json",
            "expected_files": [],
        },
    )
    generated = SimpleNamespace(
        manifest_path=tmp_path / "processed_v2/manifests/generated.json",
        inventory=cast(LanguageInventory, object()),
        files=(),
        processed_root=tmp_path / "processed_v2",
    )

    fixture_root = _write_v2_fixture(tmp_path)
    release_plan = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1)
    plan = release_plan.releases[0]
    generated.inventory = plan.inventory
    generated.processed_root = fixture_root

    payload = _generated_release_payload(plan, generated, tmp_path)

    assert payload["manifest_path"] == "processed_v2/manifests/generated.json"


def test_generated_payload_passes_recomputed_inventory_to_plan_serializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_v2_fixture(tmp_path)
    planned = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1).releases[0]
    actual = replace(planned.inventory, dataset_id="recomputed-dataset")
    captured: dict[str, object] = {}

    def fake_to_dict(
        data_root: Path, *, inventory: LanguageInventory | None = None
    ) -> dict[str, object]:
        captured["data_root"] = data_root
        captured["inventory"] = inventory
        return {"expected_files": []}

    plan = SimpleNamespace(
        version=LanguageSplitVersion.V2,
        inventory=planned.inventory,
        processed_root=planned.processed_root,
        to_dict=fake_to_dict,
    )
    generated = SimpleNamespace(
        manifest_path=root / "manifests/language_splits.json",
        inventory=actual,
        files=(),
        processed_root=root,
    )
    monkeypatch.setattr(language_split_release, "_recompute_inventory", lambda _: actual)

    _generated_release_payload(plan, generated, tmp_path)

    assert captured == {"data_root": tmp_path, "inventory": actual}


def test_generated_inventory_without_attribute_is_rejected(tmp_path: Path) -> None:
    _write_v2_fixture(tmp_path)
    plan = plan_language_split_release(tmp_path, dataset_version="v2", batch_size=1).releases[0]

    with pytest.raises(LanguageSplitReleaseError, match="missing its source inventory"):
        _validate_generated_inventory(plan, SimpleNamespace(), plan.inventory)


@pytest.mark.parametrize("version", [LanguageSplitVersion.V1, LanguageSplitVersion.V2])
def test_recompute_inventory_passes_planned_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: LanguageSplitVersion,
) -> None:
    processed_root = tmp_path / version.value
    source_manifest = "manifests/processed_pbfs.json"
    expected_inventory = object()
    calls: list[tuple[Path, DatasetContract, Path]] = []

    def fake_build_inventory(
        root: Path,
        contract: DatasetContract,
        *,
        manifest_path: Path,
    ) -> object:
        calls.append((root, contract, manifest_path))
        return expected_inventory

    monkeypatch.setattr(language_split_release, "build_language_inventory", fake_build_inventory)
    plan = SimpleNamespace(
        version=version,
        processed_root=processed_root,
        inventory=SimpleNamespace(source_manifest=source_manifest),
    )

    assert _recompute_inventory(plan) is expected_inventory
    assert calls == [
        (
            processed_root,
            DatasetContract.V1 if version is LanguageSplitVersion.V1 else DatasetContract.V2,
            processed_root / source_manifest,
        )
    ]


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
    assert payload["dry_run"] is False
    assert payload["status"] == "generated"
    assert release["manifest_path"] == (
        "processed/language_splits/manifests/language_splits_v1.json"
    )
    assert all(
        set(item) == {"table", "configuration", "split", "path", "row_count", "sha256", "columns"}
        for item in cast(list[dict[str, object]], release["files"])
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
    assert payload["dry_run"] is False
    assert payload["status"] == "generated"
    assert release["manifest_path"] == ("processed_v2/manifests/language_splits.json")
    assert all(
        set(item)
        == {
            "configuration",
            "language",
            "path",
            "row_count",
            "sha256",
            "source_files",
            "split",
            "table",
        }
        for item in cast(list[dict[str, object]], release["files"])
    )
    assert all(
        "_" not in cast(str, item["split"]) and cast(str, item["split"]).startswith("lang-")
        for item in cast(list[dict[str, object]], release["files"])
    )
    assert not (tmp_path / "processed_v2/language_splits/polygons").exists()


def test_generated_v2_payload_rejects_source_fingerprint_mismatch(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    result = run_language_split_release(DataRoot(tmp_path), dataset_version="v2", batch_size=1)

    source = root / "wikipedia/documents/a-latest.parquet"
    rows = pq.read_table(source).to_pylist()
    rows[0]["title"] = "Changed after generation"
    _write_table(source, rows, wikipedia_document_v2_schema())

    with pytest.raises(LanguageSplitReleaseError, match="fingerprint"):
        result.to_payload()


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


def test_v2_language_split_result_loads_with_standard_datasets_loader(tmp_path: Path) -> None:
    _write_v2_fixture(tmp_path)

    result = run_language_split_release(DataRoot(tmp_path), dataset_version="v2", batch_size=1)
    generated = cast(V2LanguageSplitResult, result.generated[0])
    french_files = [
        generated.processed_root / file.path
        for file in generated.files
        if file.table.value == "wikipedia_documents" and file.language == "fr"
    ]
    assert french_files

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
                    "    data_files=sys.argv[1:-1],",
                    '    split="train",',
                    "    cache_dir=sys.argv[-1],",
                    ")",
                    'print(json.dumps({"num_rows": dataset.num_rows, "document_id": list(dataset["document_id"])}))',
                )
            ),
            *(str(path) for path in french_files),
            str(tmp_path / "hf-v2-cache"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"num_rows": 1, "document_id": ["doc-fr"]}
