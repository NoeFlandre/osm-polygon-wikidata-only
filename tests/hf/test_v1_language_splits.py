"""RED-to-GREEN tests for the V1 language partition release stage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_wikidata_only.hf.v1_language_splits as v1_language_splits
from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import POLYGON_COLUMNS, empty_row, polygon_schema
from osm_polygon_wikidata_only.hf.v1_language_splits import (
    V1_LANGUAGE_SPLIT_MANIFEST,
    V1LanguageSplitError,
    _previous_partition_path,
    _restore_files,
    _validate_partition_counts,
    generate_v1_language_splits,
)


def _row_for_schema(schema: pa.Schema, **values: object) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        elif pa.types.is_boolean(field.type):
            row[field.name] = False
        elif pa.types.is_list(field.type):
            row[field.name] = []
        else:
            row[field.name] = ""
    row.update(values)
    return row


def _polygon_row(polygon_id: str, *, best_language: str) -> dict[str, object]:
    row = empty_row(POLYGON_COLUMNS)
    row.update(
        {
            "polygon_id": polygon_id,
            "region": "fixture",
            "source_pbf": f"{polygon_id}.osm.pbf",
            "osm_type": "way",
            "osm_id": int(polygon_id),
            "wikidata": f"Q{polygon_id}",
            "best_language": best_language,
        }
    )
    return row


def _write_table(
    path: Path,
    rows: list[dict[str, object]],
    schema: pa.Schema,
    *,
    row_group_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="snappy",
        row_group_size=row_group_size,
    )


def _write_manifest(processed: Path, stems: tuple[str, ...] = ("a", "b")) -> None:
    entries = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": "fixture",
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
        }
        for stem in stems
    }
    path = processed / "manifests/processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries) + "\n", encoding="utf-8")


def _write_fixture(processed: Path, *, row_group_size: int = 1) -> None:
    link_schema = polygon_document_link_schema()
    _write_table(
        processed / "polygons/a.parquet",
        [_polygon_row("1", best_language="en")],
        polygon_schema(),
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "polygons/b.parquet",
        [_polygon_row("2", best_language="en")],
        polygon_schema(),
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "polygon_articles/a.parquet",
        [
            _row_for_schema(
                link_schema,
                polygon_id="1",
                document_id="a-en",
                project="wikipedia",
                language=" en ",
                wikidata="Q1",
                source_pbf="a.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=1,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="1",
                document_id="a-fr",
                project="wikipedia",
                language="fr",
                wikidata="Q1",
                source_pbf="a.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=1,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="1",
                document_id="a-null",
                project="wikipedia",
                language=None,
                wikidata="Q1",
                source_pbf="a.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=1,
            ),
        ],
        link_schema,
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "polygon_articles/b.parquet",
        [
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-fr",
                project="wikipedia",
                language="FR",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-de",
                project="wikipedia",
                language="de",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-blank",
                project="wikivoyage",
                language=" \t",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-malformed",
                project="wikipedia",
                language="en/fr",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            ),
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-legacy",
                project="wikipedia",
                language="be_x_old",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            ),
        ],
        link_schema,
        row_group_size=row_group_size,
    )

    document_v1_schema = wikipedia_document_schema()
    _write_table(
        processed / "wikipedia/documents/a.parquet",
        [
            _row_for_schema(
                document_v1_schema,
                document_id="doc-fr",
                language="fr",
                full_text="",
                source_api="fixture",
                fetch_status="ok",
            ),
            _row_for_schema(
                document_v1_schema,
                document_id="doc-de",
                language="de",
                full_text="German text",
                source_api="fixture",
                fetch_status="ok",
            ),
        ],
        document_v1_schema,
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "wikipedia/sections/a.parquet",
        [
            _row_for_schema(
                section_schema(),
                section_id="section-fr",
                document_id="doc-fr",
                language="fr",
                text="",
            )
        ],
        section_schema(),
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "wikivoyage/documents/a.parquet",
        [_row_for_schema(document_schema(), document_id="voy-unknown", language="simple")],
        document_schema(),
        row_group_size=row_group_size,
    )
    _write_table(
        processed / "wikivoyage/sections/a.parquet",
        [_row_for_schema(section_schema(), section_id="voy-section", language="be_x_old")],
        section_schema(),
        row_group_size=row_group_size,
    )
    _write_manifest(processed)


def _rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def _partition_path(output: Path, configuration: str, split: str) -> Path:
    return output / "data" / configuration / f"{split}-00000-of-00001.parquet"


def test_v1_partition_routes_each_row_by_its_own_language(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    release = generate_v1_language_splits(processed, output, batch_size=1)

    french_links = _rows(_partition_path(output, "polygon_articles_by_language", "lang-fr"))
    german_links = _rows(_partition_path(output, "polygon_articles_by_language", "lang-de"))

    assert [row["document_id"] for row in french_links] == ["a-fr", "b-fr"]
    assert [row["document_id"] for row in german_links] == ["b-de"]
    assert release.inventory.contract.value == "v1"
    assert "best_language" not in polygon_document_link_schema().names


def test_v1_partition_routes_invalid_values_to_unknown_and_conserves_rows(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    release = generate_v1_language_splits(processed, output, batch_size=2)

    unknown_links = _rows(_partition_path(output, "polygon_articles_by_language", "lang-unknown"))
    legacy_links = _rows(_partition_path(output, "polygon_articles_by_language", "lang-be-tarask"))
    link_inventory = release.inventory.table("polygon_articles")

    assert [row["document_id"] for row in unknown_links] == [
        "a-null",
        "b-blank",
        "b-malformed",
    ]
    assert [row["language"] for row in unknown_links] == [None, " \t", "en/fr"]
    assert [row["document_id"] for row in legacy_links] == ["b-legacy"]
    assert sum(bucket.row_count for bucket in link_inventory.buckets) == link_inventory.row_count
    assert (
        sum(item.row_count for item in release.files if item.table == "polygon_articles")
        == link_inventory.row_count
    )


def test_v1_partition_preserves_schema_identity_provenance_and_empty_text(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    generate_v1_language_splits(processed, output, batch_size=1)

    document_path = _partition_path(output, "wikipedia_documents_by_language", "lang-fr")
    section_path = _partition_path(output, "wikipedia_sections_by_language", "lang-fr")
    assert pq.read_schema(document_path).equals(wikipedia_document_schema(), check_metadata=True)
    assert pq.read_schema(section_path).equals(section_schema(), check_metadata=True)
    assert _rows(document_path)[0]["document_id"] == "doc-fr"
    assert _rows(document_path)[0]["full_text"] == ""
    assert _rows(section_path)[0]["section_id"] == "section-fr"


def test_v1_partition_is_byte_stable_for_unchanged_input(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed, row_group_size=8)

    first = generate_v1_language_splits(processed, output, batch_size=1)
    first_bytes = {item.path.relative_to(output): item.path.read_bytes() for item in first.files}
    first_manifest = (output / V1_LANGUAGE_SPLIT_MANIFEST).read_bytes()

    second = generate_v1_language_splits(processed, output, batch_size=3)
    second_bytes = {item.path.relative_to(output): item.path.read_bytes() for item in second.files}
    second_manifest = (output / V1_LANGUAGE_SPLIT_MANIFEST).read_bytes()

    assert first_bytes == second_bytes
    assert first_manifest == second_manifest


def test_v1_partition_conserves_rows_in_every_language_table(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    release = generate_v1_language_splits(processed, output, batch_size=2)

    for table_inventory in release.inventory.tables:
        emitted_rows = sum(
            item.row_count for item in release.files if item.table == table_inventory.table.value
        )
        assert emitted_rows == table_inventory.row_count


def test_v1_partition_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        generate_v1_language_splits(tmp_path / "processed", tmp_path / "release", batch_size=0)

    with pytest.raises(V1LanguageSplitError, match="separate from processed input"):
        generate_v1_language_splits(tmp_path, tmp_path, batch_size=1)

    output_file = tmp_path / "output-file"
    output_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(V1LanguageSplitError, match="not a directory"):
        generate_v1_language_splits(tmp_path / "processed", output_file, batch_size=1)


def test_v1_partition_removes_stale_splits_on_a_deterministic_rerun(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    source_bytes = {
        path.relative_to(processed): path.read_bytes() for path in processed.rglob("*.parquet")
    }
    generate_v1_language_splits(processed, output, batch_size=1)
    stale_paths = (
        _partition_path(output, "polygon_articles_by_language", "lang-de"),
        _partition_path(output, "polygon_articles_by_language", "lang-be-tarask"),
    )
    assert all(path.is_file() for path in stale_paths)

    link_schema = polygon_document_link_schema()
    _write_table(
        processed / "polygon_articles/b.parquet",
        [
            _row_for_schema(
                link_schema,
                polygon_id="2",
                document_id="b-fr",
                project="wikipedia",
                language="FR",
                wikidata="Q2",
                source_pbf="b.osm.pbf",
                region="fixture",
                osm_type="way",
                osm_id=2,
            )
        ],
        link_schema,
        row_group_size=1,
    )
    generate_v1_language_splits(processed, output, batch_size=2)

    assert all(not path.exists() for path in stale_paths)
    assert {
        path.relative_to(processed): path.read_bytes()
        for path in processed.rglob("*.parquet")
        if path != processed / "polygon_articles/b.parquet"
    } == {
        path: content
        for path, content in source_bytes.items()
        if path != Path("polygon_articles/b.parquet")
    }


def test_v1_partition_validates_conservation_and_previous_paths(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)
    release = generate_v1_language_splits(processed, output, batch_size=1)
    inventory = release.inventory.table("polygon_articles")
    expected = {bucket.language: bucket.row_count for bucket in inventory.buckets}

    with pytest.raises(V1LanguageSplitError, match="row count changed"):
        _validate_partition_counts(inventory, inventory.row_count + 1, expected)
    with pytest.raises(V1LanguageSplitError, match="partition counts changed"):
        _validate_partition_counts(inventory, inventory.row_count, {})

    assert _previous_partition_path(output, None) is None
    assert _previous_partition_path(output, {"path": "../escape.parquet"}) is None
    assert _previous_partition_path(output, {"path": "data/config/not-a-parquet.txt"}) is None
    assert (
        _previous_partition_path(output, {"path": "data/config/lang-fr.parquet"})
        == (output / "data/config/lang-fr.parquet").resolve()
    )


def test_v1_partition_restores_backed_up_files_after_an_install_failure(tmp_path: Path) -> None:
    final = tmp_path / "data/final.parquet"
    backup = tmp_path / "data/final.parquet.backup"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"new")
    backup.write_bytes(b"old")

    _restore_files([final], {final: backup})

    assert final.read_bytes() == b"old"
    assert not backup.exists()


def test_v1_partition_restores_a_partial_backup_when_the_next_backup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    final_a = release / "data/a.parquet"
    final_b = release / "data/b.parquet"
    temporary_a = tmp_path / "temporary-a.parquet"
    temporary_b = tmp_path / "temporary-b.parquet"
    final_a.parent.mkdir(parents=True)
    final_a.write_bytes(b"old-a")
    final_b.write_bytes(b"old-b")
    temporary_a.write_bytes(b"new-a")
    temporary_b.write_bytes(b"new-b")

    original_backup = v1_language_splits._backup_existing
    calls = 0

    def fail_on_second_backup(path: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture backup failure")
        return original_backup(path)

    monkeypatch.setattr(v1_language_splits, "_backup_existing", fail_on_second_backup)
    with pytest.raises(OSError, match="fixture backup failure"):
        v1_language_splits._install_staged_files(
            release,
            {final_a: temporary_a, final_b: temporary_b},
        )

    assert final_a.read_bytes() == b"old-a"
    assert final_b.read_bytes() == b"old-b"
    assert not temporary_a.exists()
    assert not temporary_b.exists()


def test_v1_partition_removes_new_files_after_a_partial_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed, row_group_size=8)

    def fail_after_first_install(
        staged: dict[Path, Path],
        installed: list[Path] | None = None,
    ) -> list[Path]:
        installed = [] if installed is None else installed
        for final, temporary in sorted(staged.items(), key=lambda item: item[0].as_posix()):
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, final)
            installed.append(final)
            if len(installed) == 1:
                raise OSError("fixture install failure")
        return installed

    monkeypatch.setattr(v1_language_splits, "_install_files", fail_after_first_install)
    with pytest.raises(OSError, match="fixture install failure"):
        generate_v1_language_splits(processed, output, batch_size=1)

    assert not [path for path in output.rglob("*") if path.is_file()]
    assert not list(output.parent.glob(f".{output.name}-*"))


def test_v1_partition_is_readable_by_standard_hugging_face_loader(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "release"
    _write_fixture(processed)

    generate_v1_language_splits(processed, output, batch_size=1)
    path = _partition_path(output, "wikipedia_documents_by_language", "lang-fr")

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(path)},
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
    )

    assert dataset["document_id"] == ["doc-fr"]
    assert dataset["language"] == ["fr"]
