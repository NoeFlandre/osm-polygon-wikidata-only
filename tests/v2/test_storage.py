from pathlib import Path

import pytest

from osm_polygon_wikidata_only.v2.storage import (
    V2RegionArtifacts,
    load_v2_manifest,
    write_v2_region,
)


def test_write_region_creates_isolated_artifacts_and_manifest(tmp_path: Path) -> None:
    artifacts = write_v2_region(
        tmp_path,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
    )
    assert isinstance(artifacts, V2RegionArtifacts)
    assert artifacts.polygons_path == tmp_path / "polygons" / "region-latest.parquet"
    assert (
        artifacts.documents_path == tmp_path / "wikipedia" / "documents" / "region-latest.parquet"
    )
    assert artifacts.sections_path == tmp_path / "wikipedia" / "sections" / "region-latest.parquet"
    assert artifacts.links_path == tmp_path / "polygon_document_links" / "region-latest.parquet"
    assert artifacts.manifest_path == tmp_path / "manifests" / "processed_pbfs.json"
    entry = load_v2_manifest(tmp_path)["region-latest"]
    assert entry["contract_version"] == "wikipedia-tags-v2"
    assert entry["sections_path"] == "wikipedia/sections/region-latest.parquet"
    assert entry["row_counts"]["sections"] == 0


def test_failed_write_leaves_no_temporary_files_or_manifest(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.storage as storage

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(storage.pq, "write_table", fail)
    with pytest.raises(OSError, match="disk full"):
        write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    assert not list(tmp_path.rglob("*.tmp"))
    assert not (tmp_path / "manifests" / "processed_pbfs.json").exists()


def test_second_identical_write_is_byte_stable(tmp_path: Path) -> None:
    first = write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    hashes_before = first.file_hashes
    second = write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    assert second.file_hashes == hashes_before


def test_provisional_write_is_marked_until_v1_reconciliation(tmp_path: Path) -> None:
    write_v2_region(
        tmp_path,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
        v1_index_reconciled=False,
    )
    assert load_v2_manifest(tmp_path)["region-latest"]["v1_index_reconciled"] is False


def test_sections_use_the_exact_v1_section_schema(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    import pyarrow.parquet as pq

    from osm_polygon_wikidata_only.augmentation.schema import section_schema

    table = pq.read_table(tmp_path / "wikipedia/sections/region-latest.parquet")
    assert table.schema.equals(section_schema(), check_metadata=True)


def test_corrupt_utf8_manifest_raises_actionable_value_error(tmp_path: Path) -> None:
    """Corrupt V2 manifest bytes must fail as a named format error."""
    path = tmp_path / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="Malformed V2 manifest"):
        load_v2_manifest(tmp_path)


def test_write_region_rejects_conflicting_duplicate_documents(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate document identity"):
        write_v2_region(
            tmp_path,
            "region-latest",
            polygons=[],
            documents=[
                {"document_id": "d1", "title": "First"},
                {"document_id": "d1", "title": "Changed"},
            ],
            links=[],
        )
