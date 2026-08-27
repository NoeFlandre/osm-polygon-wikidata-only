from pathlib import Path

import pytest

from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import LOCAL_V2_DATASET_HERO_FILE
from osm_polygon_wikidata_only.v2.publication import (
    _REGION_UPLOAD_BATCH_SIZE,
    metadata_publication_ops,
    region_publication_ops,
    remote_region_complete,
    sentence_publication_ops,
    upload_region_batches,
)


def _write_required_region_files(root: Path, stem: str) -> None:
    for relative in (
        f"polygons/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"polygon_document_links/{stem}.parquet",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_region_publication_requires_wikipedia_sections(tmp_path: Path) -> None:
    stem = "region-latest"
    for relative in (
        f"polygons/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"polygon_document_links/{stem}.parquet",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with pytest.raises(FileNotFoundError, match="wikipedia/sections"):
        region_publication_ops(tmp_path, stem)


def test_upload_region_batches_preserves_order_and_commit_bound(tmp_path: Path) -> None:
    stems = tuple(f"region-{index:02d}-latest" for index in range(_REGION_UPLOAD_BATCH_SIZE + 1))
    for stem in stems:
        _write_required_region_files(tmp_path, stem)
    uploads: list[tuple[list[str], str]] = []

    upload_region_batches(
        tmp_path,
        stems,
        upload=lambda ops, message: uploads.append(
            ([operation.path_in_repo for operation in ops], message)
        ),
        repair=False,
    )

    assert [len(paths) for paths, _message in uploads] == [_REGION_UPLOAD_BATCH_SIZE * 4, 4]
    assert uploads[0][0][:4] == [
        f"polygons/{stems[0]}.parquet",
        f"wikipedia/documents/{stems[0]}.parquet",
        f"wikipedia/sections/{stems[0]}.parquet",
        f"polygon_document_links/{stems[0]}.parquet",
    ]
    assert uploads[0][1] == (
        f"Add V2 regions {stems[0]} through {stems[_REGION_UPLOAD_BATCH_SIZE - 1]} "
        f"({_REGION_UPLOAD_BATCH_SIZE} regions)"
    )
    assert uploads[1][1] == f"Add V2 region {stems[-1]}"


def test_upload_region_batches_uses_repair_message(tmp_path: Path) -> None:
    stem = "region-latest"
    _write_required_region_files(tmp_path, stem)
    messages: list[str] = []

    upload_region_batches(
        tmp_path,
        [stem],
        upload=lambda _ops, message: messages.append(message),
        repair=True,
    )

    assert messages == [f"Repair V2 region {stem}"]


def test_remote_region_complete_requires_every_planned_path(tmp_path: Path) -> None:
    stem = "region-latest"
    _write_required_region_files(tmp_path, stem)
    paths = [operation.path_in_repo for operation in region_publication_ops(tmp_path, stem)]

    assert remote_region_complete(RemoteInventory(set(paths)), tmp_path, stem)
    assert not remote_region_complete(RemoteInventory(set(paths[:-1])), tmp_path, stem)
    assert not remote_region_complete(None, tmp_path, stem)


def test_metadata_publication_includes_v2_maps_and_card(tmp_path: Path) -> None:
    for relative in (
        "README.md",
        "manifests/processed_pbfs.json",
        "assets/coverage_map.png",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/v2_added_wikipedia_tag_documents.png",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    operations = metadata_publication_ops(tmp_path)
    assert operations[0].local_path == LOCAL_V2_DATASET_HERO_FILE
    assert [op.path_in_repo for op in operations] == [
        "assets/dataset_hero.png",
        "assets/coverage_map.png",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/v2_added_wikipedia_tag_documents.png",
        "manifests/processed_pbfs.json",
        "README.md",
    ]


def test_sentence_publication_includes_sidecars_manifest_and_card(tmp_path: Path) -> None:
    stem = "region-latest"
    for relative in (
        f"wikipedia/sentences/{stem}.parquet",
        f"wikivoyage/sentences/{stem}.parquet",
        "manifests/sentence_splitting.json",
        "README.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    operations = sentence_publication_ops(tmp_path, [stem])

    assert [op.path_in_repo for op in operations] == [
        f"wikipedia/sentences/{stem}.parquet",
        f"wikivoyage/sentences/{stem}.parquet",
        "manifests/sentence_splitting.json",
        "README.md",
    ]
