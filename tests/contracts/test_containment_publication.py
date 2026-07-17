"""Atomic publication contracts for contained-region retirement."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf import publication
from osm_polygon_wikidata_only.hf._uploader.plan import add_op, delete_op
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.repo_layout import canonical_region_paths
from osm_polygon_wikidata_only.hf.uploader import UploadError, upload_files


def test_containment_publication_adds_parent_deletes_child_and_readme_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in canonical_region_paths("parent-latest"):
        path = data_root.processed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")
    retirement = data_root.processed_manifests / "containment_retirements.json"
    retirement.write_text('{"contract_version":"contained-region-v1","retired":{}}\n')
    readme = tmp_path / "README.md"
    readme.write_text("card")
    monkeypatch.setattr(
        publication,
        "assemble_metadata_only_upload",
        lambda **_kwargs: [add_op(readme, path_in_repo="README.md")],
    )
    ops = publication.assemble_containment_retirement_upload(
        data_root=data_root,
        repo_id="owner/dataset",
        parent_children={"parent-latest": ("child-latest",)},
    )
    additions = {op.path_in_repo for op in ops if op.action == "add"}
    deletions = {op.path_in_repo for op in ops if op.action == "delete"}
    assert set(canonical_region_paths("parent-latest").values()) <= additions
    assert deletions == set(canonical_region_paths("child-latest").values())
    assert "manifests/containment_retirements.json" in additions
    assert ops[-1].path_in_repo == "README.md"


def test_uploader_rejects_unpaired_canonical_region_delete(tmp_path: Path) -> None:
    replacement = tmp_path / "parent.parquet"
    replacement.write_bytes(b"x")
    with pytest.raises(UploadError, match="containment_retirements"):
        upload_files(
            "owner/dataset",
            ops=[
                add_op(replacement, path_in_repo="polygons/parent-latest.parquet"),
                delete_op("polygons/child-latest.parquet"),
            ],
            hub=StubHfHub(),
            commit_message="unsafe",
        )
