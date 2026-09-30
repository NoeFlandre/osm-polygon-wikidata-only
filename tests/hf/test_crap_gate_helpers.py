from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmentation_stats
from osm_polygon_wikidata_only.hf._stats_release import manifest, remote
from osm_polygon_wikidata_only.hf._uploader import operations, token
from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.coverage_map import _load_centroid_file


def test_manifest_parsers_reject_invalid_container_and_metadata_shapes(tmp_path: Path) -> None:
    with pytest.raises(manifest.StatsReleaseError, match="not an object"):
        manifest._manifest_entries(None, tmp_path / "manifest.json")
    with pytest.raises(manifest.StatsReleaseError, match="no region entries"):
        manifest._manifest_entries({"regions": []}, tmp_path / "manifest.json")

    path = tmp_path / "manifest.json"
    metadata = {
        "polygons_path": "polygons/region-latest.parquet",
        "polygon_count": 7,
    }
    assert manifest._manifest_polygon_metadata("region-latest.osm.pbf", metadata, path) == (
        "polygons/region-latest.parquet",
        "region-latest.osm.pbf",
        7,
    )
    with pytest.raises(manifest.StatsReleaseError, match="must point to"):
        manifest._manifest_polygon_metadata(
            "region-latest.osm.pbf", {**metadata, "polygons_path": "wrong.parquet"}, path
        )
    with pytest.raises(manifest.StatsReleaseError, match="source_pbf"):
        manifest._manifest_polygon_metadata(
            "region-latest.osm.pbf", {**metadata, "source_pbf": 3}, path
        )


def test_legacy_custom_verifier_signature_remains_supported() -> None:
    def legacy_verifier(repo_id: str, _files: tuple) -> str:
        return repo_id

    assert remote._invoke_custom_verifier(
        cast(Any, legacy_verifier), "owner/repo", (), "revision"
    ) == ("owner/repo")


def test_custom_verifier_falls_back_when_signature_cannot_be_inspected() -> None:
    def opaque_verifier(repo_id: str, _files: tuple) -> str:
        return repo_id

    setattr(opaque_verifier, "__signature__", object())
    assert remote._invoke_custom_verifier(
        cast(Any, opaque_verifier), "owner/repo", (), "revision"
    ) == ("owner/repo")


def test_publication_operation_translation_covers_add_delete_and_invalid_paths(
    tmp_path: Path,
) -> None:
    local = tmp_path / "source.txt"
    local.write_text("data", encoding="utf-8")
    add = operations._translate_publication_op(
        PublicationOp("add", "remote.txt", local_path=local),
        cast(Any, SimpleNamespace),
        cast(Any, SimpleNamespace),
    )
    assert add.path_in_repo == "remote.txt"
    assert add.path_or_fileobj == str(local)

    delete = operations._translate_publication_op(
        PublicationOp("delete", "remote.txt"),
        cast(Any, SimpleNamespace),
        cast(Any, SimpleNamespace),
    )
    assert delete.path_in_repo == "remote.txt"
    with pytest.raises(UploadError, match="does not exist"):
        operations._translate_publication_op(
            PublicationOp("add", "missing.txt", local_path=tmp_path / "missing.txt"),
            cast(Any, SimpleNamespace),
            cast(Any, SimpleNamespace),
        )

    malformed = SimpleNamespace(action="replace", path_in_repo="x", local_path=None)
    with pytest.raises(UploadError, match="Unknown action"):
        cast(Any, operations._translate_publication_op)(malformed, SimpleNamespace, SimpleNamespace)


def test_hf_token_loader_handles_token_and_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = ModuleType("huggingface_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    setattr(hub, "get_token", lambda: "token")
    assert token._load_hf_token() == "token"
    setattr(hub, "get_token", lambda: "")
    assert token._load_hf_token() is None

    def broken_backend() -> str:
        raise RuntimeError("cache unavailable")

    setattr(hub, "get_token", broken_backend)
    assert token._load_hf_token() is None


def test_augmentation_scanner_dispatch_and_path_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    calls: list[str] = []
    for kind, scanner, relative in (
        ("documents", "_scan_documents_file", "wikipedia/documents/a.parquet"),
        ("sections", "_scan_sections_file", "wikipedia/sections/a.parquet"),
        ("facts", "_scan_facts_file", "wikidata/facts/a.parquet"),
    ):
        monkeypatch.setattr(
            augmentation_stats,
            scanner,
            lambda _root, _path, selected=kind: calls.append(selected),
        )
        assert augmentation_stats._scan_one_file(processed, processed / relative) is None
        assert calls[-1] == kind

    assert (
        augmentation_stats._scan_one_file(processed, processed / "unmanaged/file.parquet") is None
    )
    assert augmentation_stats._scan_one_file(processed, tmp_path / "outside.parquet") is None


@pytest.mark.parametrize("error", [OSError("unreadable"), KeyError("columns")])
def test_centroid_file_scan_skips_unreadable_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from osm_polygon_wikidata_only.hf import coverage_map

    def fail(_path: Path):
        raise error

    monkeypatch.setattr(coverage_map, "open_parquet", fail)
    assert _load_centroid_file(tmp_path / "corrupt.parquet") == ([], [])
