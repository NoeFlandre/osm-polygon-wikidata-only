"""Freeze the upload file-list contracts.

The publication paths in ``cli.commands`` build lists of
``(local_path, remote_path)`` tuples in a deterministic order. This
module captures those lists as golden expectations so refactors that
reorder, drop, or add entries will fail loudly. The tests use stub
data only — no Parquet I/O, no network, no HF client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.cli import commands as commands_mod
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult


def _stub_process_result(tmp_path: Path) -> tuple[ProcessResult, DataRoot]:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygons = data_root.processed_polygons / "monaco-latest.parquet"
    articles = data_root.processed_articles / "monaco-latest.parquet"
    links = data_root.processed_links / "monaco-latest.parquet"
    manifest = data_root.processed_manifests / "processed_pbfs.json"
    for p in (polygons, articles, links, manifest):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    manifest.write_text("{}", encoding="utf-8")
    return (
        ProcessResult(
            polygons_path=polygons,
            articles_path=articles,
            polygon_articles_path=links,
            manifest_path=manifest,
            polygon_count=10,
            article_count=5,
            link_count=15,
            manifest_entry={"source_pbf": "monaco-latest.osm.pbf"},
            stage_timings_s={},
        ),
        data_root,
    )


def _stub_augmentation_result(processed_root: Path) -> AugmentationResult:
    """Build an :class:`AugmentationResult` whose paths live under ``processed_root``.

    Mirrors the real layout: ``data_root.processed/{wikipedia,wikivoyage,wikidata,...}``.
    """
    paths = {
        "wikipedia_documents_path": processed_root
        / "wikipedia"
        / "documents"
        / "monaco-latest.parquet",
        "wikipedia_sections_path": processed_root
        / "wikipedia"
        / "sections"
        / "monaco-latest.parquet",
        "wikivoyage_documents_path": processed_root
        / "wikivoyage"
        / "documents"
        / "monaco-latest.parquet",
        "wikivoyage_sections_path": processed_root
        / "wikivoyage"
        / "sections"
        / "monaco-latest.parquet",
        "wikidata_facts_path": processed_root / "wikidata" / "facts" / "monaco-latest.parquet",
        "manifest_path": processed_root
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    return AugmentationResult(**paths, counts={"wikipedia_documents": 1})


def _stub_augmentation_result_in_tmp(tmp_path: Path) -> AugmentationResult:
    """Helper for tests that pass ``tmp_path`` directly as processed_root."""
    return _stub_augmentation_result(tmp_path)


def test_augmentation_upload_files_includes_five_sidecars_manifest_and_readme(
    tmp_path: Path,
) -> None:
    aug = _stub_augmentation_result(tmp_path)
    readme = tmp_path / "README.md"
    files = commands_mod._augmentation_upload_files(aug, tmp_path, readme)
    remotes = [remote for _, remote in files]
    # 5 sidecars + 1 augmentation manifest + 1 README = 7
    assert len(files) == 7
    assert remotes[-1] == "README.md"
    assert "wikipedia/documents/monaco-latest.parquet" in remotes
    assert "wikipedia/sections/monaco-latest.parquet" in remotes
    assert "wikivoyage/documents/monaco-latest.parquet" in remotes
    assert "wikivoyage/sections/monaco-latest.parquet" in remotes
    assert "wikidata/facts/monaco-latest.parquet" in remotes
    assert "augmentation/manifests/augmentation_manifest.json" in remotes


def test_augmentation_upload_files_uses_processed_relative_paths(tmp_path: Path) -> None:
    aug = _stub_augmentation_result(tmp_path)
    readme = tmp_path / "README.md"
    files = commands_mod._augmentation_upload_files(aug, tmp_path, readme)
    remotes = sorted(remote for _, remote in files if remote != "README.md")
    expected_sorted = sorted(
        [
            "augmentation/manifests/augmentation_manifest.json",
            "wikipedia/documents/monaco-latest.parquet",
            "wikipedia/sections/monaco-latest.parquet",
            "wikivoyage/documents/monaco-latest.parquet",
            "wikivoyage/sections/monaco-latest.parquet",
            "wikidata/facts/monaco-latest.parquet",
        ]
    )
    assert remotes == expected_sorted


def test_coverage_refresh_required_returns_false_when_no_core() -> None:
    assert commands_mod._coverage_refresh_required(None) is False


def test_coverage_refresh_required_returns_true_when_core_present() -> None:
    assert commands_mod._coverage_refresh_required(object()) is True


def test_sync_upload_files_without_core_returns_augmentation_only(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    files = commands_mod._sync_upload_files(data_root, "owner/repo", "monaco-latest", aug, None)
    remotes = [remote for _, remote in files]
    assert remotes == [
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "README.md",
    ]


def test_sync_upload_files_with_core_prepends_core_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core artifacts must be the FIRST entries of the unified upload list."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)

    # Stub the asset generators and the README writer so we don't depend
    # on matplotlib, PyArrow contents, or real dataset stats.
    monkeypatch.setattr(
        commands_mod,
        "_generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1],
    )
    monkeypatch.setattr(
        commands_mod,
        "_generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1],
    )
    monkeypatch.setattr(commands_mod, "load_centroids_from_parquet", lambda _dir: ([], []))
    monkeypatch.setattr(commands_mod, "ensure_world_land", lambda _dir: None)
    monkeypatch.setattr(
        commands_mod,
        "generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest,
    )
    monkeypatch.setattr(commands_mod, "_write_readme_snapshot", lambda *a, **kw: None)

    files = commands_mod._sync_upload_files(data_root, "owner/repo", "monaco-latest", aug, core)
    remotes = [remote for _, remote in files]

    # Core parquet + manifest + assets come first, in deterministic order.
    assert remotes[0] == "polygons/monaco-latest.parquet"
    assert remotes[1] == "articles/monaco-latest.parquet"
    assert remotes[2] == "polygon_articles/monaco-latest.parquet"
    assert remotes[3] == "manifests/processed_pbfs.json"
    assert remotes[4] == "assets/geographic_wikipedia_text_coverage.png"
    assert remotes[5] == "assets/geographic_polygon_count.png"
    assert remotes[6] == "coverage_map.png"
    # The augmentation block follows in the same fixed order as the no-core case.
    assert remotes[7:] == [
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "README.md",
    ]
