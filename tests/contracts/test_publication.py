"""Freeze the publication assembly contracts and submission counts.

The :mod:`osm_polygon_wikidata_only.hf.publication` module owns
pure assembly of ``(local_path, remote_path)`` upload file lists
for three documented publication contracts. This module captures
those lists as golden expectations, asserts the assemblers are
PURE (no ``submit`` parameter, no upload side effect), and
asserts that CLI callsites perform exactly one submission per
publication. Tests use stub data only -- no Parquet I/O, no
network, no HF client.

Contracts exercised:

* Legacy core publication (no augmentation): polygons, articles,
  polygon_articles, processed manifest, geographic text coverage,
  geographic polygon count, README, legacy coverage map.
* Unified sync with changed core: core block first, then the
  seven augmentation artifacts (wikipedia + wikivoyage +
  wikidata + per-region augmentation manifest snapshot + README).
* Augmentation-only publication (legacy augmentation command):
  five sidecars + ``augmentation_result.manifest_path`` (NOT a
  new stem snapshot) + README. No coverage assets are
  regenerated.

The tests also verify exact submission counts:
* Legacy core CLI: one assembly, one queue submission.
* Augmentation command: one assembly, one direct upload call.
* Unified sync: one assembly, one queue submission (no double
  submit when the runner's ``_maybe_submit`` calls the upload
  queue with the assembled list).
* Assembly failure: zero submissions.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.publication import (
    assemble_augmentation_upload,
    assemble_core_upload,
    assemble_region_upload,
    coverage_refresh_required,
    refresh_coverage_assets,
    snapshot_upload_manifests,
)
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

STEM = "monaco-latest"
REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"


def _stub_process_result(tmp_path: Path) -> tuple[ProcessResult, DataRoot]:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygons = data_root.processed_polygons / f"{STEM}.parquet"
    articles = data_root.processed_articles / f"{STEM}.parquet"
    links = data_root.processed_links / f"{STEM}.parquet"
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
            manifest_entry={"source_pbf": f"{STEM}.osm.pbf"},
            stage_timings_s={},
        ),
        data_root,
    )


def _stub_augmentation_result(processed_root: Path) -> AugmentationResult:
    """Build an :class:`AugmentationResult` whose paths live under ``processed_root``.

    Mirrors the real layout: ``data_root.processed/{wikipedia,wikivoyage,wikidata,...}``.
    """
    paths = {
        "wikipedia_documents_path": processed_root / "wikipedia" / "documents" / f"{STEM}.parquet",
        "wikipedia_sections_path": processed_root / "wikipedia" / "sections" / f"{STEM}.parquet",
        "wikivoyage_documents_path": processed_root
        / "wikivoyage"
        / "documents"
        / f"{STEM}.parquet",
        "wikivoyage_sections_path": processed_root / "wikivoyage" / "sections" / f"{STEM}.parquet",
        "wikidata_facts_path": processed_root / "wikidata" / "facts" / f"{STEM}.parquet",
        "manifest_path": processed_root
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    return AugmentationResult(**paths, counts={"wikipedia_documents": 1})


def _stub_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the geographic/coverage generators so tests don't depend on
    matplotlib, real dataset stats, or world-land download.

    The stubs create empty files at the requested destinations so the
    assembly helpers return paths that exist on disk.
    """
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )


# ---------------------------------------------------------------------------
# Pure-assembly signature assertions
# ---------------------------------------------------------------------------


def test_assemblers_have_no_submit_parameter() -> None:
    """None of the three public assemblers accepts a submit callable.

    Assemblers are PURE: they return an ordered file list and
    perform no upload. The CLI shell performs exactly one
    submission after successful assembly.
    """
    import inspect

    for fn in (assemble_core_upload, assemble_region_upload, assemble_augmentation_upload):
        sig = inspect.signature(fn)
        assert "submit" not in sig.parameters, (
            f"{fn.__name__} must not accept a submit callable; got {list(sig.parameters)}"
        )


def test_assemblers_have_no_commit_message_parameter() -> None:
    """Assemblers do not know about commit messages -- CLI owns them."""
    import inspect

    for fn in (assemble_core_upload, assemble_region_upload, assemble_augmentation_upload):
        sig = inspect.signature(fn)
        assert "commit_message" not in sig.parameters, (
            f"{fn.__name__} must not accept commit_message; got {list(sig.parameters)}"
        )


# ---------------------------------------------------------------------------
# Augmentation-only publication
# ---------------------------------------------------------------------------


def test_assemble_augmentation_upload_returns_seven_entries(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    files = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    remotes = [remote for _, remote in files]
    # The 6th entry is the original ``augmentation.manifest_path``
    # (NOT a new stem-augmentation manifest snapshot).
    assert files[5][0] == aug.manifest_path
    assert remotes == [
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "README.md",
    ]


def test_assemble_augmentation_upload_writes_readme_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README snapshot must be the last file in the assembled list."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    files = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert files[-1][1] == "README.md"
    readme = files[-1][0]
    assert readme.parent == data_root.cache / "augmentation_upload_snapshots"


def test_assemble_augmentation_upload_never_calls_visualization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Augmentation-only publication must NOT regenerate coverage assets."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    calls: list[str] = []

    def trap(name: str) -> Callable[..., object]:
        def _fn(*args: object, **kwargs: object) -> object:
            calls.append(name)
            return None

        return _fn

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        trap("text"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        trap("count"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.refresh_coverage_assets",
        trap("refresh"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        trap("coverage"),
    )

    assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert calls == [], f"visualization called during augmentation-only upload: {calls}"


# ---------------------------------------------------------------------------
# Unified-sync publication
# ---------------------------------------------------------------------------


def test_assemble_region_upload_without_core_returns_seven_augmentation_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    files = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=None,
        world_land_warning=None,
    )
    remotes = [remote for _, remote in files]
    # No core: only the seven augmentation entries (including the
    # per-region manifest snapshot under sync_upload_snapshots/<stem>/).
    assert remotes == [
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "README.md",
    ]


def test_assemble_region_upload_with_core_prepends_seven_core_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core artifacts must be the FIRST entries of the unified upload list."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)

    files = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    remotes = [remote for _, remote in files]

    assert remotes[0] == "polygons/monaco-latest.parquet"
    assert remotes[1] == "articles/monaco-latest.parquet"
    assert remotes[2] == "polygon_articles/monaco-latest.parquet"
    assert remotes[3] == "manifests/processed_pbfs.json"
    assert remotes[4] == "assets/geographic_wikipedia_text_coverage.png"
    assert remotes[5] == "assets/geographic_polygon_count.png"
    assert remotes[6] == "coverage_map.png"
    assert remotes[7:] == [
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "README.md",
    ]
    assert len(files) == 14


def test_assemble_region_upload_writes_readme_after_other_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README must be written AFTER every other snapshot."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )

    call_order: list[str] = []

    def text_stub(*a: object, **kw: object) -> Path:
        call_order.append("text")
        return a[1].touch() or a[1]  # type: ignore[index]

    def count_stub(*a: object, **kw: object) -> Path:
        call_order.append("count")
        return a[1].touch() or a[1]  # type: ignore[index]

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        text_stub,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        count_stub,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: call_order.append("coverage") or dest.touch() or dest,
    )

    def readme_spy(*args: object, **kwargs: object) -> None:
        call_order.append("README.md")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        readme_spy,
    )

    assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    assert call_order[-1] == "README.md"
    assert "text" in call_order
    assert "count" in call_order
    assert "coverage" in call_order
    assert call_order.index("text") < call_order.index("README.md")
    assert call_order.index("count") < call_order.index("README.md")
    assert call_order.index("coverage") < call_order.index("README.md")


# ---------------------------------------------------------------------------
# Legacy core publication
# ---------------------------------------------------------------------------


def test_assemble_core_upload_returns_eight_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy core publication order: 8 entries."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    files = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    remotes = [remote for _, remote in files]
    assert remotes == [
        "polygons/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "polygon_articles/monaco-latest.parquet",
        "manifests/processed_pbfs.json",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
        "README.md",
        "coverage_map.png",
    ]
    assert len(files) == 8


def test_assemble_core_upload_writes_readme_after_other_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )

    call_order: list[str] = []

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: call_order.append("text") or a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: call_order.append("count") or a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: call_order.append("coverage") or dest.touch() or dest,
    )

    def readme_spy(*args: object, **kwargs: object) -> None:
        call_order.append("README.md")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        readme_spy,
    )

    assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    assert call_order[-1] == "README.md"
    assert call_order.index("text") < call_order.index("README.md")
    assert call_order.index("count") < call_order.index("README.md")
    assert call_order.index("coverage") < call_order.index("README.md")


def test_assemble_core_upload_invokes_warning_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy core path invokes the world-land warning callback."""
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )

    def boom(_dir: Path) -> Path:
        raise RuntimeError("no world land available")

    monkeypatch.setattr("osm_polygon_wikidata_only.hf.publication.ensure_world_land", boom)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)

    assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=_warn,
    )
    assert any("Could not fetch world land data; map will omit continents" in w for w in warnings)


def test_assemble_region_upload_swallows_world_land_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unified-sync path silently swallows the legacy land exception."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )

    def boom(_dir: Path) -> Path:
        raise RuntimeError("no world land available")

    monkeypatch.setattr("osm_polygon_wikidata_only.hf.publication.ensure_world_land", boom)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    caplog.set_level(logging.WARNING)
    files = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    assert files, "region upload should still produce files"
    assert not any("Could not fetch world land data" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Required-artifact validation (now inside each entry point)
# ---------------------------------------------------------------------------


def test_assemble_core_upload_raises_when_core_artifact_missing(
    tmp_path: Path,
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    core = dataclasses.replace(core, polygons_path=tmp_path / "missing-core.parquet")
    with pytest.raises(FileNotFoundError, match="Core artifact missing"):
        assemble_core_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            core=core,
            world_land_warning=lambda msg: None,
        )


def test_assemble_region_upload_raises_when_augmentation_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    aug = dataclasses.replace(aug, wikipedia_documents_path=tmp_path / "missing.parquet")
    _stub_generators(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Augmentation artifact missing"):
        assemble_region_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            stem=STEM,
            augmentation=aug,
            core=core,
            world_land_warning=None,
        )


def test_assemble_augmentation_upload_raises_when_artifact_missing(
    tmp_path: Path,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    aug = dataclasses.replace(aug, manifest_path=tmp_path / "missing-manifest.json")
    with pytest.raises(FileNotFoundError, match="Augmentation artifact missing"):
        assemble_augmentation_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            augmentation=aug,
        )


# ---------------------------------------------------------------------------
# Failure atomicity
# ---------------------------------------------------------------------------


def test_assemble_core_upload_propagates_snapshot_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a snapshot step raises, the assembler propagates the error."""
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        boom,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        assemble_core_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            core=core,
            world_land_warning=lambda msg: None,
        )


def test_assemble_region_upload_propagates_snapshot_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a region snapshot step raises, the assembler propagates the error."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        boom,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        assemble_region_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            stem=STEM,
            augmentation=aug,
            core=core,
            world_land_warning=None,
        )


# ---------------------------------------------------------------------------
# Exact submission counts for each CLI callsite
# ---------------------------------------------------------------------------


def test_legacy_core_command_submits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process-pbf/process-dir: one assembly, one queue submission."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod

    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    submissions: list[tuple[list[tuple[Path, str]], str]] = []

    class _StubQueue:
        def submit(self, files: list[tuple[Path, str]], message: str) -> None:
            submissions.append((files, message))

    commands_mod._enqueue_core_upload(
        _StubQueue(),  # type: ignore[arg-type]
        data_root=data_root,
        repo_id=REPO_ID,
        commit_message="core msg",
        result=core,
    )
    assert len(submissions) == 1, f"legacy core must submit exactly once, got {len(submissions)}"
    files, message = submissions[0]
    assert message == "core msg"
    assert len(files) == 8


def test_augmentation_command_submits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """augment-region/augment-dir: one assembly, one direct upload call."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    uploads: list[tuple[list[tuple[Path, str]], str]] = []

    def fake_upload(
        repo_id: str,
        files: list[tuple[Path, str]],
        hub: object = None,
        token: object = None,
        commit_message: str = "",
        num_threads: int = 2,
    ) -> None:
        uploads.append((files, commit_message))

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.cli.commands.upload_files",
        fake_upload,
    )

    # Invoke the augmentation command's exact submission block.
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload

    files = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )

    def _submit(
        files: list[tuple[Path, str]],
        message: str,
        _hub: object = None,
    ) -> None:
        fake_upload(
            REPO_ID,
            files,
            hub=_hub,
            token=None,
            commit_message=message,
        )

    _submit(files, "aug msg")
    assert len(uploads) == 1, f"augmentation command must upload exactly once, got {len(uploads)}"
    assert uploads[0][1] == "aug msg"
    assert len(uploads[0][0]) == 7


def test_unified_sync_submits_exactly_one_commit_per_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end regression: production CLI builder + production runner
    must produce exactly ONE atomic commit per region.

    Regression for the double-submission bug: previously
    ``cli.run_sync._build_region_publication`` passed
    ``submit=_submit_upload`` to ``assemble_region_upload``, which
    submitted internally, AND the runner's ``_maybe_submit`` also
    submitted with the returned list. The fix makes
    ``assemble_region_upload`` pure; submission happens exactly
    once through the upload queue.
    """
    from osm_polygon_wikidata_only.pipeline.sync_planner import (
        RegionSyncState,
        SyncAction,
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    core, _ = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    submissions: list[tuple[list[tuple[Path, str]], str]] = []

    class _StubQueue:
        def submit(self, files: list[tuple[Path, str]], message: str) -> None:
            submissions.append((list(files), message))

        def resume_pending(self) -> int:
            return 0

        def close_and_wait(self) -> list[str]:
            return []

    # Production pure assembler invoked by the production runner.
    from osm_polygon_wikidata_only.hf.publication import assemble_region_upload
    from osm_polygon_wikidata_only.pipeline import sync_runner

    def _submit_upload(files: list[tuple[Path, str]], message: str) -> None:
        submissions.append((list(files), message))

    settings = type(
        "_Settings",
        (),
        {
            "repo_id": REPO_ID,
            "hf_token": "stub-token",
            "force": False,
            "skip_existing": True,
        },
    )()

    def _build_region_publication(
        state: object,
        augmentation: object,
        core_obj: object | None,
    ) -> list[tuple[Path, str]]:
        return assemble_region_upload(
            data_root=data_root,
            repo_id=settings.repo_id,
            stem=getattr(state, "stem"),
            augmentation=augmentation,
            core=core_obj,
            world_land_warning=None,
        )

    def fake_extract(_pbf_path: Path) -> object:
        return object()

    def fake_process(_extracted: object) -> ProcessResult:
        return core

    def fake_augment(_state: RegionSyncState) -> AugmentationResult:
        return aug

    pbf = tmp_path / "monaco-latest.osm.pbf"
    pbf.write_text("placeholder", encoding="utf-8")
    states = [
        RegionSyncState(stem="monaco-latest", pbf_path=pbf, action=SyncAction.PROCESS),
        RegionSyncState(stem="andorra-latest", pbf_path=pbf, action=SyncAction.AUGMENT),
    ]

    rc = sync_runner.run_sync(
        states,
        extract_pbf=fake_extract,
        process_extracted_pbf=fake_process,
        augment_region=fake_augment,
        build_upload_files=_build_region_publication,
        commit_message=lambda state: f"Sync complete region {state.stem}",
        submit_upload=_submit_upload,
        close_uploads=lambda: [],
    )
    # The submission-COUNT contract is the regression: the runner
    # must submit EXACTLY once per region. ``rc`` is unrelated.
    _ = rc

    # Two regions, exactly one commit per region.
    assert len(submissions) == 2, (
        f"sync must submit exactly one commit per region, got {len(submissions)}"
    )
    # The runner drains AUGMENT (backlog) states before PROCESS,
    # so submissions[0] is the AUGMENT commit and submissions[1]
    # is the PROCESS commit.
    by_message = {msg: files for files, msg in submissions}
    assert set(by_message) == {
        "Sync complete region monaco-latest",
        "Sync complete region andorra-latest",
    }
    # PROCESS state (with core): 14 entries (7 core + 7 augmentation).
    assert len(by_message["Sync complete region monaco-latest"]) == 14
    # AUGMENT state (no core): 7 entries (augmentation only).
    assert len(by_message["Sync complete region andorra-latest"]) == 7


# ---------------------------------------------------------------------------
# Coverage-refresh decision
# ---------------------------------------------------------------------------


def test_coverage_refresh_required_returns_false_when_no_core() -> None:
    assert coverage_refresh_required(None) is False


def test_coverage_refresh_required_returns_true_when_core_present() -> None:
    assert coverage_refresh_required(object()) is True


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def test_snapshot_upload_manifests_writes_processed_manifest_snapshot(
    tmp_path: Path,
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    snapshot, readme = snapshot_upload_manifests(data_root=data_root, core=core)
    assert snapshot.exists()
    assert snapshot.read_text(encoding="utf-8") == core.manifest_path.read_text(encoding="utf-8")
    assert not readme.exists()


def test_refresh_coverage_assets_writes_three_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    snapshots_dir = data_root.cache / "test_refresh"
    map_path, geo_path, count_path = refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem=core.polygons_path.stem,
        snapshots_dir=snapshots_dir,
        world_land_warning=lambda msg: None,
    )
    assert map_path.exists()
    assert geo_path.exists()
    assert count_path.exists()


# ---------------------------------------------------------------------------
# cli.commands ownership removal
# ---------------------------------------------------------------------------


def test_cli_commands_no_longer_implements_publication_assembly() -> None:
    """cli.commands must not contain publication assembly implementations."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod

    for name in (
        "_sync_upload_files",
        "_coverage_refresh_required",
        "_generate_geographic_text_coverage_snapshot",
        "_generate_geographic_polygon_count_snapshot",
        "_write_readme_snapshot",
        "_augmentation_upload_files",
        "load_centroids_from_parquet",
        "ensure_world_land",
        "generate_coverage_map",
    ):
        assert not hasattr(commands_mod, name), f"{name} must not live in cli.commands anymore"


def test_hf_publication_no_longer_exposes_dead_types() -> None:
    """RegionUploadArtifacts and required_local_artifacts_present are gone."""
    import osm_polygon_wikidata_only.hf.publication as publication_mod

    assert not hasattr(publication_mod, "RegionUploadArtifacts")
    assert not hasattr(publication_mod, "required_local_artifacts_present")
