"""Split coverage tests (part 1)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.contracts.publication_support import *

# Every module in this split opts into the publication text-map stub.
pytestmark = pytest.mark.usefixtures("stub_combined_text_map")


def test_assemble_augmentation_upload_returns_combined_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    remotes = [op.path_in_repo for op in ops]
    # 6th op is the canonical ``manifests/augmentation_manifest.json``
    # add; its local source is the original ``augmentation.manifest_path``
    # (NOT a new stem-augmentation manifest snapshot). The 7th op
    # is the migration ``delete`` of the legacy augmentation path.
    add_ops = [op for op in ops if op.action == "add"]
    delete_ops = [op for op in ops if op.action == "delete"]
    assert len(add_ops) == 11
    assert len(delete_ops) == 4
    assert add_ops[5].local_path == aug.manifest_path
    assert add_ops[5].path_in_repo == "manifests/augmentation_manifest.json"
    assert {op.path_in_repo for op in delete_ops} == {
        "articles/monaco-latest.parquet",
        "augmentation/manifests/augmentation_manifest.json",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
    }
    assert remotes == [
        "wikipedia/documents/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "manifests/augmentation_manifest.json",
        "augmentation/manifests/augmentation_manifest.json",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
        "assets/dataset_hero.png",
        "stats.json",
        "README.md",
    ]
    assert len(ops) == 15


def test_assemble_augmentation_upload_writes_readme_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README snapshot must be the last ``add`` op in the assembled list."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    add_ops = [op for op in ops if op.action == "add"]
    assert add_ops[-1].path_in_repo == "README.md"
    readme = add_ops[-1].local_path
    assert readme is not None
    assert readme.parent == data_root.cache / "augmentation_upload_snapshots"


def test_assemble_augmentation_upload_refreshes_only_combined_visualization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Augmentation-only publication refreshes only its Wikivoyage-sensitive map."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        trap("density"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.refresh_coverage_assets",
        trap("refresh"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        trap("coverage"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_presence",
        trap("presence"),
    )

    assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert calls == ["presence", "density"]


def test_assemble_augmentation_upload_logs_world_land_fallback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Augmentation publication remains usable when land data is unavailable."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)

    def boom(_cache: Path) -> Path:
        raise RuntimeError("land unavailable")

    monkeypatch.setattr("osm_polygon_wikidata_only.hf.publication.ensure_world_land", boom)
    caplog.set_level(logging.WARNING)

    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )

    assert ops
    assert any("combined text map will omit continents" in r.getMessage() for r in caplog.records)


def test_assemble_metadata_only_upload_includes_manifest_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata refresh publishes the canonical and legacy manifest operations."""
    _core, data_root = _stub_process_result(tmp_path)
    augmentation_manifest = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    augmentation_manifest.parent.mkdir(parents=True, exist_ok=True)
    augmentation_manifest.write_text("{}", encoding="utf-8")

    def refresh_assets(**kwargs: object) -> tuple[Path, Path, Path]:
        snapshots_dir = kwargs["snapshots_dir"]
        assert isinstance(snapshots_dir, Path)
        paths = tuple(
            snapshots_dir / name for name in ("coverage.png", "presence.png", "density.png")
        )
        for path in paths:
            path.touch()
        return paths  # type: ignore[return-value]

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.refresh_coverage_assets", refresh_assets
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda _root, _repo, destination: destination.write_text("card", encoding="utf-8"),
    )

    ops = assemble_metadata_only_upload(data_root=data_root, repo_id=REPO_ID)
    remotes = [op.path_in_repo for op in ops]

    assert "manifests/augmentation_manifest.json" in remotes
    assert "augmentation/manifests/augmentation_manifest.json" in remotes
    assert remotes[-1] == "README.md"


def test_assemble_metadata_only_upload_requires_processed_manifest(tmp_path: Path) -> None:
    """Metadata publication fails closed when its source manifest is absent."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(FileNotFoundError, match="Local processed manifest is missing"):
        assemble_metadata_only_upload(data_root=data_root, repo_id=REPO_ID)


def test_assemble_region_upload_without_core_refreshes_combined_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    ops = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=None,
        world_land_warning=None,
    )
    remotes = [op.path_in_repo for op in ops]
    # No core: only the augmentation block (incl. migration ``add``
    # and ``delete`` op for the augmentation manifests) + README.
    assert remotes == [
        "wikipedia/documents/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "manifests/augmentation_manifest.json",
        "augmentation/manifests/augmentation_manifest.json",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
        "assets/dataset_hero.png",
        "stats.json",
        "README.md",
    ]


def test_assemble_region_upload_without_core_logs_world_land_fallback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Augmentation-only unified sync survives a missing land overlay."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)

    def boom(_cache: Path) -> Path:
        raise RuntimeError("land unavailable")

    monkeypatch.setattr("osm_polygon_wikidata_only.hf.publication.ensure_world_land", boom)
    caplog.set_level(logging.WARNING)

    ops = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=None,
        world_land_warning=None,
    )

    assert ops
    assert any("combined text map will omit continents" in r.getMessage() for r in caplog.records)


def test_assemble_region_upload_with_core_prepends_eight_core_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core artifacts must be the FIRST entries of the unified upload list."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)

    ops = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    remotes = [op.path_in_repo for op in ops]

    assert remotes[0] == "polygons/monaco-latest.parquet"
    assert remotes[1] == "polygon_articles/monaco-latest.parquet"
    assert remotes[2] == "manifests/processed_pbfs.json"
    assert remotes[3] == "assets/geographic_text_presence.png"
    assert remotes[4] == "assets/geographic_text_density.png"
    assert remotes[5] == "assets/geographic_wikipedia_text_coverage.png"
    assert remotes[6] == "assets/geographic_polygon_count.png"
    assert remotes[7] == "assets/coverage_map.png"
    assert remotes[8] == "coverage_map.png"
    assert remotes[9:] == [
        "wikipedia/documents/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "manifests/augmentation_manifest.json",
        "augmentation/manifests/augmentation_manifest.json",
        "assets/dataset_hero.png",
        "stats.json",
        "README.md",
    ]
    assert len(ops) == 20


def test_region_upload_skips_coverage_rendering_when_map_inputs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda *_args, **_kwargs: pytest.fail("coverage rendering must be skipped"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    ops = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
        refresh_maps=False,
    )

    remotes = [op.path_in_repo for op in ops]
    assert remotes[:3] == [
        "polygons/monaco-latest.parquet",
        "polygon_articles/monaco-latest.parquet",
        "manifests/processed_pbfs.json",
    ]
    assert not any(path.startswith("assets/") for path in remotes)


def test_region_upload_can_defer_all_dataset_metadata_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-region sync must build expensive metadata only once at the end."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_polygon_stats_snapshot",
        lambda *_args, **_kwargs: pytest.fail("deferred region upload must not scan polygons"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *_args, **_kwargs: pytest.fail("deferred region upload must not render README"),
    )

    ops = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
        defer_metadata_assets=True,
    )

    remotes = [op.path_in_repo for op in ops]
    assert not any(path == "stats.json" or path == "README.md" for path in remotes)
    assert not any(path.startswith("assets/") for path in remotes)
    assert "polygons/monaco-latest.parquet" in remotes
    assert "wikipedia/documents/monaco-latest.parquet" in remotes


def test_assemble_region_upload_writes_readme_after_other_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README must be written AFTER every other snapshot."""
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
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

    def density_stub(*a: object, **kw: object) -> Path:
        call_order.append("density")
        return a[1].touch() or a[1]  # type: ignore[index]

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        density_stub,
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
    assert "density" in call_order
    assert "coverage" in call_order
    assert call_order.index("density") < call_order.index("README.md")
    assert call_order.index("coverage") < call_order.index("README.md")


def test_assemble_core_upload_returns_twelve_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy core publication migrates from two old H3 maps to one."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    ops = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    remotes = [op.path_in_repo for op in ops]
    assert remotes == [
        "polygons/monaco-latest.parquet",
        "wikipedia/documents/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "polygon_articles/monaco-latest.parquet",
        "manifests/processed_pbfs.json",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
        "assets/dataset_hero.png",
        "stats.json",
        "README.md",
        "assets/coverage_map.png",
        "coverage_map.png",
    ]
    assert len(ops) == 14


def test_assemble_core_upload_writes_readme_after_other_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        lambda *a, **kw: call_order.append("density") or a[1].touch() or a[1],
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
    assert call_order.index("density") < call_order.index("README.md")
    assert call_order.index("coverage") < call_order.index("README.md")
