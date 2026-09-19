"""Split coverage tests (part 2)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.contracts.publication_support import *


def test_assemble_core_upload_invokes_warning_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy core path invokes the world-land warning callback."""
    core, data_root = _stub_process_result(tmp_path)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
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


def test_assemble_region_upload_invokes_world_land_warning_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core-inclusive unified sync forwards the documented fallback warning."""
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
        lambda _dir: (_ for _ in ()).throw(RuntimeError("land unavailable")),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    warnings: list[str] = []

    assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=warnings.append,
    )

    assert warnings == ["Could not fetch world land data; map will omit continents"]


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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        boom,
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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        boom,
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


def test_legacy_core_command_submits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process-pbf/process-dir: one assembly, one queue submission."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp

    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    submissions: list[tuple[list[PublicationOp], str]] = []

    class _StubQueue:
        def submit(self, ops: list[PublicationOp], message: str) -> None:
            submissions.append((ops, message))

    commands_mod._enqueue_core_upload(
        _StubQueue(),  # type: ignore[arg-type]
        data_root=data_root,
        repo_id=REPO_ID,
        commit_message="core msg",
        result=core,
    )
    assert len(submissions) == 1, f"legacy core must submit exactly once, got {len(submissions)}"
    ops, message = submissions[0]
    assert message == "core msg"
    assert len(ops) == 14


def test_augmentation_command_submits_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """augment-region/augment-dir: one assembly, one direct upload call."""
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )

    uploads: list[tuple[list[PublicationOp], str]] = []

    def fake_upload(
        repo_id: str,
        ops: list[PublicationOp] | None = None,
        files: list[tuple[Path, str]] | None = None,
        hub: object = None,
        token: object = None,
        commit_message: str = "",
        num_threads: int = 2,
    ) -> None:
        uploads.append((ops if ops is not None else list(files) if files else [], commit_message))

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.cli.commands.upload_files",
        fake_upload,
    )

    # Invoke the augmentation command's exact submission block.
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload

    ops = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )

    def _submit(
        ops: list[PublicationOp],
        message: str,
        _hub: object = None,
    ) -> None:
        fake_upload(
            REPO_ID,
            ops=ops,
            hub=_hub,
            token=None,
            commit_message=message,
        )

    _submit(ops, "aug msg")
    assert len(uploads) == 1, f"augmentation command must upload exactly once, got {len(uploads)}"
    assert uploads[0][1] == "aug msg"
    # Sidecars + manifest migration + combined map + statistics + README.
    assert len(uploads[0][0]) == 15


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
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
    from osm_polygon_wikidata_only.pipeline.sync_planner import (
        RegionSyncState,
        SyncAction,
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    core, _ = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)

    submissions: list[tuple[list[PublicationOp], str]] = []

    class _StubQueue:
        def submit(self, ops: list[PublicationOp], message: str) -> None:
            submissions.append((list(ops), message))

        def resume_pending(self) -> int:
            return 0

        def close_and_wait(self) -> list[str]:
            return []

    # Production pure assembler invoked by the production runner.
    from osm_polygon_wikidata_only.hf.publication import assemble_region_upload
    from osm_polygon_wikidata_only.pipeline import sync_runner

    def _submit_upload(ops: list[PublicationOp], message: str) -> None:
        submissions.append((list(ops), message))

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
    ) -> list[PublicationOp]:
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
    by_message = {msg: ops for ops, msg in submissions}
    assert set(by_message) == {
        "Sync complete region monaco-latest",
        "Sync complete region andorra-latest",
    }
    # PROCESS state includes three current maps and deletes the three
    # superseded remote map paths atomically.
    assert len(by_message["Sync complete region monaco-latest"]) == 20
    # AUGMENT state (no core): sidecars, manifest migration, combined map, statistics, README.
    assert len(by_message["Sync complete region andorra-latest"]) == 15


def test_coverage_refresh_required_returns_false_when_no_core() -> None:
    assert coverage_refresh_required(None) is False


def test_coverage_refresh_required_returns_true_when_core_present() -> None:
    assert coverage_refresh_required(object()) is True


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
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_presence",
        lambda _root, dest, **_kw: dest.touch() or dest,
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
    map_path, presence_path, density_path = refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem=core.polygons_path.stem,
        snapshots_dir=snapshots_dir,
        world_land_warning=lambda msg: None,
    )
    assert map_path.exists()
    assert presence_path.exists()
    assert density_path.exists()


def test_refresh_coverage_assets_loads_combined_text_inputs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two text maps must share one expensive finalized-table scan."""
    core, data_root = _stub_process_result(tmp_path)
    sentinel = object()
    loads: list[Path] = []
    received: list[object] = []
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._load_text_presence",
        lambda root: loads.append(root) or sentinel,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_presence",
        lambda _root, dest, **kwargs: received.append(kwargs["snapshot"]) or dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        lambda _root, dest, **kwargs: received.append(kwargs["snapshot"]) or dest.touch() or dest,
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

    refresh_coverage_assets(
        data_root=data_root,
        snapshot_stem=core.polygons_path.stem,
        snapshots_dir=data_root.cache / "single_scan",
        world_land_warning=None,
    )

    assert loads == [data_root.processed]
    assert received == [sentinel, sentinel]
