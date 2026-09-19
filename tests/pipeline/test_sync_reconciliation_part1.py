"""Split coverage tests (part 1)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.pipeline.sync_reconciliation_support import *


def test_recovered_region_publication_loads_repaired_core(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    _setup_mock_region(data_root, "recovered-latest", augmented=True)

    core = run_sync._load_existing_core_for_publication(
        data_root,
        "recovered-latest",
        None,
        required=True,
    )

    assert core is not None
    assert core.polygons_path == data_root.processed_polygons / "recovered-latest.parquet"
    assert core.polygon_articles_path == (data_root.processed_links / "recovered-latest.parquet")


def test_sync_reconciliation_integration_success(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # 1. Setup local completed/augmented region
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote missing core polygons/links but has augmentation sidecars
    stub_files = {
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)

    # Setup dummy raw pbf
    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    # Execute main sync-dir
    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--repo-id",
        "NoeFlandre/osm-polygon-wikidata-only",
        "--hf-token",
        "fake-token",
        "--skip-existing",
    ]

    # Mock runtime with None clients to prove they are never invoked
    def mock_build_wikimedia_runtime(*args: Any, **kwargs: Any) -> Any:
        class DummyRuntime:
            settings = Settings(repo_id="test", user_agent="test")
            scheduler = type("DummyScheduler", (), {"snapshot": {}})()
            session = type("DummySession", (), {"auth_snapshot": {}})()
            wikidata = None
            wikipedia = None
            cache = None

        return DummyRuntime()

    monkeypatch.setattr(run_sync, "build_wikimedia_runtime", mock_build_wikimedia_runtime)

    # First run: should repair the remote region by uploading core parquets, README, manifests, maps.
    rc = commands.main(args)
    assert rc == 0

    # Check that core files are now in the remote_files
    assert stub.remote_files is not None
    assert f"polygons/{stem}.parquet" in stub.remote_files
    assert f"polygon_articles/{stem}.parquet" in stub.remote_files

    # 2. Second sync run: should be a complete no-op (no repair, converged)
    rc_second = commands.main(args)
    assert rc_second == 0


def test_metadata_only_gaps_repaired_and_enqueued_last(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Setup local completed/augmented region
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has everything except README.md and coverage map
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)

    # Setup dummy raw pbf
    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    # Run sync-dir
    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--repo-id",
        "NoeFlandre/osm-polygon-wikidata-only",
        "--hf-token",
        "fake-token",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # Verify README.md is repaired
    assert stub.remote_files is not None
    assert "README.md" in stub.remote_files
    assert "assets/coverage_map.png" in stub.remote_files
    assert "assets/dataset_hero.png" in stub.remote_files

    # Verify enqueued last:
    # The last commit in stub.commits should be the metadata-only repair commit
    assert len(stub.commits) > 0
    last_commit = stub.commits[-1]
    assert last_commit["commit_message"] == "Repair remote repository metadata and maps"


def test_incomplete_local_augmentation_remains_augment(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Setup local completed core, but INCOMPLETE augmentation (augmented=False)
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=False)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    # Run sync planner states directly to verify it classifies as SyncAction.AUGMENT
    all_pending_stems: set[str] = set()
    core_stems = {"mexico-latest"}

    # Since augmentation is incomplete, it should be AUGMENT action
    states = plan_sync_states(
        [pbf_file],
        core_stems=core_stems,
        augmentation_stems=set(),  # incomplete
        pending_stems=all_pending_stems,
    )
    assert states[0].action == SyncAction.AUGMENT


def test_reconciliation_limited_to_input_stems(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Two regions local: mexico-latest and hungary-latest
    _setup_mock_region(data_root, "mexico-latest", augmented=True)
    # hungary-latest is inconsistent/malformed (invalid_core=True)
    _setup_mock_region(data_root, "hungary-latest", augmented=True, invalid_core=True)

    # Reconciliation tests must never reach the network. The fail-loud
    # guard turns any accidental AUGMENT classification (which would
    # trigger a real Wiki fetch) into a test failure.
    recorder = _block_reconciliation_network(monkeypatch)

    # Remote missing everything
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    # The finalized mexico-latest region must classify as COMPLETE, not
    # AUGMENT: both its processed_pbfs entry and its augmentation
    # manifest entry survive after setting up a second region.
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )

    assert augmentation_is_current(data_root, "mexico-latest"), (
        "mexico-latest was not finalized after setting up a second region; "
        "the augmentation manifest entry was likely clobbered."
    )

    # Input PBF contains ONLY mexico-latest.osm.pbf
    pbf_file = data_root.raw / "mexico-latest.osm.pbf"
    pbf_file.touch()

    # Running sync-dir on mexico-latest should succeed and validate only mexico-latest.
    # It should not fail due to hungary-latest being malformed, because hungary-latest is outside input scope!
    args = [
        "sync-dir",
        str(pbf_file),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # The network guard must not have been tripped: no augmentation
    # transport call and no coverage-map download.
    assert recorder.wikimedia_calls == [], (
        f"Augmentation transport was called {len(recorder.wikimedia_calls)} "
        "time(s); a finalized region was mis-classified as AUGMENT."
    )
    assert recorder.urlretrieve_calls == [], (
        f"urllib.request.urlretrieve was called {len(recorder.urlretrieve_calls)} "
        "time(s); the coverage-map download was not stubbed."
    )


def test_remote_extras_reported_never_deleted(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has everything, plus an extra remote file
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
        "polygons/extra-remote.parquet",  # Extra remote file
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    # Verify extra-remote.parquet is NOT deleted
    assert stub.remote_files is not None
    assert "polygons/extra-remote.parquet" in stub.remote_files


def test_remote_inventory_fetched_exactly_once(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    fetch_count = 0
    original_fetch = RemoteInventory.fetch

    def tracked_fetch(*args: Any, **kwargs: Any) -> RemoteInventory:
        nonlocal fetch_count
        fetch_count += 1
        return original_fetch(*args, **kwargs)

    monkeypatch.setattr(RemoteInventory, "fetch", tracked_fetch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    assert fetch_count == 1  # Inventory fetched exactly once


def test_no_remote_inventory_call_without_push(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Setup completed local region
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

    fetch_called = False

    def tracked_fetch(*args: Any, **kwargs: Any) -> RemoteInventory:
        nonlocal fetch_called
        fetch_called = True
        return RemoteInventory(set())

    monkeypatch.setattr(RemoteInventory, "fetch", tracked_fetch)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    # sync-dir WITHOUT --push
    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    assert not fetch_called  # No RemoteInventory.fetch calls occurred


def test_inventory_auth_failure_raises_upload_error(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Mock token/authorization check rejection
    from osm_polygon_wikidata_only.hf.uploader import UploadError

    def mock_verify_fail(token: str | None) -> str:
        raise UploadError("Invalid HF_TOKEN")

    monkeypatch.setattr(commands, "verify_hf_token", mock_verify_fail)

    pbf_file = data_root.raw / "mexico-latest.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--repo-id",
        "NoeFlandre/osm-polygon-wikidata-only",
        "--hf-token",
        "fake-token",
        "--skip-existing",
    ]
    # commands.main handles UploadError and exits with 2
    with pytest.raises(SystemExit) as excinfo:
        commands.main(args)
    assert excinfo.value.code == 2


def test_malformed_local_core_fails_closed(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Setup local stem with inconsistent finalized core (polygons exist, links do not)
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True, invalid_core=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    with pytest.raises(ValueError, match="Inconsistent core state"):
        commands.main(args)


def test_one_of_polygons_or_links_missing_remotely(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote missing only polygon articles links (has polygons, sidecars, etc.)
    stub_files = {
        f"polygons/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    # Both polygons and links must be uploaded in a coherent commit
    assert f"polygons/{stem}.parquet" in stub.remote_files
    assert f"polygon_articles/{stem}.parquet" in stub.remote_files


@pytest.mark.parametrize(
    "missing_corp",
    [
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ],
)
def test_each_of_the_five_augmentation_corpora_missing_independently(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch, missing_corp: str
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = f"mexico-{missing_corp.replace('/', '-')}"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has everything except the missing_corp sidecar
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
    }
    corpora = [
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ]
    for corp in corpora:
        if corp != missing_corp:
            stub_files.add(f"{corp}/{stem}.parquet")

    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)

    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    args = [
        "sync-dir",
        str(pbf_file),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    assert f"{missing_corp}/{stem}.parquet" in stub.remote_files
