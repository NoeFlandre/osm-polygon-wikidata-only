from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.cli import commands, run_sync
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.schema import (
    article_schema,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline.sync_planner import SyncAction, plan_sync_states


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_mock_region(
    data_root: DataRoot, stem: str, augmented: bool = True, invalid_core: bool = False
) -> None:
    data_root.processed_polygons.mkdir(parents=True, exist_ok=True)
    data_root.processed_links.mkdir(parents=True, exist_ok=True)
    data_root.processed_manifests.mkdir(parents=True, exist_ok=True)

    poly_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "wikidata": "Q1", "lat": 1.0, "lon": 2.0}],
        schema=polygon_schema(),
    )
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    pq.write_table(poly_table, polygons_path)  # type: ignore[no-untyped-call]

    if not invalid_core:
        links_table = pa.Table.from_pylist(
            [
                {
                    "polygon_id": "1",
                    "article_id": "Q1:es:1234:5678",
                    "wikidata": "Q1",
                }
            ],
            schema=polygon_article_schema(),
        )
        pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {}
    processed_pbfs_path = data_root.processed_manifests / "processed_pbfs.json"
    if processed_pbfs_path.exists():
        manifest_data = json.loads(processed_pbfs_path.read_text(encoding="utf-8"))
    if not invalid_core:
        manifest_data[f"{stem}.osm.pbf"] = {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
        processed_pbfs_path.write_text(json.dumps(manifest_data, indent=2))

    # wikipedia documents
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wikipedia_documents_path.parent.mkdir(parents=True, exist_ok=True)
    doc_table = pa.Table.from_pylist(
        [
            {
                "document_id": "Q1:wikipedia:es:1234:5678",
                "article_id": "Q1:es:1234:5678",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "es",
            }
        ],
        schema=wikipedia_document_schema(),
    )
    pq.write_table(doc_table, wikipedia_documents_path)  # type: ignore[no-untyped-call]

    if augmented:
        wikipedia_sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
        wikivoyage_documents_path = (
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
        )
        wikivoyage_sections_path = (
            data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
        )
        wikidata_facts_path = data_root.processed / "wikidata" / "facts" / f"{stem}.parquet"

        from osm_polygon_wikidata_only.augmentation.schema import (
            document_schema,
            fact_schema,
            section_schema,
        )

        for p, schema in [
            (wikipedia_sections_path, section_schema()),
            (wikivoyage_documents_path, document_schema()),
            (wikivoyage_sections_path, section_schema()),
            (wikidata_facts_path, fact_schema()),
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_batches([], schema=schema), p)  # type: ignore[no-untyped-call]

        aug_manifest = {}
        aug_manifest[stem] = {
            "contract_version": "text-sidecars-v1",
            "core_hashes": {
                str(polygons_path): compute_sha256(polygons_path),
                str(wikipedia_documents_path): compute_sha256(wikipedia_documents_path),
            },
            "counts": {
                "wikipedia_documents": 1,
                "wikipedia_sections": 0,
                "wikivoyage_documents": 0,
                "wikivoyage_sections": 0,
                "wikidata_facts": 0,
            },
        }
        aug_manifest_path = (
            data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge into any existing augmentation manifest so that setting
        # up multiple regions (e.g. mexico-latest + hungary-latest) does
        # not clobber earlier entries. A clobbered entry would drop the
        # stem from the finalized set and wrongly classify it as AUGMENT,
        # which then triggers a real Wikidata/Wikipedia fetch.
        if aug_manifest_path.exists():
            existing = json.loads(aug_manifest_path.read_text(encoding="utf-8"))
            existing.update(aug_manifest)
            aug_manifest = existing
        aug_manifest_path.write_text(json.dumps(aug_manifest, indent=2))


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


@pytest.fixture
def mock_hf_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "fake-token")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")


class _LoggerSpy:
    """In-process recorder for ``LOGGER`` info/error calls.

    Each emission is stored with its interpolated message string.
    Tests can assert on the recorded list without depending on
    pytest's caplog state, which can be reset by earlier tests
    that call ``configure_logging``. ``error`` is intercepted in
    addition to ``info`` so failure-path messages (e.g.
    ``"Unified sync completed with failures"``) are visible to
    tests, while ``warning``, ``debug``, and ``critical`` are
    stubbed defensively to keep the spy stable across any future
    changes to the cli module.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.messages.append(str(message))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.messages.append(str(message))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _install_logger_spy(monkeypatch: pytest.MonkeyPatch) -> _LoggerSpy:
    """Replace ``cli.LOGGER`` methods with a deterministic spy.

    ``info`` and ``error`` are swapped for the spy's recording
    implementations; ``warning``/``debug``/``critical`` are
    stubbed so any silent fallback paths don't surface as
    real logger calls during a focused test. Restoration is
    automatic when the ``monkeypatch`` fixture is torn down.
    """
    spy = _LoggerSpy()
    monkeypatch.setattr(run_sync.LOGGER, "info", spy.info)
    monkeypatch.setattr(run_sync.LOGGER, "error", spy.error)
    monkeypatch.setattr(run_sync.LOGGER, "warning", spy.warning)
    monkeypatch.setattr(run_sync.LOGGER, "debug", spy.debug)
    monkeypatch.setattr(run_sync.LOGGER, "critical", spy.critical)
    return spy


class _NetworkBoundaryRecorder:
    """Tracks invocations of the stubbed ``ensure_world_land``.

    Tests assert on ``calls`` to confirm that the production
    code reached the stubbed boundary (and therefore would have
    hit a real download in production) and on ``urlretrieve_calls``
    to confirm that ``urllib.request.urlretrieve`` -- the actual
    network primitive -- was never invoked.
    """

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.urlretrieve_calls: list[tuple[Any, ...]] = []


def _block_network(monkeypatch: pytest.MonkeyPatch) -> _NetworkBoundaryRecorder:
    """Replace the real network boundary with a deterministic stub.

    The publication module imports ``ensure_world_land`` directly
    from :mod:`hf.coverage_map` (``from .coverage_map import
    ensure_world_land``), so the binding actually used at runtime
    lives on :mod:`hf.publication`. Patching only
    ``hf.coverage_map.ensure_world_land`` leaves the publication
    module holding the original function and the boundary is
    silently bypassed. This helper therefore patches both the
    canonical definition AND the symbol already imported by
    :mod:`hf.publication`, ensuring any call from the publication
    code path is intercepted.

    ``urllib.request.urlretrieve`` is patched to raise as a
    secondary guard: any future caller that re-introduces a direct
    download becomes a deterministic test failure rather than a
    silent HTTP request.
    """
    recorder = _NetworkBoundaryRecorder()

    def _fake_ensure_world_land(cache_dir: Path) -> Path:
        recorder.calls.append(Path(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "world_land.geojson"
        target.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return target

    # Canonical definition: any future caller that imports
    # ``ensure_world_land`` from :mod:`hf.coverage_map` is covered.
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.coverage_map.ensure_world_land",
        _fake_ensure_world_land,
    )
    # Publication binding: this is the symbol actually called at
    # runtime because :mod:`hf.publication` does
    # ``from .coverage_map import ensure_world_land`` at import
    # time, binding the function into its own namespace.
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        _fake_ensure_world_land,
    )

    def _no_urlretrieve(*args: Any, **kwargs: Any) -> Any:
        recorder.urlretrieve_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: urllib.request.urlretrieve; "
            "tests must mock network boundaries."
        )

    monkeypatch.setattr("urllib.request.urlretrieve", _no_urlretrieve, raising=False)
    return recorder


class _ReconciliationNetworkRecorder:
    """Records any guarded network-boundary invocation.

    Tests assert ``wikimedia_calls == []`` and ``urlretrieve_calls == []``
    to prove the production code reached neither the augmentation
    transport nor the coverage-map download.
    """

    def __init__(self) -> None:
        self.wikimedia_calls: list[tuple[Any, ...]] = []
        self.urlretrieve_calls: list[tuple[Any, ...]] = []


def _block_reconciliation_network(
    monkeypatch: pytest.MonkeyPatch,
) -> _ReconciliationNetworkRecorder:
    """Fail loudly on the real network boundaries used by the sync flow.

    The augmentation client issues HTTPS fetches through
    :func:`augmentation.mediawiki.read_wikimedia_json` (re-exported
    from :mod:`enrichment.wikimedia`). A test that wrongly classifies
    a finalized region as AUGMENT would call it. The publication
    coverage-map rendering calls ``urllib.request.urlretrieve`` to
    download the Natural Earth land GeoJSON.

    Patching the actual augmentation boundary (not the low-level
    ``urllib.request.urlopen`` it happens to sit on) makes the guard
    authoritative: the symbol the production code imports is the one
    intercepted, so a future refactor that swaps the transport cannot
    silently bypass the guard. Both primitives are replaced with
    fail-loud stubs and their invocations are recorded.

    This guard is defense-in-depth only: callers MUST still correct
    their action classification so the network code is never reached
    in the first place.
    """
    recorder = _ReconciliationNetworkRecorder()

    import osm_polygon_wikidata_only.augmentation.mediawiki as mediawiki_mod

    def _no_read_wikimedia_json(*args: Any, **kwargs: Any) -> Any:
        recorder.wikimedia_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: augmentation.mediawiki.read_wikimedia_json; "
            "reconciliation tests must classify finalized regions as "
            "COMPLETE/PUBLISH, not AUGMENT."
        )

    monkeypatch.setattr(mediawiki_mod, "read_wikimedia_json", _no_read_wikimedia_json)

    # Stub the coverage-map download so the publication path never
    # reaches ``urllib.request.urlretrieve``. The publication module
    # imports ``ensure_world_land`` directly from ``hf.coverage_map``,
    # so both the canonical definition and the symbol already bound
    # into ``hf.publication`` are replaced. This is the deterministic
    # substitute that satisfies the "urlretrieve calls are zero"
    # assertion; the ``urlretrieve`` guard below remains as a
    # fail-loud backstop if a future caller re-introduces a direct
    # download.
    def _fake_ensure_world_land(cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "world_land.geojson"
        target.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return target

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.coverage_map.ensure_world_land",
        _fake_ensure_world_land,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        _fake_ensure_world_land,
    )

    def _no_urlretrieve(*args: Any, **kwargs: Any) -> Any:
        recorder.urlretrieve_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: urllib.request.urlretrieve; "
            "tests must stub the coverage-map network boundary."
        )

    monkeypatch.setattr("urllib.request.urlretrieve", _no_urlretrieve, raising=False)
    return recorder


def _refresh_augmentation_manifest(data_root: DataRoot, stem: str) -> None:
    """Recompute and persist the canonical core-hash entry for *stem*.

    Used after a test overwrites the canonical Wikipedia-document (or
    legacy articles) file so the stored manifest hashes match the
    on-disk bytes. Without this refresh ``augmentation_is_current``
    returns False and the region is mis-classified as AUGMENT, which
    would then issue a real Wikidata/Wikipedia fetch.
    """
    from osm_polygon_wikidata_only.augmentation.steps import sha256_file

    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    core_hashes = {
        str(polygons_path): sha256_file(polygons_path),
        str(wikipedia_documents_path): sha256_file(wikipedia_documents_path),
    }
    aug_manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(aug_manifest_path.read_text(encoding="utf-8"))
        if aug_manifest_path.exists()
        else {}
    )
    entry = manifest.get(stem, {})
    entry["contract_version"] = "text-sidecars-v1"
    entry["core_hashes"] = core_hashes
    manifest[stem] = entry
    aug_manifest_path.write_text(json.dumps(manifest, indent=2))


def setup_test_hub(monkeypatch: pytest.MonkeyPatch, stub: StubHfHub) -> None:
    # 1. Mock RemoteInventory.fetch to return files from the stub
    from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory

    def mock_fetch(repo_id: str, token: str | None = None, hub: Any = None) -> RemoteInventory:
        files = stub.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return RemoteInventory(set(files))

    monkeypatch.setattr(RemoteInventory, "fetch", mock_fetch)

    # 2. Inject the stub HfHub into the background upload queue construction
    original_build_queue = run_sync._build_upload_queue

    def mock_build_queue(*args: Any, **kwargs: Any) -> Any:
        kwargs["_hub"] = stub
        return original_build_queue(*args, **kwargs)

    monkeypatch.setattr(run_sync, "_build_upload_queue", mock_build_queue)


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


def test_upload_failure_remains_retryable(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote missing core, trigger publish/repair
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    # Force upload_files failure
    def failing_upload(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("HF upload failed")

    monkeypatch.setattr(run_sync, "upload_files", failing_upload)

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
    # Failure should return non-zero
    assert rc != 0

    # No success log should be printed
    assert not any("Remote reconciliation complete" in message for message in spy.messages)


def test_dry_run_causes_no_mutation_or_retirement(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Place a pending publication stem to simulate retired state
    from osm_polygon_wikidata_only.pipeline.pending_publications import (
        add_pending_publications,
        load_pending_publications,
    )

    add_pending_publications(data_root, {stem})

    # Dry-run execution
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
    rc = commands.main(args)
    assert rc == 0

    # Local retirement is untouched (still pending)
    assert stem in load_pending_publications(data_root)


def test_existing_paired_legacy_retirement_remains_intact(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Overwrite processed_articles AND wikipedia/documents with matching, valid data!
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
        build_wikipedia_document_table,
    )

    row_data = {
        "article_id": "Q1:es:1234:5678",
        "wikidata": "Q1",
        "language": "es",
        "site": "eswiki",
        "title": "Test Title",
        "url": "https://es.wikipedia.org/wiki/Test",
        "page_id": 1234,
        "revision_id": 5678,
        "revision_timestamp": "2026-07-16T12:00:00Z",
        "retrieved_at": "2026-07-16T12:00:00Z",
        "wikidata_label": "Label",
        "wikidata_description": "Description",
        "wikidata_aliases": "",
        "lead_text": "Lead",
        "extract": "Extract",
        "full_text": "Full text",
        "full_text_format": "text",
        "article_length_chars": 9,
        "article_length_words": 2,
        "article_length_tokens_estimate": 2,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": "",
        "license": "CC-BY-SA",
        "attribution": "Attribution",
        "source_api": "rest",
        "fetch_status": "success",
        "fetch_error": "",
        "content_hash": "hash123",
    }

    # Write to local processed_articles
    articles_path = data_root.processed_articles / f"{stem}.parquet"
    data_root.processed_articles.mkdir(parents=True, exist_ok=True)
    art_table = pa.Table.from_pylist([row_data], schema=article_schema())
    pq.write_table(art_table, articles_path)  # type: ignore[no-untyped-call]

    # Convert to canonical document and write to wikipedia/documents
    doc_table = build_wikipedia_document_table(art_table)
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    pq.write_table(doc_table, wikipedia_documents_path)  # type: ignore[no-untyped-call]

    # Write links table with matching article_id so assert_references_resolve passes
    links_table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "1",
                "article_id": "Q1:es:1234:5678",
                "wikidata": "Q1",
            }
        ],
        schema=polygon_article_schema(),
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    # The overwrite invalidated the stored augmentation-manifest hashes
    # (they still point at the pre-overwrite bytes). Refresh the manifest
    # so the region classifies as finalized (COMPLETE/PUBLISH) rather
    # than AUGMENT, which would otherwise trigger a real Wiki fetch.
    _refresh_augmentation_manifest(data_root, stem)

    # Reconciliation tests must never reach the network. The fail-loud
    # guard turns any accidental AUGMENT classification into a test
    # failure instead of a silent es.wikipedia.org request.
    recorder = _block_reconciliation_network(monkeypatch)

    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )

    assert augmentation_is_current(data_root, stem), (
        "mexico-latest was not finalized after overwriting its data files; "
        "the augmentation manifest hashes were left stale."
    )

    # Stub remote
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
        # NO dry-run so retirement executes
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # Verify legacy articles file is retired (deleted locally)
    assert not (data_root.processed_articles / f"{stem}.parquet").exists()

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


def test_reconciliation_network_guard_intercepts_augmentation_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reconciliation network guard must intercept the actual
    augmentation transport symbol -- ``augmentation.mediawiki.
    read_wikimedia_json`` -- not some unrelated low-level primitive.

    This prevents a future refactor from swapping the transport
    (e.g. from ``urllib.request.urlopen`` to ``requests``) and
    silently bypassing a guard that patched the wrong layer. We
    invoke the production symbol directly and assert the guard both
    recorded the call and raised, proving the boundary is the one
    the code actually imports.
    """
    import osm_polygon_wikidata_only.augmentation.mediawiki as mediawiki_mod

    recorder = _block_reconciliation_network(monkeypatch)

    # The production code imports ``read_wikimedia_json`` from
    # ``enrichment.wikimedia`` into the ``augmentation.mediawiki``
    # namespace. The guard must replace that exact binding so the
    # call the code makes is the one intercepted (a guard that
    # patched a different layer would leave this untouched).
    from osm_polygon_wikidata_only.enrichment.wikimedia import (
        read_wikimedia_json as upstream_read_wikimedia_json,
    )

    assert mediawiki_mod.read_wikimedia_json is not upstream_read_wikimedia_json

    with pytest.raises(AssertionError, match="read_wikimedia_json"):
        mediawiki_mod.read_wikimedia_json("https://es.wikipedia.org/w/api.php")

    assert len(recorder.wikimedia_calls) == 1
    assert recorder.urlretrieve_calls == []


def test_augmentation_is_current_called_exactly_once_per_stem(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    import osm_polygon_wikidata_only.augmentation.orchestrator as orch

    call_count = 0
    original_is_current = orch.augmentation_is_current

    def spy_is_current(*args: Any, **kwargs: Any) -> bool:
        nonlocal call_count
        call_count += 1
        return original_is_current(*args, **kwargs)

    monkeypatch.setattr(orch, "augmentation_is_current", spy_is_current)
    monkeypatch.setattr(run_sync, "augmentation_is_current", spy_is_current)

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
    rc = commands.main(args)
    assert rc == 0
    assert call_count == 1  # Verified exactly once!


def test_token_resolver_failure_but_injected_hub_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Force resolve_hf_token to fail
    def failing_resolver(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Token resolver failed")

    monkeypatch.setattr(commands, "resolve_hf_token", failing_resolver)

    stub = StubHfHub(remote_files=set())
    pbf_file = data_root.raw / f"{stem}.osm.pbf"
    pbf_file.touch()

    parser = commands.build_parser()
    args = parser.parse_args(
        [
            "sync-dir",
            str(data_root.raw),
            "--data-root",
            str(tmp_path),
            "--push",
            "--dry-run",
            "--skip-existing",
        ]
    )
    settings = Settings(repo_id="test/repo", hf_token="fake", skip_existing=True)

    # Call run_sync.execute directly with injected collaborators, verifying it works
    # without raising TokenResolver errors.
    remote_inventory = RemoteInventory(
        {
            f"polygons/{stem}.parquet",
            f"polygon_articles/{stem}.parquet",
            f"wikipedia/documents/{stem}.parquet",
            f"wikipedia/sections/{stem}.parquet",
            f"wikivoyage/documents/{stem}.parquet",
            f"wikivoyage/sections/{stem}.parquet",
            f"wikidata/facts/{stem}.parquet",
        }
    )
    rc = run_sync.execute(
        args,
        data_root=data_root,
        settings=settings,
        _remote_inventory=remote_inventory,
        _hub=stub,
    )
    assert rc == 0


def test_logging_core_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

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

    assert any(
        "Remote reconciliation complete: 1 regions repaired; README and maps refreshed" in message
        for message in spy.messages
    )


def test_logging_sidecar_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has core files, manifests, etc. but missing wikipedia documents
    stub_files = {
        f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
        "assets/coverage_map.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
    }
    stub = StubHfHub(remote_files=stub_files)
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

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

    # Assert that maps are NOT reported as refreshed
    assert any(
        "Remote reconciliation complete: 1 regions repaired" in message for message in spy.messages
    )
    assert not any("README and maps refreshed" in message for message in spy.messages)


def test_logging_metadata_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote has everything except README
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
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

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

    assert any(
        "Remote reconciliation complete: README and maps refreshed" in message
        for message in spy.messages
    )
    assert not any("regions repaired" in message for message in spy.messages)


def test_logging_upload_failure(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    _block_network(monkeypatch)
    spy = _install_logger_spy(monkeypatch)

    # Fail the upload queue
    def failing_upload(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Upload failure")

    monkeypatch.setattr(run_sync, "upload_files", failing_upload)

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
    assert rc != 0

    # Verify aborted logging exists, and success logging is absent
    assert any(
        "Unified sync aborted:" in message or "Unified sync completed with failures" in message
        for message in spy.messages
    )
    assert not any("Remote reconciliation complete" in message for message in spy.messages)


# ---------------------------------------------------------------------------
# Direct unit test for the summary helper -- independent of caplog state
# ---------------------------------------------------------------------------


def test_log_remote_reconciliation_summary_unit() -> None:
    """The summary helper accepts an injected log callable.

    Verifies the four documented branches without any caplog
    interaction, so the assertions are deterministic regardless
    of module-level logger configuration.
    """
    spy = _LoggerSpy()

    # Core + metadata refresh with repaired regions
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps={"mexico-latest"},
        core_repaired=True,
        metadata_repaired=False,
        log=spy.info,
    )
    # Maps refreshed but no repaired regions
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps=set(),
        core_repaired=True,
        metadata_repaired=False,
        log=spy.info,
    )
    # Repaired regions but maps NOT refreshed
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps={"andorra-latest", "mexico-latest"},
        core_repaired=False,
        metadata_repaired=False,
        log=spy.info,
    )
    # Converged
    run_sync._log_remote_reconciliation_summary(
        stems_with_gaps=set(),
        core_repaired=False,
        metadata_repaired=False,
        log=spy.info,
    )

    assert spy.messages[0] == (
        "Remote reconciliation complete: 1 regions repaired; README and maps refreshed"
    )
    assert spy.messages[1] == ("Remote reconciliation complete: README and maps refreshed")
    assert spy.messages[2] == "Remote reconciliation complete: 2 regions repaired"
    assert spy.messages[3] == "Remote reconciliation complete: converged"


def test_no_real_network_during_logging_core_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``test_logging_core_repair`` path must make zero real
    network calls.

    ``ensure_world_land`` (the symbol actually called by the
    publication module) is patched to a deterministic stub, and
    ``urllib.request.urlretrieve`` is patched to raise so any
    unstubbed network path becomes a deterministic test failure
    rather than a silent download. The assertions verify both
    sides of that contract:

    * the stubbed boundary was invoked (proving the test
      exercises the production code path that would have made
      a real HTTP request), and
    * ``urlretrieve`` was never invoked (proving the stub
      actually replaced the network primitive instead of relying
      on production code to swallow an ``AssertionError`` raised
      from a different module binding).
    """
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)
    recorder = _block_network(monkeypatch)
    _install_logger_spy(monkeypatch)

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

    # The publication code path reached the stubbed boundary at
    # least once -- this is the call that, in production, would
    # have invoked ``urllib.request.urlretrieve`` to download the
    # Natural Earth land GeoJSON.
    assert recorder.calls, (
        "Stubbed ensure_world_land was never invoked; "
        "publication code path did not exercise the network boundary."
    )
    # The urlretrieve guard was never tripped -- the stub actually
    # replaced the network primitive for the duration of the test.
    assert recorder.urlretrieve_calls == [], (
        "urllib.request.urlretrieve was invoked "
        f"{len(recorder.urlretrieve_calls)} time(s); the stub did not "
        "fully replace the network boundary."
    )
