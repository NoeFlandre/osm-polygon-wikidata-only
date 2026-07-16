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
        [{"polygon_id": "1", "lat": 1.0, "lon": 2.0}], schema=polygon_schema()
    )
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    pq.write_table(poly_table, polygons_path)  # type: ignore[no-untyped-call]

    if not invalid_core:
        links_table = pa.Table.from_pylist(
            [{"polygon_id": "1", "article_id": "Q1:es:1234:5678"}], schema=polygon_article_schema()
        )
        pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {}
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
        (data_root.processed_manifests / "processed_pbfs.json").write_text(
            json.dumps(manifest_data)
        )

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
        aug_manifest_path.write_text(json.dumps(aug_manifest))


@pytest.fixture
def mock_hf_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "fake-token")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")


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

    # Remote missing everything
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

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
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote missing core, trigger publish/repair
    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

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

    caplog.clear()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")

    rc = commands.main(args)
    # Failure should return non-zero
    assert rc != 0

    # No success log should be printed
    assert not any(
        "Remote reconciliation complete" in record.getMessage() for record in caplog.records
    )


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
        [{"polygon_id": "1", "article_id": "Q1:es:1234:5678"}], schema=polygon_article_schema()
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

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
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

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

    caplog.clear()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")

    rc = commands.main(args)
    assert rc == 0

    assert any(
        "Remote reconciliation complete: 1 regions repaired; README and maps refreshed"
        in record.getMessage()
        for record in caplog.records
    )


def test_logging_sidecar_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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

    caplog.clear()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")

    rc = commands.main(args)
    assert rc == 0

    # Assert that maps are NOT reported as refreshed
    assert any(
        "Remote reconciliation complete: 1 regions repaired" in record.getMessage()
        for record in caplog.records
    )
    assert not any("README and maps refreshed" in record.getMessage() for record in caplog.records)


def test_logging_metadata_only_repair(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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

    caplog.clear()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")

    rc = commands.main(args)
    assert rc == 0

    assert any(
        "Remote reconciliation complete: README and maps refreshed" in record.getMessage()
        for record in caplog.records
    )
    assert not any("regions repaired" in record.getMessage() for record in caplog.records)


def test_logging_upload_failure(
    tmp_path: Path,
    mock_hf_auth: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    stub = StubHfHub(remote_files=set())
    setup_test_hub(monkeypatch, stub)

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

    caplog.clear()
    caplog.set_level("INFO", logger="osm_polygon_wikidata_only.cli")

    rc = commands.main(args)
    assert rc != 0

    # Verify aborted logging exists, and success logging is absent
    assert any(
        "Unified sync aborted:" in record.getMessage()
        or "Unified sync completed with failures" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "Remote reconciliation complete" in record.getMessage() for record in caplog.records
    )
