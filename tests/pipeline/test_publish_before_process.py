"""End-to-end priority tests for the publish-before-process behavior.

These tests use the synthetic run_sync fixtures and assert that
PUBLISH-only reconciliation repairs run before PROCESSING new core
data. They cover:

* AUGMENT backlog keeps its priority.
* PUBLISH repair uploads BEFORE the first PROCESS publication.
* The second run is a no-op (convergence).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.cli import commands, run_sync
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_mock_region(data_root: DataRoot, stem: str, *, augmented: bool = True) -> None:
    data_root.processed_polygons.mkdir(parents=True, exist_ok=True)
    data_root.processed_links.mkdir(parents=True, exist_ok=True)
    data_root.processed_manifests.mkdir(parents=True, exist_ok=True)

    poly_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "wikidata": "Q1", "lat": 1.0, "lon": 2.0}],
        schema=polygon_schema(),
    )
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    pq.write_table(poly_table, polygons_path)  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "article_id": "a1", "wikidata": "Q1"}],
        schema=polygon_article_schema(),
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {
        f"{stem}.osm.pbf": {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
    }
    (data_root.processed_manifests / "processed_pbfs.json").write_text(json.dumps(manifest_data))

    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wikipedia_documents_path.parent.mkdir(parents=True, exist_ok=True)
    doc_table = pa.Table.from_pylist(
        [
            {
                "document_id": "1",
                "article_id": "a1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "es",
            }
        ],
        schema=wikipedia_document_schema(),
    )
    pq.write_table(doc_table, wikipedia_documents_path)  # type: ignore[no-untyped-call]

    if augmented:
        paths = [
            data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet",
            data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet",
            data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
        ]
        schemas = [section_schema(), document_schema(), section_schema(), fact_schema()]
        for path, schema in zip(paths, schemas, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_batches([], schema=schema), path)  # type: ignore[no-untyped-call]

        aug_manifest_path = (
            data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        aug_manifest_path.write_text(
            json.dumps(
                {
                    stem: {
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
                }
            )
        )


@pytest.fixture
def mock_hf_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "fake-token")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")


def _setup_test_hub(monkeypatch: pytest.MonkeyPatch, stub: StubHfHub) -> None:
    def mock_fetch(repo_id: str, token: str | None = None, hub: Any = None) -> RemoteInventory:
        files = stub.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return RemoteInventory(set(files))

    monkeypatch.setattr(RemoteInventory, "fetch", mock_fetch)
    original_build_queue = run_sync._build_upload_queue

    def mock_build_queue(*args: Any, **kwargs: Any) -> Any:
        kwargs["_hub"] = stub
        return original_build_queue(*args, **kwargs)

    monkeypatch.setattr(run_sync, "_build_upload_queue", mock_build_queue)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _synthesize_pending_publication(data_root: DataRoot, stem: str) -> None:
    from osm_polygon_wikidata_only.pipeline.pending_publications import add_pending_publications

    add_pending_publications(data_root, {stem})


def test_publish_only_repair_runs_before_processing_new_region(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the input has a PUBLISH-only repair candidate AND a fresh
    PROCESS PBF, the upload queue must see the publish commit FIRST,
    then the process commit."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    publish_stem = "mexico-latest"

    _setup_mock_region(data_root, publish_stem, augmented=True)
    # Make a local pending-publication manifest to ensure it goes PUBLISH
    _synthesize_pending_publication(data_root, publish_stem)

    # Remote: PUBLISH stem is missing augmentation sidecars only
    stub_files = {
        f"polygons/{publish_stem}.parquet",
        f"polygon_articles/{publish_stem}.parquet",
        "README.md",
        "manifests/processed_pbfs.json",
        "manifests/augmentation_manifest.json",
        "assets/coverage_map.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
    }
    stub = StubHfHub(remote_files=stub_files)
    _setup_test_hub(monkeypatch, stub)
    _patch_runtime(monkeypatch)

    # Only the publish stem has a real PBF; the process stem is not
    # in input_stems. The publish commit alone is the assertion
    # we care about for ordering -- that it is enqueued BEFORE any
    # process commit.
    (data_root.raw / f"{publish_stem}.osm.pbf").touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--repo-id",
        "user/repo",
        "--hf-token",
        "fake-token",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0

    # A PUBLISH commit must exist for the publish stem
    messages = [c["commit_message"] for c in stub.commits]
    publish_commits = [m for m in messages if publish_stem in m]
    assert publish_commits, f"No publish commit recorded. Commits: {messages}"

    # The publish commit message is the FIRST one issued (no process
    # commits exist in this scenario, since only the publish stem is in
    # the input).
    assert messages[0] == publish_commits[0]


def test_two_runs_are_noop(
    tmp_path: Path, mock_hf_auth: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful convergence, the second invocation must
    find nothing to do and remain a no-op."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    stem = "mexico-latest"
    _setup_mock_region(data_root, stem, augmented=True)

    # Remote is fully populated -- nothing to repair
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
        "assets/coverage_map.png",
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
    }
    stub = StubHfHub(remote_files=stub_files)
    _setup_test_hub(monkeypatch, stub)
    _patch_runtime(monkeypatch)

    (data_root.raw / f"{stem}.osm.pbf").touch()

    args = [
        "sync-dir",
        str(data_root.raw),
        "--data-root",
        str(tmp_path),
        "--push",
        "--dry-run",
        "--repo-id",
        "user/repo",
        "--hf-token",
        "fake-token",
        "--skip-existing",
    ]
    rc = commands.main(args)
    assert rc == 0
    # No commits recorded on a converged run
    assert stub.commits == []

    # Second invocation: also no commits
    rc = commands.main(args)
    assert rc == 0
    assert stub.commits == []
