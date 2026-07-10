"""Tests for the HF Hub subpackage (repo_layout, uploader, dataset_card)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_ARTICLES_DIR,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
    local_to_remote,
    remote_dataset_card_path,
    remote_parquet_path,
)
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import (
    StubHfHub,
    UploadError,
    upload_card,
    upload_files,
    upload_manifest,
    upload_parquet,
)


def test_remote_parquet_path() -> None:
    assert (
        remote_parquet_path(REMOTE_POLYGONS_DIR, "monaco-latest")
        == "polygons/monaco-latest.parquet"
    )
    assert (
        remote_parquet_path(REMOTE_ARTICLES_DIR, "monaco-latest")
        == "articles/monaco-latest.parquet"
    )
    assert (
        remote_parquet_path(REMOTE_LINKS_DIR, "monaco-latest")
        == "polygon_articles/monaco-latest.parquet"
    )


def test_remote_dataset_card_path() -> None:
    assert remote_dataset_card_path() == "README.md"


def test_local_to_remote() -> None:
    p = Path("/x/processed/polygons/monaco-latest.parquet")
    assert local_to_remote(p, "polygons") == "polygons/monaco-latest.parquet"


def test_remote_manifest_file_is_deterministic() -> None:
    assert REMOTE_MANIFEST_FILE == "manifests/processed_pbfs.json"


def _small_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.parquet"
    # Write a minimal placeholder (not real parquet, the stub doesn't care).
    p.write_text("placeholder", encoding="utf-8")
    return p


def test_upload_parquet_records_call(tmp_path: Path) -> None:
    stub = StubHfHub()
    p = _small_parquet(tmp_path)
    remote = upload_parquet("org/name", p, path_in_repo="polygons/x.parquet", hub=stub)
    assert remote == "polygons/x.parquet"
    assert len(stub.uploads) == 1
    up = stub.uploads[0]
    assert up["path_in_repo"] == "polygons/x.parquet"
    assert up["repo_id"] == "org/name"
    assert up["repo_type"] == "dataset"
    assert up["size_bytes"] > 0


def test_upload_parquet_missing_file_raises(tmp_path: Path) -> None:
    stub = StubHfHub()
    with pytest.raises(UploadError):
        upload_parquet(
            "org/name",
            tmp_path / "missing.parquet",
            path_in_repo="polygons/x.parquet",
            hub=stub,
        )


def test_upload_parquet_custom_commit_message(tmp_path: Path) -> None:
    stub = StubHfHub()
    p = _small_parquet(tmp_path)
    upload_parquet(
        "org/name",
        p,
        path_in_repo="polygons/x.parquet",
        hub=stub,
        commit_message="manual update",
    )
    assert stub.uploads[0]["commit_message"] == "manual update"


def test_upload_manifest(tmp_path: Path) -> None:
    stub = StubHfHub()
    p = tmp_path / "manifest.json"
    p.write_text("{}", encoding="utf-8")
    upload_manifest(
        "org/name",
        p,
        path_in_repo=REMOTE_MANIFEST_FILE,
        hub=stub,
    )
    assert stub.uploads[0]["path_in_repo"] == REMOTE_MANIFEST_FILE


def test_upload_files_commits_every_artifact_atomically(tmp_path: Path) -> None:
    stub = StubHfHub()
    polygon = _small_parquet(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    upload_files(
        "org/name",
        [(polygon, "polygons/x.parquet"), (manifest, REMOTE_MANIFEST_FILE)],
        hub=stub,
        commit_message="Update PBF x",
        num_threads=3,
    )

    assert len(stub.commits) == 1
    assert stub.commits[0]["paths"] == ["polygons/x.parquet", REMOTE_MANIFEST_FILE]
    assert stub.commits[0]["num_threads"] == 3


def test_background_upload_submit_does_not_wait_for_upload(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def upload(files: list[tuple[Path, str]], message: str) -> None:
        started.set()
        release.wait(timeout=2)

    queue = BackgroundUploadQueue(upload=upload, max_pending=2)
    queue.submit([(_small_parquet(tmp_path), "polygons/x.parquet")], "x")
    assert started.wait(timeout=1)
    release.set()
    assert queue.close_and_wait() == []


def test_background_upload_reports_worker_failure(tmp_path: Path) -> None:
    def upload(files: list[tuple[Path, str]], message: str) -> None:
        raise UploadError("offline")

    queue = BackgroundUploadQueue(upload=upload, max_pending=2)
    queue.submit([(_small_parquet(tmp_path), "polygons/x.parquet")], "x")
    failures = queue.close_and_wait()
    assert len(failures) == 1
    assert "offline" in failures[0]


def test_background_upload_failure_is_resumed_from_durable_state(tmp_path: Path) -> None:
    state = tmp_path / "jobs"
    artifact = _small_parquet(tmp_path)
    failing = BackgroundUploadQueue(
        upload=lambda files, message: (_ for _ in ()).throw(UploadError("offline")),
        state_dir=state,
    )
    failing.submit([(artifact, "polygons/x.parquet")], "x")
    assert failing.close_and_wait()
    assert len(list(state.glob("*.json"))) == 1

    calls: list[str] = []
    resumed = BackgroundUploadQueue(
        upload=lambda files, message: calls.append(message),
        state_dir=state,
    )
    assert resumed.resume_pending() == 1
    assert resumed.close_and_wait() == []
    assert calls == ["x"]
    assert not list(state.glob("*.json"))


def test_upload_card_rejects_empty() -> None:
    stub = StubHfHub()
    with pytest.raises(UploadError):
        upload_card("org/name", "", hub=stub)


def test_upload_card_records_markdown() -> None:
    stub = StubHfHub()
    remote = upload_card("org/name", "# card", hub=stub, commit_message="add card")
    assert remote == "README.md"
    assert stub.uploads[0]["size_bytes"] == len(b"# card")


def test_render_dataset_card_includes_schema() -> None:
    markdown = render_dataset_card(
        repo_id="org/name",
        stats={"polygon_count": 1, "article_count": 2, "unique_wikidata_count": 1},
        polygon_columns=["polygon_id", "name", "area_m2"],
        polygon_descriptions={
            "polygon_id": "Stable per-PBF polygon identifier.",
            "name": "OSM name tag (may be empty).",
            "area_m2": "Polygon area, square meters.",
        },
        article_columns=["article_id", "language", "full_text"],
        article_descriptions={
            "article_id": "Stable article identifier.",
            "language": "ISO 639-1 code.",
            "full_text": "Plain-text body.",
        },
        link_columns=["polygon_id", "article_id"],
        link_descriptions={
            "polygon_id": "Polygon row this article links to.",
            "article_id": "Article row this polygon links to.",
        },
    )
    assert markdown.startswith("---\n")
    assert "license: odbl" in markdown
    assert "polygons/*.parquet" in markdown
    assert "`polygons`" in markdown
    assert "`articles`" in markdown
    assert "`polygon_articles`" in markdown
    assert "Stable per-PBF polygon identifier." in markdown


def test_render_dataset_card_mentions_licenses() -> None:
    markdown = render_dataset_card(
        repo_id="org/name",
        stats={},
        polygon_columns=[],
        polygon_descriptions={},
        article_columns=[],
        article_descriptions={},
        link_columns=[],
        link_descriptions={},
    )
    assert "ODbL" in markdown
    assert "CC BY-SA" in markdown
    assert "Wikipedia" in markdown


def test_render_dataset_card_identifies_multilingual_scope_and_maintainer() -> None:
    markdown = render_dataset_card(
        repo_id="NoeFlandre/osm-polygon-wikidata-only",
        stats={},
        polygon_columns=[],
        polygon_descriptions={},
        article_columns=[],
        article_descriptions={},
        link_columns=[],
        link_descriptions={},
    )
    assert "Noé Flandre" in markdown
    assert "every valid language-Wikipedia sitelink" in markdown
    assert "no per-QID article cap" in markdown
    assert "  - multilingual" in markdown
