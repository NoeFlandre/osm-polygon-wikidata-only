from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_wikidata_only.cli import sync_publication
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf import core_publication
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import upload_files as real_upload_files


def _publisher_queue(
    publisher: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    data_root: DataRoot,
    settings: Settings,
    hub: StubHfHub,
) -> tuple[Any, Any, threading.Event, threading.Event, list[list[PublicationOp]]]:
    if publisher == "core":
        module = core_publication
        options = SimpleNamespace(
            push=True,
            dry_run=True,
            upload_threads=2,
            commit_message=None,
        )
        monkeypatch.setattr(module, "StubHfHub", lambda: hub)
        entered, release, uploaded_ops = _install_upload_barrier(monkeypatch, module, hub=hub)
        queue = module._build_upload_queue(options, settings, data_root=data_root)
    elif publisher == "sync":
        module = sync_publication
        entered, release, uploaded_ops = _install_upload_barrier(monkeypatch, module, hub=hub)
        queue = sync_publication.build_upload_queue(
            push=True,
            dry_run=True,
            settings=settings,
            data_root=data_root,
            num_threads=2,
            _hub=hub,
        )
    else:
        raise AssertionError(f"unknown publisher: {publisher}")
    assert queue is not None
    return queue, module, entered, release, uploaded_ops


def _install_upload_barrier(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    hub: StubHfHub,
) -> tuple[threading.Event, threading.Event, list[list[PublicationOp]]]:
    entered = threading.Event()
    release = threading.Event()
    uploaded_ops: list[list[PublicationOp]] = []

    def upload(repo_id: str, **kwargs: Any) -> str:
        uploaded_ops.append(kwargs["ops"])
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test upload barrier was not released")
        kwargs["hub"] = hub
        return real_upload_files(repo_id, **kwargs)

    monkeypatch.setattr(module, "upload_files", upload)
    if module is core_publication:
        monkeypatch.setattr(module, "StubHfHub", lambda: hub)
    return entered, release, uploaded_ops


@pytest.mark.parametrize("publisher", ["core", "sync"])
@pytest.mark.parametrize("original_after_enqueue", ["mutated", "removed"])
def test_queued_publication_uploads_snapshot_after_canonical_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher: str,
    original_after_enqueue: str,
) -> None:
    canonical = tmp_path / "region.parquet"
    canonical.write_bytes(b"VERSION-1")
    data_root = DataRoot(tmp_path / "data")
    settings = Settings(repo_id="org/dataset", hf_token="test-token")
    hub = StubHfHub()
    queue, _module, entered, release, uploaded_ops = _publisher_queue(
        publisher,
        monkeypatch=monkeypatch,
        data_root=data_root,
        settings=settings,
        hub=hub,
    )

    try:
        queue.submit([add_op(canonical, path_in_repo="polygons/region.parquet")], "publish")
        assert entered.wait(timeout=10), "publisher did not reach the upload barrier"
        if original_after_enqueue == "mutated":
            canonical.write_bytes(b"VERSION-2")
        else:
            canonical.unlink()
    finally:
        release.set()
        failures = queue.close_and_wait()

    assert failures == []
    assert len(uploaded_ops) == 1
    assert hub.remote_content == {"polygons/region.parquet": b"VERSION-1"}


@pytest.mark.parametrize("publisher", ["core", "sync"])
def test_resumed_upload_uses_snapshot_after_original_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher: str,
) -> None:
    canonical = tmp_path / "region.parquet"
    canonical.write_bytes(b"VERSION-1")
    data_root = DataRoot(tmp_path / "data")
    settings = Settings(repo_id="org/dataset", hf_token="test-token")
    state_dir = data_root.cache / ("upload_jobs" if publisher == "core" else "sync_upload_jobs")

    def unavailable(*_args: Any) -> None:
        raise RuntimeError("simulated interruption before publication")

    interrupted = BackgroundUploadQueue(upload=unavailable, state_dir=state_dir, attempts=1)
    interrupted.submit([add_op(canonical, path_in_repo="polygons/region.parquet")], "resume")
    assert interrupted.close_and_wait()
    canonical.unlink()

    hub = StubHfHub()
    queue, _module, entered, release, uploaded_ops = _publisher_queue(
        publisher,
        monkeypatch=monkeypatch,
        data_root=data_root,
        settings=settings,
        hub=hub,
    )
    try:
        assert entered.wait(timeout=10), "resumed publisher did not reach the upload barrier"
    finally:
        release.set()
        failures = queue.close_and_wait()

    assert failures == []
    assert len(uploaded_ops) == 1
    assert hub.remote_content == {"polygons/region.parquet": b"VERSION-1"}


@pytest.mark.parametrize("publisher", ["core", "sync"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_resume_rejects_missing_or_corrupt_snapshot_without_live_file_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher: str,
    damage: str,
) -> None:
    canonical = tmp_path / "region.parquet"
    canonical.write_bytes(b"VERSION-2")
    data_root = DataRoot(tmp_path / "data")
    settings = Settings(repo_id="org/dataset", hf_token="test-token")
    state_dir = data_root.cache / ("upload_jobs" if publisher == "core" else "sync_upload_jobs")

    def unavailable(*_args: Any) -> None:
        raise RuntimeError("simulated interruption before publication")

    interrupted = BackgroundUploadQueue(upload=unavailable, state_dir=state_dir, attempts=1)
    interrupted.submit([add_op(canonical, path_in_repo="polygons/region.parquet")], "integrity")
    assert interrupted.close_and_wait()

    snapshot = next((state_dir / "snapshots").rglob("region.parquet"))
    if damage == "missing":
        snapshot.unlink()
    else:
        snapshot.write_bytes(b"CORRUPTED")
    envelope = next(state_dir.glob("*.json"))

    hub = StubHfHub()
    calls: list[list[PublicationOp]] = []

    def upload(repo_id: str, **kwargs: Any) -> str:
        calls.append(kwargs["ops"])
        kwargs["hub"] = hub
        return real_upload_files(repo_id, **kwargs)

    if publisher == "core":
        monkeypatch.setattr(core_publication, "upload_files", upload)
        monkeypatch.setattr(core_publication, "StubHfHub", lambda: hub)
        options = SimpleNamespace(push=True, dry_run=True, upload_threads=2, commit_message=None)
        queue = core_publication._build_upload_queue(options, settings, data_root=data_root)
    else:
        monkeypatch.setattr(sync_publication, "upload_files", upload)
        queue = sync_publication.build_upload_queue(
            push=True,
            dry_run=True,
            settings=settings,
            data_root=data_root,
            num_threads=2,
            _hub=hub,
        )
    assert queue is not None
    failures = queue.close_and_wait()

    assert failures
    assert any("snapshot" in failure.lower() for failure in failures)
    assert calls == []
    assert hub.remote_content == {}
    assert envelope.is_file()
    assert snapshot.exists() is (damage == "corrupt")
