from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_wikidata_only.hf._uploader import operations
from osm_polygon_wikidata_only.hf._uploader.errors import UploadError


def test_upload_operation_validation_rejects_empty_and_duplicate_paths() -> None:
    with pytest.raises(UploadError, match="empty upload commit"):
        operations._validate_upload_operations([])

    duplicate = [
        SimpleNamespace(path_in_repo="same.parquet"),
        SimpleNamespace(path_in_repo="same.parquet"),
    ]
    with pytest.raises(UploadError, match="duplicate remote paths"):
        operations._validate_upload_operations(duplicate)


def test_absent_delete_filter_is_a_noop_without_delete_paths() -> None:
    operation = SimpleNamespace(path_in_repo="region.parquet")

    class NoQueryHub:
        def file_exists(self, *_args: object, **_kwargs: object) -> bool:
            raise AssertionError("file_exists must not be called for an empty delete set")

    assert operations._drop_absent_deletes(NoQueryHub(), "owner/repo", [operation], set()) == [
        operation
    ]


def test_existing_delete_paths_filters_remote_absences_and_translates_errors() -> None:
    class Hub:
        def file_exists(self, _repo_id: str, path: str, *, repo_type: str) -> bool:
            assert repo_type == "dataset"
            return path == "present.parquet"

    assert operations._existing_delete_paths(
        Hub(), "owner/repo", {"present.parquet", "absent.parquet"}
    ) == {"present.parquet"}

    class FailingHub:
        def file_exists(self, *_args: object, **_kwargs: object) -> bool:
            raise RuntimeError("request failed")

    with pytest.raises(UploadError, match="request failed"):
        operations._existing_delete_paths(FailingHub(), "owner/repo", {"present.parquet"})


def test_upload_queue_reads_only_current_envelopes(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf.upload_queue import (
        QUEUE_CONTRACT_VERSION,
        _read_envelope,
    )

    envelope = tmp_path / "pending.json"
    envelope.write_text(json.dumps({"contract_version": QUEUE_CONTRACT_VERSION}))
    assert _read_envelope(envelope) is not None

    envelope.write_text(json.dumps({"message": "legacy"}))
    assert _read_envelope(envelope) is None


def test_upload_queue_removes_failed_upgrade_artifacts(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    state_dir = tmp_path / "state"
    snapshot_dir = state_dir / "snapshots" / "000001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "data.bin").write_bytes(b"pending")
    state_path = state_dir / "000001.json"
    state_path.write_text("{}")

    BackgroundUploadQueue._remove_failed_upgrade(state_dir, 1, snapshot_dir)

    assert not state_path.exists()
    assert not snapshot_dir.exists()


def test_upload_queue_synchronous_upload_requires_closed_queue() -> None:
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    calls: list[str] = []
    queue = BackgroundUploadQueue(upload=lambda _ops, message: calls.append(message))
    with pytest.raises(RuntimeError, match="drained, closed"):
        queue.upload_synchronously([], "too-early")
    queue.close_and_wait()

    queue.upload_synchronously([], "final")
    assert calls == ["final"]


def test_upload_queue_rejects_submit_after_close() -> None:
    from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue

    queue = BackgroundUploadQueue(upload=lambda _ops, _message: None)
    queue.close_and_wait()
    with pytest.raises(RuntimeError, match="upload queue is closed"):
        queue.submit([], "late")
