from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_wikidata_only.hf import _upload_retry
from osm_polygon_wikidata_only.hf._upload_retry import _run_upload_attempts
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


@pytest.fixture
def recorded_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    def fake_wait(delay: float) -> bool:
        waits.append(delay)
        return False

    monkeypatch.setattr(_upload_retry, "wait_for_retry_or_cancel", fake_wait)
    return waits


def test_run_upload_attempts_retries_transient_failures_until_success(
    recorded_waits: list[float],
) -> None:
    calls: list[int] = []

    def upload(_ops: list, _message: str) -> None:
        calls.append(1)
        if len(calls) < 3:
            raise UploadError("temporary failure", transient=True)

    _run_upload_attempts(upload, [], "retry", attempts=3)

    assert len(calls) == 3
    assert len(recorded_waits) == 2


def test_run_upload_attempts_reraises_final_transient_failure(
    recorded_waits: list[float],
) -> None:
    calls: list[int] = []

    def upload(_ops: list, _message: str) -> None:
        calls.append(1)
        raise UploadError(f"failure {len(calls)}", transient=True)

    with pytest.raises(UploadError, match="failure 3"):
        _run_upload_attempts(upload, [], "retry", attempts=3)

    assert len(calls) == 3
    assert len(recorded_waits) == 2


def test_run_upload_attempts_does_not_retry_permanent_failures(
    recorded_waits: list[float],
) -> None:
    calls: list[int] = []

    def upload(_ops: list, _message: str) -> None:
        calls.append(1)
        raise UploadError("Invalid HF_TOKEN")

    with pytest.raises(UploadError, match="Invalid HF_TOKEN"):
        _run_upload_attempts(upload, [], "retry", attempts=3)

    assert len(calls) == 1
    assert recorded_waits == []


def test_run_upload_attempts_backs_off_and_logs_each_retried_failure(
    recorded_waits: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    def upload(_ops: list, _message: str) -> None:
        raise UploadError("rate limited", transient=True)

    with caplog.at_level("WARNING"), pytest.raises(UploadError):
        _run_upload_attempts(upload, [], "retry", attempts=3)

    assert 0.5 <= recorded_waits[0] <= 1.0
    assert 1.0 <= recorded_waits[1] <= 2.0
    retry_logs = [r for r in caplog.records if "retrying in" in r.getMessage()]
    assert len(retry_logs) == 2
    assert "rate limited" in retry_logs[0].getMessage()


def test_run_upload_attempts_stops_retrying_when_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(_upload_retry, "wait_for_retry_or_cancel", lambda _delay: True)

    def upload(_ops: list, _message: str) -> None:
        calls.append(1)
        raise UploadError("transient", transient=True)

    with pytest.raises(UploadError, match="transient"):
        _run_upload_attempts(upload, [], "retry", attempts=3)

    assert len(calls) == 1


class _HubHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code, text="server said no")


def test_translated_hub_errors_are_transient_only_for_retryable_statuses() -> None:
    assert operations._translate_hf_error(_HubHttpError(503), repo_id="a/b").transient
    assert operations._translate_hf_error(_HubHttpError(429), repo_id="a/b").transient
    assert not operations._translate_hf_error(_HubHttpError(404), repo_id="a/b").transient
    assert not operations._translate_hf_error(_HubHttpError(422), repo_id="a/b").transient


def test_translated_hub_error_without_response_is_transient_on_network_cause() -> None:
    try:
        raise ConnectionResetError(104, "reset")
    except ConnectionResetError as cause:
        try:
            raise RuntimeError("wrapped") from cause
        except RuntimeError as wrapped:
            translated = operations._translate_hf_error(wrapped, repo_id="a/b")
    assert translated.transient

    assert not operations._translate_hf_error(ValueError("bad payload"), repo_id="a/b").transient


def test_absent_delete_filter_is_a_noop_without_delete_paths() -> None:
    operation = SimpleNamespace(path_in_repo="region.parquet")

    class NoQueryHub:
        def file_exists(self, *_args: object, **_kwargs: object) -> bool:
            raise AssertionError("file_exists must not be called for an empty delete set")

    assert operations._drop_absent_deletes(
        cast(operations.HfHub, NoQueryHub()), "owner/repo", [operation], set()
    ) == [operation]


def test_existing_delete_paths_filters_remote_absences_and_translates_errors() -> None:
    class Hub:
        def file_exists(self, _repo_id: str, path: str, *, repo_type: str) -> bool:
            assert repo_type == "dataset"
            return path == "present.parquet"

    assert operations._existing_delete_paths(
        cast(operations.HfHub, Hub()), "owner/repo", {"present.parquet", "absent.parquet"}
    ) == {"present.parquet"}

    class FailingHub:
        def file_exists(self, *_args: object, **_kwargs: object) -> bool:
            raise RuntimeError("request failed")

    with pytest.raises(UploadError, match="request failed"):
        operations._existing_delete_paths(
            cast(operations.HfHub, FailingHub()), "owner/repo", {"present.parquet"}
        )


def test_response_message_falls_back_when_response_text_cannot_be_read() -> None:
    class BrokenResponse:
        @property
        def text(self) -> str:
            raise RuntimeError("response body unavailable")

    error = RuntimeError("request failed")
    setattr(error, "server_message", "server detail")
    assert operations._response_message(BrokenResponse(), error) == "server detail"

    error = RuntimeError("request failed")
    assert operations._response_message(BrokenResponse(), error) == "request failed"


def test_upload_queue_reads_only_current_envelopes(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf._upload_state_files import read_envelope as _read_envelope
    from osm_polygon_wikidata_only.hf.upload_queue import QUEUE_CONTRACT_VERSION

    envelope = tmp_path / "pending.json"
    envelope.write_text(json.dumps({"contract_version": QUEUE_CONTRACT_VERSION}))
    assert _read_envelope(envelope) is not None

    envelope.write_text(json.dumps({"message": "legacy"}))
    assert _read_envelope(envelope) is None


def test_upload_queue_removes_failed_upgrade_artifacts(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf._upload_state_files import remove_failed_upgrade

    state_dir = tmp_path / "state"
    snapshot_dir = state_dir / "snapshots" / "000001"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "data.bin").write_bytes(b"pending")
    state_path = state_dir / "000001.json"
    state_path.write_text("{}")

    remove_failed_upgrade(state_dir, 1, snapshot_dir)

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
