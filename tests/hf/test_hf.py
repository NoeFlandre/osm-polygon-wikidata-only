"""Tests for the HF Hub subpackage (repo_layout, uploader, dataset_card)."""

from __future__ import annotations

import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_ARTICLES_DIR,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
    local_to_remote,
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


def test_resolve_hf_token_prefers_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert resolve_hf_token("hf_explicit") == "hf_explicit"


def test_resolve_hf_token_reads_hf_token_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert resolve_hf_token(None) == "hf_from_env"


def test_resolve_hf_token_returns_none_when_nothing_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    # resolve_hf_token falls back to huggingface_hub.get_token which may
    # still find a saved file. Assert the result is either None or a
    # non-empty string - the contract is "don't crash".
    token = resolve_hf_token(None)
    assert token is None or (isinstance(token, str) and token)


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


def test_stub_hub_preserves_upload_and_commit_source_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"path-bytes")
    stub = StubHfHub()

    for path, source in (
        ("upload-bytes", b"raw-bytes"),
        ("upload-bytearray", bytearray(b"array-bytes")),
        ("upload-path", artifact),
        ("upload-stream", BytesIO(b"stream-bytes")),
    ):
        stub.upload_file(
            path_or_fileobj=source,
            path_in_repo=path,
            repo_id="org/name",
            repo_type="dataset",
            commit_message="x",
        )

    operations = [
        SimpleNamespace(path_in_repo="commit-path", path_or_fileobj=str(artifact)),
        SimpleNamespace(path_in_repo="commit-bytes", path_or_fileobj=b"commit-raw"),
        SimpleNamespace(path_in_repo="commit-stream", path_or_fileobj=BytesIO(b"commit-stream")),
        SimpleNamespace(path_in_repo="commit-unsupported", path_or_fileobj=None),
    ]
    stub.create_commit(
        repo_id="org/name",
        operations=operations,
        commit_message="x",
        repo_type="dataset",
        num_threads=1,
    )

    assert stub.remote_content == {
        "upload-bytes": b"raw-bytes",
        "upload-bytearray": b"array-bytes",
        "upload-path": b"path-bytes",
        "upload-stream": b"stream-bytes",
        "commit-path": b"path-bytes",
        "commit-bytes": b"commit-raw",
        "commit-stream": b"commit-stream",
        "commit-unsupported": b"",
    }


def test_background_upload_submit_does_not_wait_for_upload(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op

    def upload(ops: list, message: str) -> None:
        started.set()
        release.wait(timeout=2)

    queue = BackgroundUploadQueue(upload=upload, max_pending=2)
    queue.submit([add_op(_small_parquet(tmp_path), path_in_repo="polygons/x.parquet")], "x")
    assert started.wait(timeout=1)
    release.set()
    assert queue.close_and_wait() == []


def test_background_upload_reports_worker_failure(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op

    def upload(ops: list, message: str) -> None:
        raise UploadError("offline")

    queue = BackgroundUploadQueue(upload=upload, max_pending=2)
    queue.submit([add_op(_small_parquet(tmp_path), path_in_repo="polygons/x.parquet")], "x")
    failures = queue.close_and_wait()
    assert len(failures) == 1
    assert "offline" in failures[0]


def test_background_upload_failure_is_resumed_from_durable_state(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op

    state = tmp_path / "jobs"
    artifact = _small_parquet(tmp_path)
    failing = BackgroundUploadQueue(
        upload=lambda ops, message: (_ for _ in ()).throw(UploadError("offline")),
        state_dir=state,
    )
    failing.submit([add_op(artifact, path_in_repo="polygons/x.parquet")], "x")
    assert failing.close_and_wait()
    assert len(list(state.glob("*.json"))) == 1

    calls: list[str] = []
    resumed = BackgroundUploadQueue(
        upload=lambda ops, message: calls.append(message),
        state_dir=state,
    )
    assert resumed.resume_pending() == 1
    assert resumed.close_and_wait() == []
    assert calls == ["x"]
    assert not list(state.glob("*.json"))


def test_stub_hf_hub_records_create_repo_call() -> None:
    stub = StubHfHub()
    stub.create_repo(repo_id="org/name", repo_type="dataset", exist_ok=True)
    assert stub.created_repos == [{"repo_id": "org/name", "repo_type": "dataset", "exist_ok": True}]


def test_upload_parquet_ensures_repo_exists(tmp_path: Path) -> None:
    stub = StubHfHub()
    p = _small_parquet(tmp_path)
    upload_parquet("org/name", p, path_in_repo="polygons/x.parquet", hub=stub)
    assert stub.created_repos == [{"repo_id": "org/name", "repo_type": "dataset", "exist_ok": True}]
    assert len(stub.uploads) == 1


def test_upload_files_ensures_repo_exists_before_commit(tmp_path: Path) -> None:
    stub = StubHfHub()
    polygon = _small_parquet(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    upload_files(
        "org/name",
        [(polygon, "polygons/x.parquet"), (manifest, REMOTE_MANIFEST_FILE)],
        hub=stub,
        commit_message="Update PBF x",
    )

    assert stub.created_repos == [{"repo_id": "org/name", "repo_type": "dataset", "exist_ok": True}]
    assert len(stub.commits) == 1


def test_upload_card_ensures_repo_exists() -> None:
    stub = StubHfHub()
    upload_card("org/name", "# card", hub=stub)
    assert stub.created_repos == [{"repo_id": "org/name", "repo_type": "dataset", "exist_ok": True}]
    assert len(stub.uploads) == 1


def test_upload_files_without_hub_requires_resolvable_token(tmp_path: Path) -> None:
    polygon = _small_parquet(tmp_path)
    with pytest.raises(UploadError, match="token"):
        upload_files(
            "org/name",
            [(polygon, "polygons/x.parquet")],
            commit_message="x",
            token=None,
            _resolve_token=lambda value: None,
        )


def test_upload_parquet_without_hub_requires_resolvable_token(tmp_path: Path) -> None:
    polygon = _small_parquet(tmp_path)
    with pytest.raises(UploadError, match="token"):
        upload_parquet(
            "org/name",
            polygon,
            path_in_repo="polygons/x.parquet",
            token=None,
            _resolve_token=lambda value: None,
        )


def test_upload_files_uses_resolved_token_to_build_hub(tmp_path: Path) -> None:
    captured: dict[str, str | None] = {}

    class _FakeApi:
        def __init__(self, *, token: str | None) -> None:
            captured["token"] = token
            self.token = token

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> str:
            return "commit-id"

    polygon = _small_parquet(tmp_path)
    upload_files(
        "org/name",
        [(polygon, "polygons/x.parquet")],
        commit_message="x",
        token=None,
        _resolve_token=lambda value: "my-token",
        _api_factory=lambda *, token: _FakeApi(token=token),
    )
    assert captured == {"token": "my-token"}


def test_upload_files_returns_the_upload_commit_oid(tmp_path: Path) -> None:
    class _CommitInfo:
        oid = "uploaded-commit-oid"

    class _FakeApi:
        token = "my-token"

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> _CommitInfo:
            return _CommitInfo()

    polygon = _small_parquet(tmp_path)
    result = upload_files(
        "org/name",
        [(polygon, "polygons/x.parquet")],
        commit_message="x",
        token="my-token",
        _api_factory=lambda *, token: _FakeApi(),
    )

    assert result == "uploaded-commit-oid"


def test_upload_files_prefers_oid_on_string_like_commit_info(tmp_path: Path) -> None:
    class _CommitInfo(str):
        oid = "uploaded-commit-oid"

    class _FakeApi:
        token = "my-token"

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> _CommitInfo:
            return _CommitInfo(
                "https://huggingface.co/datasets/org/name/commit/uploaded-commit-oid"
            )

    polygon = _small_parquet(tmp_path)
    result = upload_files(
        "org/name",
        [(polygon, "polygons/x.parquet")],
        commit_message="x",
        token="my-token",
        _api_factory=lambda *, token: _FakeApi(),
    )

    assert result == "uploaded-commit-oid"


def test_upload_files_can_report_noop_after_absent_delete_filtering() -> None:
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op

    hub = StubHfHub(remote_files=set())
    result = upload_files(
        "org/name",
        ops=[delete_op("language_splits/old.parquet")],
        hub=hub,
        token="stub-token",
        commit_message="idempotent cleanup",
        allow_noop=True,
    )

    assert result == ""
    assert hub.commits == []


def test_upload_files_rejects_empty_after_absent_delete_filtering_by_default() -> None:
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op

    hub = StubHfHub(remote_files=set())
    with pytest.raises(UploadError, match="No upload operations remain"):
        upload_files(
            "org/name",
            ops=[delete_op("language_splits/old.parquet")],
            hub=hub,
            token="stub-token",
            commit_message="idempotent cleanup",
        )

    assert hub.commits == []


def test_upload_files_requires_exactly_one_source() -> None:
    with pytest.raises(UploadError, match="exactly one"):
        upload_files(
            "org/name",
            files=(),
            ops=(),
            hub=StubHfHub(),
            token="stub-token",
            commit_message="invalid plan",
        )


def _fake_hf_response(status_code: int, body: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=body.encode("utf-8"),
        headers={"content-type": "text/plain"},
        request=httpx.Request("POST", "https://huggingface.co/api/datasets/x/y/preupload/main"),
    )


def test_upload_files_translates_repository_not_found_to_auth_hint(tmp_path: Path) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    class _AuthFailingApi:
        token = "good"

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> str:
            raise RepositoryNotFoundError(
                "401 Client Error",
                response=_fake_hf_response(401, "Invalid username or password"),
                server_message="Invalid username or password.",
            )

    polygon = _small_parquet(tmp_path)
    with pytest.raises(UploadError, match=r"Invalid username or password"):
        upload_files(
            "org/name",
            [(polygon, "polygons/x.parquet")],
            hub=_AuthFailingApi(),
            commit_message="x",
        )


def test_upload_files_translates_401_to_token_hint(tmp_path: Path) -> None:
    from huggingface_hub.errors import HfHubHTTPError

    class _BadTokenApi:
        token = "expired"

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> str:
            raise HfHubHTTPError(
                "Invalid user token.",
                response=_fake_hf_response(401, "Invalid user token."),
                server_message="Invalid user token.",
            )

    polygon = _small_parquet(tmp_path)
    with pytest.raises(UploadError, match=r"token"):
        upload_files(
            "org/name",
            [(polygon, "polygons/x.parquet")],
            hub=_BadTokenApi(),
            commit_message="x",
        )


def test_upload_files_translates_unexpected_exception_to_upload_error(
    tmp_path: Path,
) -> None:
    """Unexpected (non-auth) Hugging Face errors must surface as ``UploadError``
    with the documented ``Hugging Face upload to {repo_id} failed: ...``
    message. This pins the ``except Exception`` boundary in
    :mod:`hf._uploader.operations`: a narrow ``except`` would risk
    leaking raw ``Exception`` types to callers. The boundary is
    retained (not narrowed) because ``huggingface_hub`` legitimately
    raises a broad set of unstable exception types.
    """
    from huggingface_hub.errors import HfHubHTTPError

    class _ServerErrorApi:
        token = "good"

        def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> str:
            return repo_id

        def create_commit(self, **_kwargs: object) -> str:
            raise HfHubHTTPError(
                "500 Server Error",
                response=_fake_hf_response(500, "Internal Server Error: disk full"),
                server_message="Internal Server Error: disk full",
            )

    polygon = _small_parquet(tmp_path)
    with pytest.raises(UploadError, match=r"Hugging Face upload to org/name failed"):
        upload_files(
            "org/name",
            [(polygon, "polygons/x.parquet")],
            hub=_ServerErrorApi(),
            commit_message="x",
        )


def test_verify_repo_authorization_passes_when_user_matches_namespace() -> None:
    from osm_polygon_wikidata_only.hf.uploader import verify_repo_authorization

    assert (
        verify_repo_authorization("tok", "noeflandre/dataset", _verify=lambda t: "noeflandre")
        == "noeflandre"
    )


def test_verify_repo_authorization_raises_when_namespace_mismatches() -> None:
    from osm_polygon_wikidata_only.hf.uploader import verify_repo_authorization

    with pytest.raises(UploadError, match=r"noeflandre"):
        verify_repo_authorization("tok", "noeflandre/dataset", _verify=lambda t: "someoneelse")


def test_verify_repo_authorization_suggests_alt_repo_id_when_mismatched() -> None:
    from osm_polygon_wikidata_only.hf.uploader import verify_repo_authorization

    with pytest.raises(UploadError, match=r"--repo-id"):
        verify_repo_authorization("tok", "noeflandre/dataset", _verify=lambda t: "someoneelse")


def test_upload_card_rejects_empty() -> None:
    stub = StubHfHub()
    with pytest.raises(UploadError):
        upload_card("org/name", "", hub=stub)


def test_upload_card_records_markdown() -> None:
    stub = StubHfHub()
    remote = upload_card("org/name", "# card", hub=stub, commit_message="add card")
    assert remote == "README.md"
    assert stub.uploads[0]["size_bytes"] == len(b"# card")
