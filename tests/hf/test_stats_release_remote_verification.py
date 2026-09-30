"""Remote Hub state loading and revision-bound verification with a fake Hub."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.hf._stats_release import remote
from osm_polygon_wikidata_only.hf._stats_release.models import (
    ReleasedFile,
    RemoteState,
    StatsReleaseError,
)
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

REVISION = "a" * 40


class FakeHub(StubHfHub):
    """In-memory dataset repository at one revision."""

    def __init__(
        self,
        root: Path,
        files: dict[str, bytes],
        *,
        sha: str | None = REVISION,
        sizes: dict[str, int] | None = None,
        legacy_paths_info: bool = False,
        legacy_download: bool = False,
    ) -> None:
        super().__init__(remote_files=set(files), remote_content=files)
        self.root = root
        self.files = files
        self.sha = sha
        self.sizes = sizes or {}
        self.legacy_paths_info = legacy_paths_info
        self.legacy_download = legacy_download
        self.downloads: list[dict[str, Any]] = []

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any:
        del repo_id
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.sha)

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str],
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[Any]:
        del repo_id
        if self.legacy_paths_info and revision is not None:
            raise TypeError("unexpected keyword 'revision'")
        return [
            SimpleNamespace(path=path, size=self.sizes.get(path, len(self.files[path])))
            for path in paths
            if path in self.files
        ]

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str,
        repo_type: str,
        cache_dir: str | None = None,
    ) -> str:
        del repo_id, revision, repo_type
        if self.legacy_download and cache_dir is not None:
            raise TypeError("unexpected keyword 'cache_dir'")
        if cache_dir is not None:
            self.downloads.append({"cache_dir": cache_dir})
        else:
            self.downloads.append({})
        target = self.root / "downloads" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


def _released(path: str, content: bytes) -> ReleasedFile:
    return ReleasedFile(path, hashlib.sha256(content).hexdigest(), len(content))


def test_load_remote_state_downloads_only_existing_paths(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {"README.md": b"card"})
    state = remote.load_remote_state(
        hub, "o/r", ["README.md", "missing.json"], cache_dir=tmp_path / "cache"
    )
    assert state == RemoteState(REVISION, {"README.md": b"card"})
    assert hub.downloads[0]["cache_dir"] == str(tmp_path / "cache")


def test_load_remote_state_supports_legacy_client_signatures(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {"a": b"1"}, legacy_paths_info=True, legacy_download=True)
    state = remote.load_remote_state(hub, "o/r", ["a"], cache_dir=tmp_path)
    assert state.contents == {"a": b"1"}
    assert "cache_dir" not in hub.downloads[0]


def test_load_remote_state_without_revision_is_empty(tmp_path: Path) -> None:
    assert remote.load_remote_state(
        FakeHub(tmp_path, {}, sha=None), "o/r", ["a"], cache_dir=tmp_path
    ) == RemoteState(None, {})


def test_load_remote_state_treats_revision_errors_as_new_repo(tmp_path: Path) -> None:
    class Broken:
        def repo_info(self, *_args: Any, **_kwargs: Any) -> object:
            raise ConnectionError("offline")

    assert remote.load_remote_state(Broken(), "o/r", ["a"], cache_dir=tmp_path) == RemoteState(  # ty: ignore[invalid-argument-type]
        None, {}
    )


def test_client_without_repo_metadata_has_no_remote_state(tmp_path: Path) -> None:
    assert remote.load_remote_state(object(), "o/r", ["a"], cache_dir=tmp_path) == RemoteState(  # ty: ignore[invalid-argument-type]
        None, {}
    )


def test_client_without_download_skips_content(tmp_path: Path) -> None:
    class NoDownload:
        def repo_info(self, *_args: Any, **_kwargs: Any) -> object:
            return SimpleNamespace(sha=REVISION)

        def get_paths_info(self, *_args: Any, paths: list[str], **_kwargs: Any) -> list[object]:
            return [SimpleNamespace(path=paths[0])]

    state = remote.load_remote_state(NoDownload(), "o/r", ["a"], cache_dir=tmp_path)  # ty: ignore[invalid-argument-type]
    assert state == RemoteState(REVISION, {})


def test_changed_paths_compares_local_bytes(tmp_path: Path) -> None:
    same, changed = tmp_path / "same", tmp_path / "changed"
    same.write_bytes(b"x")
    changed.write_bytes(b"new")
    files = [_released("same", b"x"), _released("changed", b"new")]
    local = {"same": same, "changed": changed}
    state = RemoteState(REVISION, {"same": b"x", "changed": b"old"})
    assert remote.changed_paths(state, files, local) == ("changed",)
    assert remote.changed_paths(RemoteState(None, {}), files, local) == ("same", "changed")


def test_default_verifier_accepts_matching_files(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {"a": b"alpha"})
    assert (
        remote.default_remote_verifier("o/r", (_released("a", b"alpha"),), hub=hub, cache_dir=None)
        == REVISION
    )
    assert "cache_dir" not in hub.downloads[0]


def test_default_verifier_rejects_empty_revision(tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="returned an empty revision"):
        remote.default_remote_verifier("o/r", (), hub=FakeHub(tmp_path, {}, sha=None))


def test_default_verifier_rejects_missing_remote_file(tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="remote file missing in revision"):
        remote.default_remote_verifier(
            "o/r", (_released("a", b"x"),), revision=REVISION, hub=FakeHub(tmp_path, {})
        )


def test_default_verifier_rejects_size_mismatch(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {"a": b"x"}, sizes={"a": 99})
    with pytest.raises(StatsReleaseError, match="remote size mismatch for a: local=1, remote=99"):
        remote.default_remote_verifier("o/r", (_released("a", b"x"),), hub=hub)


def test_default_verifier_rejects_hash_mismatch(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {"a": b"y"})
    with pytest.raises(StatsReleaseError, match="remote SHA-256 mismatch for a"):
        remote.default_remote_verifier("o/r", (_released("a", b"x"),), hub=hub)


def test_verification_requires_a_download_capable_client(tmp_path: Path) -> None:
    class NoDownload(FakeHub):
        hf_hub_download = None  # type: ignore[assignment]

    with pytest.raises(StatsReleaseError, match="cannot download files"):
        remote.default_remote_verifier(
            "o/r",
            (_released("a", b"x"),),
            hub=cast(HfHub, NoDownload(tmp_path, {"a": b"x"})),
        )


def _verify(verifier: Any, *, upload_revision: str = REVISION, client: Any = None) -> str:
    return remote.verify_uploaded_files(
        "o/r",
        (),
        upload_revision=upload_revision,
        verifier=verifier,
        client=client,
        token=None,
        cache_dir=Path("unused"),
    )


def test_keyword_custom_verifier_receives_revision() -> None:
    seen: list[str] = []

    def verifier(repo_id: str, files: tuple[ReleasedFile, ...], *, revision: str) -> str:
        seen.append(revision)
        return revision

    assert _verify(verifier) == REVISION
    assert seen == [REVISION]


def test_var_keyword_custom_verifier_receives_revision() -> None:
    assert _verify(lambda _repo, _files, **kwargs: kwargs["revision"]) == REVISION


def test_legacy_custom_verifier_is_called_positionally() -> None:
    assert _verify(lambda _repo, _files: REVISION) == REVISION


def test_uninspectable_custom_verifier_falls_back_to_legacy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_obj: object) -> None:
        raise ValueError("no signature")

    monkeypatch.setattr(remote.inspect, "signature", refuse)
    assert _verify(lambda _repo, _files: REVISION) == REVISION


def test_empty_verified_revision_is_rejected() -> None:
    with pytest.raises(StatsReleaseError, match="empty revision"):
        _verify(lambda _repo, _files: "")


def test_verification_bound_to_another_commit_is_rejected() -> None:
    with pytest.raises(StatsReleaseError, match="not bound to the uploaded commit"):
        _verify(lambda _repo, _files: "b" * 40)


def test_default_verification_path_uses_supplied_client(tmp_path: Path) -> None:
    assert _verify(None, client=FakeHub(tmp_path, {})) == REVISION


def test_client_for_release_prefers_supplied_hub(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, {})
    assert remote.client_for_release(hub, "token") is hub
