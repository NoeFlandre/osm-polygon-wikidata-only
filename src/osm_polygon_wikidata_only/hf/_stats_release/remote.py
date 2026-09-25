"""Remote Hub state loading and revision-bound release verification."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from osm_polygon_wikidata_only.hf._publication.hub_snapshot import read_repo_sha
from osm_polygon_wikidata_only.hf._stats_release.models import (
    ReleasedFile,
    RemoteState,
    StatsReleaseError,
    hub_revision,
)
from osm_polygon_wikidata_only.hf._uploader.operations import build_hf_api as _build_hf_api
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.token import resolve_hf_token
from osm_polygon_wikidata_only.io.hashing import sha256_file


class RemoteVerifier(Protocol):
    """Confirm the released files exist at the uploaded revision."""

    def __call__(
        self,
        repo_id: str,
        files: tuple[ReleasedFile, ...],
        *,
        revision: str,
    ) -> str: ...


def _remote_revision(client: HfHub, repo_id: str) -> str | None:
    repo_info = getattr(client, "repo_info", None)
    if not callable(repo_info):
        return None
    revision = read_repo_sha(client, repo_id)
    return str(revision) if revision else None


def _remote_entries(client: HfHub, repo_id: str, path: str, revision: str) -> list[Any]:
    get_paths_info = getattr(client, "get_paths_info", None)
    if not callable(get_paths_info):
        return []
    try:
        entries = get_paths_info(
            repo_id,
            paths=[path],
            revision=hub_revision(revision),
            repo_type="dataset",
        )
    except TypeError:
        entries = get_paths_info(repo_id, paths=[path], repo_type="dataset")
    return list(entries)


def _download_remote_file(
    client: HfHub,
    repo_id: str,
    path: str,
    revision: str,
    *,
    cache_dir: Path | None,
) -> Path:
    download = getattr(client, "hf_hub_download", None)
    if not callable(download):
        raise StatsReleaseError("Hub client cannot download files for release verification")
    kwargs: dict[str, Any] = {
        "repo_type": "dataset",
        "revision": hub_revision(revision),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        return Path(download(repo_id, path, **kwargs))
    except TypeError:
        kwargs.pop("cache_dir", None)
        return Path(download(repo_id, path, **kwargs))


def load_remote_state(
    client: HfHub,
    repo_id: str,
    paths: Sequence[str],
    *,
    cache_dir: Path,
) -> RemoteState:
    try:
        revision = _remote_revision(client, repo_id)
    # Deliberately broad: any failure to read the remote revision (auth,
    # network, a client without repo metadata) means "no remote card to
    # merge". The subsequent upload and verification still fail loudly.
    except Exception:
        return RemoteState(None, {})
    if not revision:
        return RemoteState(None, {})
    contents: dict[str, bytes] = {}
    for path in paths:
        content = _download_remote_content(
            client,
            repo_id,
            path,
            revision,
            cache_dir=cache_dir,
        )
        if content is not None:
            contents[path] = content
    return RemoteState(revision, contents)


def _download_remote_content(
    client: HfHub,
    repo_id: str,
    path: str,
    revision: str,
    *,
    cache_dir: Path,
) -> bytes | None:
    try:
        if not _remote_entries(client, repo_id, path, revision):
            return None
        local_path = _download_remote_file(
            client,
            repo_id,
            path,
            revision,
            cache_dir=cache_dir,
        )
        return local_path.read_bytes()
    except (OSError, StatsReleaseError):
        return None


def changed_paths(
    remote: RemoteState,
    files: Sequence[ReleasedFile],
    local_paths: Mapping[str, Path],
) -> tuple[str, ...]:
    if not remote.revision:
        return tuple(item.path_in_repo for item in files)
    return tuple(
        item.path_in_repo
        for item in files
        if remote.contents.get(item.path_in_repo) != local_paths[item.path_in_repo].read_bytes()
    )


def client_for_release(hub: HfHub | None, token: str | None) -> HfHub:
    return hub or cast(HfHub, _build_hf_api(resolve_hf_token(token)))


def _invoke_custom_verifier(
    verifier: RemoteVerifier,
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    revision: str,
) -> str:
    legacy_verifier = cast(Callable[[str, tuple[ReleasedFile, ...]], str], verifier)
    try:
        parameters = inspect.signature(verifier).parameters.values()
    except (TypeError, ValueError):
        return legacy_verifier(repo_id, files)
    accepts_keyword = any(
        parameter.name == "revision" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_keyword:
        return verifier(repo_id, files, revision=revision)
    return legacy_verifier(repo_id, files)


def verify_uploaded_files(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    upload_revision: str,
    verifier: RemoteVerifier | None,
    client: HfHub | None,
    token: str | None,
    cache_dir: Path,
) -> str:
    if verifier is None:
        verified_revision = default_remote_verifier(
            repo_id,
            files,
            revision=upload_revision,
            hub=client,
            token=token,
            cache_dir=cache_dir,
        )
    else:
        verified_revision = _invoke_custom_verifier(
            verifier,
            repo_id,
            files,
            upload_revision,
        )
    if not verified_revision:
        raise StatsReleaseError("remote verification returned an empty revision")
    if verified_revision != upload_revision:
        raise StatsReleaseError(
            "remote verification was not bound to the uploaded commit: "
            f"uploaded={upload_revision}, verified={verified_revision}"
        )
    return verified_revision


def _remote_revision_for_verifier(
    client: HfHub,
    repo_id: str,
    revision: str | None,
) -> str:
    selected = revision or _remote_revision(client, repo_id)
    if not selected:
        raise StatsReleaseError(f"hub repository {repo_id} returned an empty revision")
    return selected


def _verify_remote_file(
    client: HfHub,
    repo_id: str,
    item: ReleasedFile,
    revision: str,
    *,
    cache_dir: Path | None,
) -> None:
    entries = _remote_entries(client, repo_id, item.path_in_repo, revision)
    entry = _find_remote_entry(entries, item.path_in_repo)
    if entry is None:
        raise StatsReleaseError(f"remote file missing in revision {revision}: {item.path_in_repo}")
    _verify_remote_size(entry, item)
    _verify_remote_hash(
        client,
        repo_id,
        item,
        revision,
        cache_dir=cache_dir,
    )


def _find_remote_entry(entries: Sequence[Any], path: str) -> Any | None:
    return next(
        (candidate for candidate in entries if getattr(candidate, "path", path) == path),
        None,
    )


def _verify_remote_size(entry: Any, item: ReleasedFile) -> None:
    size = getattr(entry, "size", None)
    if size is not None and int(size) != item.size_bytes:
        raise StatsReleaseError(
            f"remote size mismatch for {item.path_in_repo}: local={item.size_bytes}, remote={size}"
        )


def _verify_remote_hash(
    client: HfHub,
    repo_id: str,
    item: ReleasedFile,
    revision: str,
    *,
    cache_dir: Path | None,
) -> None:
    local_path = _download_remote_file(
        client,
        repo_id,
        item.path_in_repo,
        revision,
        cache_dir=cache_dir,
    )
    if sha256_file(local_path) != item.sha256:
        raise StatsReleaseError(f"remote SHA-256 mismatch for {item.path_in_repo}")


def default_remote_verifier(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    revision: str | None = None,
    hub: HfHub | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Confirm each released file's identity at one exact Hub revision."""
    client = client_for_release(hub, token)
    selected_revision = _remote_revision_for_verifier(client, repo_id, revision)
    for item in files:
        _verify_remote_file(
            client,
            repo_id,
            item,
            selected_revision,
            cache_dir=cache_dir,
        )
    return selected_revision
