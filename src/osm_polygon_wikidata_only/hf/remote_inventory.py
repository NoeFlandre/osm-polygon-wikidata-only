"""Provide read-only inventory helpers for Hugging Face repositories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from osm_polygon_wikidata_only.hf._uploader.operations import _build_hf_api, _translate_hf_error
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token


@dataclass(frozen=True, slots=True)
class RemoteFileInfo:
    """Hub metadata needed to verify one remote file without downloading it."""

    path: str
    size: int
    sha256: str | None


class RemoteInventory:
    """Read-only representation of canonical files present on the Hugging Face Hub."""

    def __init__(
        self,
        files: set[str],
        metadata: Mapping[str, RemoteFileInfo] | None = None,
    ) -> None:
        self._files = files
        self._metadata = dict(metadata or {})

    @classmethod
    def fetch(
        cls,
        repo_id: str,
        *,
        hub: HfHub | None = None,
        token: str | None = None,
        _resolve_token: Any = resolve_hf_token,
        _api_factory: Any = None,
    ) -> RemoteInventory:
        """Fetch files in dataset repository exactly once."""
        client = _resolve_hub_client(
            hub=hub,
            token=token,
            resolve_token=_resolve_token,
            api_factory=_api_factory,
        )
        try:
            files = client.list_repo_files(repo_id=repo_id, repo_type="dataset")
            return cls(set(files))
        except Exception as error:
            raise _translate_hf_error(error, repo_id=repo_id) from error

    @classmethod
    def fetch_paths(
        cls,
        repo_id: str,
        *,
        paths: Sequence[str],
        hub: HfHub | None = None,
        token: str | None = None,
        _resolve_token: Any = resolve_hf_token,
        _api_factory: Any = None,
    ) -> RemoteInventory:
        """Fetch metadata for exactly the requested remote paths."""
        client = _resolve_hub_client(
            hub=hub,
            token=token,
            resolve_token=_resolve_token,
            api_factory=_api_factory,
        )
        get_paths_info = getattr(client, "get_paths_info", None)
        if not callable(get_paths_info):
            return cls.fetch(repo_id, hub=client, token=token)
        try:
            entries = get_paths_info(
                repo_id=repo_id,
                paths=list(paths),
                repo_type="dataset",
            )
            metadata = {
                info.path: info
                for entry in entries
                if (info := _remote_file_info(entry)) is not None
            }
            return cls(set(metadata), metadata)
        except Exception as error:
            raise _translate_hf_error(error, repo_id=repo_id) from error

    def contains(self, path_in_repo: str) -> bool:
        return path_in_repo in self._files

    @property
    def files(self) -> set[str]:
        return self._files

    def metadata(self, path_in_repo: str) -> RemoteFileInfo | None:
        """Return exact Hub metadata for a fetched path, when available."""
        return self._metadata.get(path_in_repo)


def _resolve_hub_client(
    *,
    hub: HfHub | None,
    token: str | None,
    resolve_token: Any,
    api_factory: Any,
) -> HfHub:
    if hub is not None:
        return hub
    return _build_hf_api(resolve_token(token), api_factory=api_factory)


def _remote_file_info(entry: Any) -> RemoteFileInfo | None:
    fields = _remote_file_fields(entry)
    if fields is None:
        return None
    path, size = fields
    return RemoteFileInfo(path=path, size=size, sha256=_remote_sha256(entry))


def _remote_file_fields(entry: Any) -> tuple[str, int] | None:
    path = getattr(entry, "path", None)
    size = getattr(entry, "size", None)
    if isinstance(path, str) and isinstance(size, int):
        return path, size
    return None


def _remote_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    return sha256 if isinstance(sha256, str) and len(sha256) == 64 else None
