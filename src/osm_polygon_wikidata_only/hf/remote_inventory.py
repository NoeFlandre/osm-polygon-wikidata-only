from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.hf._uploader.operations import _build_hf_api, _translate_hf_error
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token


class RemoteInventory:
    """Read-only representation of canonical files present on the Hugging Face Hub."""

    def __init__(self, files: set[str]) -> None:
        self._files = files

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
        if hub is not None:
            client = hub
        else:
            resolved_token = _resolve_token(token)
            client = _build_hf_api(resolved_token, api_factory=_api_factory)
        try:
            files = client.list_repo_files(repo_id=repo_id, repo_type="dataset")
            return cls(set(files))
        except Exception as error:
            raise _translate_hf_error(error, repo_id=repo_id) from error

    def contains(self, path_in_repo: str) -> bool:
        return path_in_repo in self._files

    @property
    def files(self) -> set[str]:
        return self._files
