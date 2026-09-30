"""Remote inventory translates expected HTTP failures, not programming bugs."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


class _BrokenHub(StubHfHub):
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[str]:
        del repo_id, revision, repo_type
        raise RuntimeError("unexpected Hub client bug")

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str],
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[Any]:
        del repo_id, paths, revision, repo_type
        raise RuntimeError("unexpected Hub client bug")


class _OfflineHub(StubHfHub):
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[str]:
        del repo_id, revision, repo_type
        raise httpx.ConnectError("Hub is unreachable")

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str],
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[Any]:
        del repo_id, paths, revision, repo_type
        raise httpx.ConnectError("Hub is unreachable")


@pytest.mark.parametrize("fetch_paths", [False, True], ids=["inventory", "paths"])
def test_unexpected_hub_errors_propagate(fetch_paths: bool) -> None:
    if fetch_paths:
        with pytest.raises(RuntimeError, match="unexpected Hub client bug"):
            RemoteInventory.fetch_paths("owner/dataset", paths=["README.md"], hub=_BrokenHub())
    else:
        with pytest.raises(RuntimeError, match="unexpected Hub client bug"):
            RemoteInventory.fetch("owner/dataset", hub=_BrokenHub())


@pytest.mark.parametrize("fetch_paths", [False, True], ids=["inventory", "paths"])
def test_http_errors_are_translated_to_upload_errors(fetch_paths: bool) -> None:
    if fetch_paths:
        with pytest.raises(UploadError, match="Hugging Face upload"):
            RemoteInventory.fetch_paths("owner/dataset", paths=["README.md"], hub=_OfflineHub())
    else:
        with pytest.raises(UploadError, match="Hugging Face upload"):
            RemoteInventory.fetch("owner/dataset", hub=_OfflineHub())
