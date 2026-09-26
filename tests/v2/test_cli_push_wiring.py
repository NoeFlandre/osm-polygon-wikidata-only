"""Push-mode wiring of the V2 CLI boundary (inventory, uploader, Trackio)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import osm_polygon_wikidata_only.hf.v2_trackio_snapshot as v2_trackio
import osm_polygon_wikidata_only.v2.cli as v2_cli
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.uploader import UploadError
from osm_polygon_wikidata_only.v2.config import V2_TRACKIO_RUN_NAME


def _args(**fields: Any) -> argparse.Namespace:
    defaults = {"push": True, "dry_run": False, "commit_message": None, "upload_threads": 3}
    return argparse.Namespace(**{**defaults, **fields})


def _settings() -> Settings:
    return Settings(repo_id="example/v2", hf_token="hf_test")


def test_non_push_runs_build_no_remote_collaborators(tmp_path: Path) -> None:
    args = _args(push=False)
    assert v2_cli._fetch_inventory(args, "example/v2", _settings(), None) is None
    assert v2_cli._build_uploader(args, "example/v2", _settings(), None) is None
    assert v2_cli._build_trackio_publisher(args, DataRoot(tmp_path)) is None


def test_dry_run_push_skips_trackio(tmp_path: Path) -> None:
    assert v2_cli._build_trackio_publisher(_args(dry_run=True), DataRoot(tmp_path)) is None


def test_inventory_fetch_uses_repo_hub_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, object]] = []
    sentinel = object()

    def fetch(repo_id: str, *, hub: object, token: object) -> object:
        calls.append((repo_id, hub, token))
        return sentinel

    monkeypatch.setattr(v2_cli.RemoteInventory, "fetch", staticmethod(fetch))
    hub = object()
    assert v2_cli._fetch_inventory(_args(), "example/v2", _settings(), hub) is sentinel  # ty: ignore[invalid-argument-type]
    assert calls == [("example/v2", hub, "hf_test")]


def test_unlistable_new_repository_yields_no_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(*_args: object, **_kwargs: object) -> object:
        raise UploadError("404 repository not found")

    monkeypatch.setattr(v2_cli.RemoteInventory, "fetch", staticmethod(fetch))
    assert v2_cli._fetch_inventory(_args(), "example/v2", _settings(), None) is None


@pytest.mark.parametrize(
    ("commit_message", "expected"), [(None, "default message"), ("operator", "operator")]
)
def test_uploader_forwards_operations_and_message(
    monkeypatch: pytest.MonkeyPatch, commit_message: str | None, expected: str
) -> None:
    uploads: list[dict[str, Any]] = []
    monkeypatch.setattr(
        v2_cli, "upload_files", lambda repo_id, **kwargs: uploads.append({"repo": repo_id, **kwargs})
    )
    upload = v2_cli._build_uploader(
        _args(commit_message=commit_message), "example/v2", _settings(), None
    )
    assert upload is not None
    upload([], "default message")
    assert uploads == [
        {
            "repo": "example/v2",
            "ops": [],
            "hub": None,
            "token": "hf_test",
            "commit_message": expected,
            "num_threads": 3,
        }
    ]


def test_trackio_publisher_targets_the_v2_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(
        v2_trackio, "publish_v2_trackio_snapshot", lambda **kwargs: published.append(kwargs)
    )
    data_root = DataRoot(tmp_path)
    publish = v2_cli._build_trackio_publisher(_args(), data_root)
    assert publish is not None
    stats = object()
    publish(stats)  # ty: ignore[invalid-argument-type]
    assert published == [
        {"output_dir": data_root.cache / "trackio" / V2_TRACKIO_RUN_NAME, "stats": stats}
    ]
