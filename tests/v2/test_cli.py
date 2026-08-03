import argparse
from pathlib import Path
from types import SimpleNamespace

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def test_sync_dir_has_explicit_v2_selector() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", "/tmp/raw", "--dataset-version", "v2"])
    assert args.dataset_version == "v2"


def test_v2_selector_defaults_to_v1() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", str(Path("/tmp/raw"))])
    assert args.dataset_version == "v1"


def test_v2_selector_routes_to_isolated_executor(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    from osm_polygon_wikidata_only.cli.commands import main

    calls: list[str] = []
    monkeypatch.setattr(v2_cli, "execute_v2", lambda *args, **kwargs: calls.append("v2") or 0)
    raw = tmp_path / "raw"
    raw.mkdir()
    assert (
        main(["sync-dir", str(raw), "--data-root", str(tmp_path), "--dataset-version", "v2"]) == 0
    )
    assert calls == ["v2"]


def test_v2_executor_honors_an_explicit_repository_override(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli

    repos: list[str] = []
    monkeypatch.setattr(
        v2_cli,
        "build_wikimedia_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(wikipedia=object()),
    )
    monkeypatch.setattr(
        v2_cli.RemoteInventory,
        "fetch",
        staticmethod(lambda repo_id, **_kwargs: repos.append(repo_id) or RemoteInventory(set())),
    )
    monkeypatch.setattr(
        v2_cli,
        "upload_files",
        lambda repo_id, **_kwargs: repos.append(repo_id),
    )

    def run(_input, *, upload, **_kwargs):
        upload([], "test")
        return 0

    monkeypatch.setattr(v2_cli, "run_v2_sync", run)
    args = argparse.Namespace(
        dry_run=True,
        push=True,
        hf_token=None,
        commit_message=None,
        upload_threads=1,
        input=str(tmp_path / "raw"),
    )
    assert (
        v2_cli.execute_v2(
            args,
            data_root=DataRoot(tmp_path),
            settings=Settings(repo_id="example/v2"),
        )
        == 0
    )
    assert repos == ["example/v2", "example/v2"]
