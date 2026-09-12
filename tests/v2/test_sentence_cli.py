from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import osm_polygon_wikidata_only.cli.commands as commands
import osm_polygon_wikidata_only.v2.cli as v2_cli
from osm_polygon_wikidata_only.cli.parser import build_parser
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings


def test_sentence_split_command_dispatches_to_its_dedicated_runner(
    tmp_path: Path, monkeypatch
) -> None:
    args = argparse.Namespace(command="split-v2-sentences")
    parser = build_parser()
    calls: list[str] = []
    monkeypatch.setattr(
        commands,
        "_run_v2_sentence_split",
        lambda _parser, _args, *, data_root, settings: calls.append(str(data_root.path)) or 17,
    )

    result = commands._dispatch_command(
        parser,
        args,
        data_root=DataRoot(tmp_path),
        settings=Settings(),
    )

    assert result == 17
    assert calls == [str(tmp_path)]


def test_sentence_split_runner_holds_dedicated_lock_and_delegates(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = DataRoot(tmp_path)
    args = argparse.Namespace(command="split-v2-sentences")
    settings = Settings(repo_id="example/repo")
    locked_paths: list[Path] = []
    delegated: list[tuple[argparse.Namespace, DataRoot, Settings]] = []

    @contextmanager
    def fake_lock(path: Path) -> Iterator[None]:
        locked_paths.append(path)
        yield

    def fake_execute(
        received_args: argparse.Namespace,
        *,
        data_root: DataRoot,
        settings: Settings,
    ) -> int:
        delegated.append((received_args, data_root, settings))
        return 23

    monkeypatch.setattr(commands, "exclusive_run_lock", fake_lock)
    monkeypatch.setattr(v2_cli, "execute_v2_sentence_split", fake_execute)

    result = commands._run_v2_sentence_split(
        build_parser(), args, data_root=data_root, settings=settings
    )

    assert result == 23
    assert locked_paths == [data_root.cache / "sentence-splitting.lock"]
    assert len(delegated) == 1
    received_args, received_root, received_settings = delegated[0]
    assert received_args is args
    assert received_root is data_root
    assert received_settings is settings
