from __future__ import annotations

import argparse
from pathlib import Path

import osm_polygon_wikidata_only.cli.commands as commands
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
