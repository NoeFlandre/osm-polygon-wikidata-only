"""Parse-then-dispatch contract shared by ``app.run`` and ``commands.main``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import app, commands, tools
from osm_polygon_wikidata_only.cli.dispatch import parse_and_dispatch
from osm_polygon_wikidata_only.cli.parser import build_parser


def _record_tool_dispatches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def fake_dispatch(args: argparse.Namespace) -> int | None:
        seen.append(args.command)
        return None

    monkeypatch.setattr(tools, "dispatch_tool", fake_dispatch)
    return seen


def test_app_run_offers_a_processing_command_to_tool_dispatch_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_tool_dispatches(monkeypatch)
    monkeypatch.setattr(commands, "run_parsed", lambda _parser, _args: 0)

    assert app.run(["process-dir", str(tmp_path)]) == 0
    assert seen == ["process-dir"]


def test_main_offers_a_processing_command_to_tool_dispatch_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_tool_dispatches(monkeypatch)
    monkeypatch.setattr(commands, "run_parsed", lambda _parser, _args: 0)

    assert commands.main(["process-dir", str(tmp_path)]) == 0
    assert seen == ["process-dir"]


def test_tool_status_short_circuits_the_command_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "dispatch_tool", lambda _args: 9)

    def unexpected(_parser: object, _args: object) -> int:
        raise AssertionError("the processing handler must not run after a tool status")

    assert parse_and_dispatch(build_parser(), ["audit-containment", str(tmp_path)], unexpected) == 9


def test_processing_command_is_handed_to_the_supplied_handler_with_its_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _record_tool_dispatches(monkeypatch)
    parser = build_parser()
    handed: list[tuple[argparse.ArgumentParser, str]] = []

    def handler(received: argparse.ArgumentParser, args: argparse.Namespace) -> int:
        handed.append((received, args.command))
        return 4

    assert parse_and_dispatch(parser, ["process-dir", str(tmp_path)], handler) == 4
    assert seen == ["process-dir"]
    assert handed == [(parser, "process-dir")]


def test_usage_errors_still_exit_with_argparse_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        commands.main(["no-such-command"])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
