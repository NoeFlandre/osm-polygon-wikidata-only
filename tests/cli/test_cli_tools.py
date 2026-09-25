"""Unified entry point: --version and tool subcommands."""

from __future__ import annotations

import argparse
import re
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
import typer

from osm_polygon_wikidata_only.cli import commands, parser, tools
from osm_polygon_wikidata_only.cli.parser import build_parser
from osm_polygon_wikidata_only.v2.config import V2_TRACKIO_SPACE_ID

README = Path(__file__).resolve().parents[2] / "README.md"


def test_version_prints_package_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        commands.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == (
        f"osm-polygon-wikidata-only {parser.package_version()}"
    )


def test_package_version_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(parser, "distribution_version", missing)
    assert parser.package_version() == "unknown"


def test_enforce_integrity_subcommand_reuses_the_standalone_implementation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = commands.main(["enforce-integrity", "--data-root", str(tmp_path / "absent")])
    assert status == 1
    err = capsys.readouterr().err
    assert err.startswith("osm-polygon-wikidata-only enforce-integrity: error:")
    assert not (tmp_path / "absent").exists()


def test_audit_remote_subcommand_forwards_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_audit(**kwargs: object) -> None:
        seen.update(kwargs)
        raise typer.Exit(3)

    monkeypatch.setattr(tools.audit_remote, "audit", fake_audit)
    status = commands.main(["audit-remote", "--repo-id", "o/r", "--hf-token", "t"])
    assert status == 3
    assert seen == {"data_root": None, "repo_id": "o/r", "hf_token": "t"}


@pytest.mark.parametrize(
    ("argv", "module", "space_id"),
    [
        ([], tools.trackio_snapshot, tools.trackio_snapshot.TRACKIO_SPACE_ID),
        (["--dataset-version", "v2"], tools.v2_trackio_snapshot, V2_TRACKIO_SPACE_ID),
        (["--space-id", "me/space"], tools.trackio_snapshot, "me/space"),
    ],
)
def test_trackio_snapshot_subcommand_selects_the_dataset_version(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], module: object, space_id: str
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(module, "publish", lambda **kwargs: seen.update(kwargs))
    assert commands.main(["trackio-snapshot", *argv]) == 0
    assert seen == {"data_root": None, "space_id": space_id}


def test_non_tool_commands_are_not_dispatched_as_tools() -> None:
    assert tools.dispatch_tool(argparse.Namespace(command="sync-dir")) is None


def _subcommands() -> set[str]:
    sub_action = next(
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return set(sub_action.choices)


def test_readme_lists_every_subcommand() -> None:
    text = README.read_text(encoding="utf-8")
    listed = set(re.findall(r"uv run osm-polygon-wikidata-only ([a-z0-9-]+)", text))
    assert _subcommands() <= listed
