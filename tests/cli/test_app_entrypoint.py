"""Smoke tests for the installed ``osm-polygon-wikidata-only`` entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

from osm_polygon_wikidata_only.cli import app


def test_console_script_targets_cli_app_run() -> None:
    scripts = entry_points(group="console_scripts")
    (script,) = [item for item in scripts if item.name == "osm-polygon-wikidata-only"]
    assert script.value == "osm_polygon_wikidata_only.cli.app:run"
    assert script.load() is app.run


def test_run_delegates_to_the_command_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["osm-polygon-wikidata-only", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        app.run()
    assert exit_info.value.code == 0
    assert "sync-dir" in capsys.readouterr().out


def test_python_dash_m_entrypoint_renders_help() -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and module
        [sys.executable, "-m", "osm_polygon_wikidata_only.cli.app", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "sync-dir" in completed.stdout


def test_python_dash_m_entrypoint_rejects_unknown_command() -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and module
        [sys.executable, "-m", "osm_polygon_wikidata_only.cli.app", "no-such-command"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
