"""Shared CLI error reporting."""

from __future__ import annotations

import re

import pytest
from typer.models import CommandInfo

from osm_polygon_wikidata_only.cli.errors import EXIT_FAILURE, _with_notice, report_cli_error


def test_report_cli_error_writes_one_line_to_stderr_and_returns_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = report_cli_error("osm-polygon-wikidata-only demo", ValueError("bad contract"))

    captured = capsys.readouterr()
    assert status == EXIT_FAILURE == 1
    assert captured.out == ""
    assert captured.err == "osm-polygon-wikidata-only demo: error: bad contract\n"


def test_with_notice_rejects_a_command_without_a_callback() -> None:
    command = CommandInfo(callback=None)

    with pytest.raises(
        RuntimeError,
        match=re.escape("Typer commands registered with @app.command() have a callback"),
    ):
        _with_notice(command, "legacy-name", "replacement-name")
