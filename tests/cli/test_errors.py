"""Shared CLI error reporting."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.cli.errors import EXIT_FAILURE, report_cli_error


def test_report_cli_error_writes_one_line_to_stderr_and_returns_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = report_cli_error("osm-polygon-wikidata-only demo", ValueError("bad contract"))

    captured = capsys.readouterr()
    assert status == EXIT_FAILURE == 1
    assert captured.out == ""
    assert captured.err == "osm-polygon-wikidata-only demo: error: bad contract\n"
