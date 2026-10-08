"""Failure-boundary tests for CLI process ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import osm_polygon_wikidata_only.cli.commands as commands
from osm_polygon_wikidata_only.cli.errors import CliFailure
from osm_polygon_wikidata_only.io.run_lock import RunLockError


@pytest.mark.parametrize("dataset_version", ["v1", "v2"])
def test_main_fails_fast_when_another_sync_holds_the_run_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_version: str,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    @contextmanager
    def busy_lock(path: Path) -> Iterator[None]:
        del path
        raise RunLockError("Unified sync is already running")
        yield

    monkeypatch.setattr(commands, "exclusive_run_lock", busy_lock)

    argv = [
        "sync-dir",
        str(raw),
        "--data-root",
        str(tmp_path),
        "--dataset-version",
        dataset_version,
    ]
    assert commands.main(argv) == 1

    err = capsys.readouterr().err
    assert err.splitlines()[-1] == (
        "osm-polygon-wikidata-only: error: Unified sync is already running"
    )


def _raise_boom(*_args: object, **_kwargs: object) -> int:
    raise CliFailure("boom")


def test_run_parsed_reports_an_expected_failure_once_and_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --data-root pins the run to tmp_path; the failure is raised by the public
    # publication service that the processing command dispatches into.
    monkeypatch.setattr(commands, "run_core_publication", _raise_boom)
    parser = commands.build_parser()
    args = parser.parse_args(["process-dir", str(tmp_path), "--data-root", str(tmp_path)])

    assert commands.run_parsed(parser, args) == 1
    assert capsys.readouterr().err == "osm-polygon-wikidata-only: error: boom\n"


def test_background_upload_failure_is_reported_on_stderr_and_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        commands,
        "run_core_publication",
        lambda *_a, **_k: SimpleNamespace(results=[], upload_failures=["stem-a", "stem-b"]),
    )

    status = commands.main(["process-dir", str(tmp_path), "--data-root", str(tmp_path)])

    assert status == 1
    assert capsys.readouterr().err == (
        "osm-polygon-wikidata-only: error: 2 background upload(s) failed\n"
    )
