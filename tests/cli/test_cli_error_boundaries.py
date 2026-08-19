"""Failure-boundary tests for CLI process ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import osm_polygon_wikidata_only.cli.commands as commands
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
    with pytest.raises(SystemExit) as excinfo:
        commands.main(argv)

    assert excinfo.value.code == 2
    assert "Unified sync is already running" in capsys.readouterr().err
