"""Step definitions for ``enforce_integrity.feature`` (#88, #115)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, scenarios, then, when

from osm_polygon_wikidata_only.cli import enforce_integrity
from osm_polygon_wikidata_only.config.paths import DataRoot
from tests.augmentation.test_integrity import (
    _minimal_link_row,
    _minimal_polygon_row,
    _write_polygon_articles,
    _write_polygons,
)

scenarios("enforce_integrity.feature")


@dataclass
class _State:
    data_root: DataRoot | None = None
    before: dict[str, bytes] = field(default_factory=dict)
    status: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def state() -> _State:
    return _State()


@given("a processed tree with a polygon article link to an unknown QID")
def seeded_defect(state: _State, tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path / "data")
    data_root.processed_polygons.mkdir(parents=True)
    data_root.processed_links.mkdir(parents=True)
    _write_polygons(
        data_root.processed_polygons / "italy-latest.parquet",
        [_minimal_polygon_row("italy-latest:way:1", "Q1")],
    )
    _write_polygon_articles(
        data_root.processed_links / "italy-latest.parquet",
        [_minimal_link_row("italy-latest:way:1", "Q2")],
    )
    state.data_root = data_root
    state.before = _snapshot(data_root.path)


def _run(state: _State, capsys: pytest.CaptureFixture[str], *extra: str) -> None:
    assert state.data_root is not None
    state.status = enforce_integrity.run(
        ["--data-root", str(state.data_root.path), "--json", *extra]
    )
    state.summary = json.loads(capsys.readouterr().out)


@when("I run enforce-integrity with --dry-run")
def run_dry(state: _State, capsys: pytest.CaptureFixture[str]) -> None:
    _run(state, capsys, "--dry-run")


@when("I run enforce-integrity")
def run_real(state: _State, capsys: pytest.CaptureFixture[str]) -> None:
    _run(state, capsys)


@then("the summary reports one rejected polygon article link")
def one_rejection(state: _State) -> None:
    assert state.status == 0
    assert state.summary["polygon_articles_rejected"] == 1


@then("the processed files are byte-identical and no audit is written")
def untouched(state: _State) -> None:
    assert state.data_root is not None
    assert state.summary["dry_run"] is True
    assert state.summary["audit_path"] is None
    assert _snapshot(state.data_root.path) == state.before


@then("an audit file is written")
def audit_written(state: _State) -> None:
    assert state.summary["dry_run"] is False
    assert Path(state.summary["audit_path"]).is_file()
