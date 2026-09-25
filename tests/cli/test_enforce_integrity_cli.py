"""CLI contract for ``osm-polygon-wikidata-only-enforce-integrity``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import enforce_integrity
from osm_polygon_wikidata_only.config.paths import DataRoot
from tests.augmentation.test_integrity import (
    _minimal_link_row,
    _minimal_polygon_row,
    _write_polygon_articles,
    _write_polygons,
)


def _seed_defect(root: Path) -> DataRoot:
    data_root = DataRoot(root)
    data_root.processed_polygons.mkdir(parents=True)
    data_root.processed_links.mkdir(parents=True)
    stem = "italy-latest"
    _write_polygons(
        data_root.processed_polygons / f"{stem}.parquet",
        [_minimal_polygon_row("italy-latest:way:1", "Q1")],
    )
    _write_polygon_articles(
        data_root.processed_links / f"{stem}.parquet",
        [_minimal_link_row("italy-latest:way:1", "Q2")],
    )
    return data_root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dry_run_leaves_processed_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _seed_defect(tmp_path / "data")
    before = _snapshot(data_root.processed)

    status = enforce_integrity.run(["--data-root", str(data_root.path), "--dry-run", "--json"])

    assert status == 0
    assert _snapshot(data_root.processed) == before
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "audit_path": None,
        "dry_run": True,
        "polygon_articles_rejected": 1,
        "wikivoyage_documents_rejected": 0,
        "wikivoyage_sections_cascaded": 0,
    }


def test_real_run_rewrites_and_reports_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _seed_defect(tmp_path / "data")

    status = enforce_integrity.run(["--data-root", str(data_root.path), "--json"])

    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is False
    assert summary["polygon_articles_rejected"] == 1
    assert Path(summary["audit_path"]).is_file()


def test_summary_is_logged_at_default_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_root = _seed_defect(tmp_path / "data")
    with caplog.at_level(logging.INFO):
        assert enforce_integrity.run(["--data-root", str(data_root.path)]) == 0
    assert "1 polygon_articles rejected" in caplog.text
    assert "Audit written to" in caplog.text


def test_dry_run_log_says_nothing_was_written(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_root = _seed_defect(tmp_path / "data")
    with caplog.at_level(logging.INFO):
        assert enforce_integrity.run(["--data-root", str(data_root.path), "--dry-run"]) == 0
    assert "Integrity dry run complete" in caplog.text
    assert "no audit written" in caplog.text


def test_missing_data_root_exits_nonzero_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OSM_POLYGON_DATA_ROOT", raising=False)
    status = enforce_integrity.run(["--data-root", str(tmp_path / "absent")])

    assert status == 1
    err = capsys.readouterr().err
    assert err.startswith(f"{enforce_integrity.PROG}: error: ")
    assert "does not exist" in err
    assert "Traceback" not in err


def test_corrupt_parquet_exits_nonzero_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _seed_defect(tmp_path / "data")
    (data_root.processed_links / "italy-latest.parquet").write_bytes(b"not parquet")

    status = enforce_integrity.run(["--data-root", str(data_root.path)])

    assert status == 1
    assert "error:" in capsys.readouterr().err


def test_unexpected_errors_propagate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _seed_defect(tmp_path / "data")

    def boom(*args: object, **kwargs: object) -> None:
        raise TypeError("programming error")

    monkeypatch.setattr(enforce_integrity, "enforce_all_regions", boom)
    with pytest.raises(TypeError, match="programming error"):
        enforce_integrity.run(["--data-root", str(data_root.path)])
