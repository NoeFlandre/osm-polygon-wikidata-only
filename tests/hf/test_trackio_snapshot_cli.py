"""Typer entry points for the V1 and V2 Trackio snapshot publishers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from osm_polygon_wikidata_only.hf import trackio_snapshot, v2_trackio_snapshot
from osm_polygon_wikidata_only.hf._trackio.publisher import publish_trackio_snapshot
from osm_polygon_wikidata_only.v2.card import V2CardStats
from osm_polygon_wikidata_only.v2.config import V2_TRACKIO_RUN_NAME, V2_TRACKIO_SPACE_URL
from tests.hf.test_trackio_snapshot import _FakeTrackio


def _stats() -> V2CardStats:
    return V2CardStats(
        regions=2,
        polygons=11,
        unique_wikidata_entities=9,
        wikipedia_documents=7,
        wikipedia_sections=14,
        wikivoyage_documents=3,
        wikivoyage_sections=6,
        wikidata_facts=20,
        polygon_document_links=12,
        wikipedia_tag_only_polygons=2,
        document_words=101,
        languages=4,
        new_polygons_vs_v1=2,
        new_wikipedia_documents_vs_v1=3,
        text_coverage_funnel=(("All polygons", 11),),
        top_wikipedia_languages=(("en", 7),),
        polygon_link_storage_bytes=1_000_000_000,
        total_parquet_storage_bytes=2_000_000_000,
    )


def _inject_fake_trackio(monkeypatch: pytest.MonkeyPatch, module: Any) -> _FakeTrackio:
    fake = _FakeTrackio()

    def publish_with_fake(**kwargs: Any) -> Any:
        return publish_trackio_snapshot(**kwargs, trackio_module=fake)

    monkeypatch.setattr(module, "publish_trackio_snapshot", publish_with_fake)
    return fake


def test_v1_publish_command_writes_artifacts_under_the_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _inject_fake_trackio(monkeypatch, trackio_snapshot)
    data_root = tmp_path / "data"

    result = CliRunner().invoke(
        trackio_snapshot.app,
        ["--data-root", str(data_root), "--space-id", "example/space"],
    )

    assert result.exit_code == 0, result.output
    assert "Trackio run published: https://huggingface.co/spaces/example/space" in result.output
    expected_dir = data_root / "cache" / "trackio" / trackio_snapshot.TRACKIO_RUN_NAME
    assert f"Artifacts: {expected_dir}" in result.output
    assert fake.sync_kwargs is not None
    assert fake.sync_kwargs["space_id"] == "example/space"


def test_v2_publish_command_uses_data_derived_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _inject_fake_trackio(monkeypatch, v2_trackio_snapshot)
    seen: list[tuple[Path, Path]] = []

    def fake_stats(processed_v2: Path, *, v1_processed: Path) -> V2CardStats:
        seen.append((processed_v2, v1_processed))
        return _stats()

    monkeypatch.setattr(v2_trackio_snapshot, "compute_v2_card_stats", fake_stats)
    data_root = tmp_path / "data"

    result = CliRunner().invoke(
        v2_trackio_snapshot.app,
        ["--data-root", str(data_root), "--space-id", "example/v2-space"],
    )

    assert result.exit_code == 0, result.output
    assert f"Trackio run published: {V2_TRACKIO_SPACE_URL}" in result.output
    assert str(data_root / "cache" / "trackio" / V2_TRACKIO_RUN_NAME) in result.output
    assert len(seen) == 1
    assert fake.logged is not None
    assert fake.logged["metrics"]["scale/polygons"] == 11
    assert fake.sync_kwargs is not None
    assert fake.sync_kwargs["space_id"] == "example/v2-space"


def test_publish_v2_snapshot_passes_the_v2_identity(tmp_path: Path) -> None:
    fake = _FakeTrackio()
    artifacts = v2_trackio_snapshot.publish_v2_trackio_snapshot(
        output_dir=tmp_path / "out", stats=_stats(), space_id="example/s", trackio_module=fake
    )
    assert artifacts.output_dir == tmp_path / "out"
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["name"] == V2_TRACKIO_RUN_NAME


@pytest.mark.parametrize("module", [trackio_snapshot, v2_trackio_snapshot])
def test_console_entry_points_render_help(
    module: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["trackio-snapshot", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.run()
    assert exit_info.value.code == 0
    assert "Trackio" in capsys.readouterr().out
