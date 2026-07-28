"""Fresh-process resumability for normal regional augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.models import Document
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    sidecar_paths,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from tests.augmentation.test_augmentation import (
    FakeAugmentationClient,
    article_row,
)


class _RecordingClient(FakeAugmentationClient):
    def __init__(self, *, fail_revision: int | None = None) -> None:
        self.fail_revision = fail_revision
        self.entity_props: list[str] = []
        self.parse_revisions: list[int] = []
        self.voyage_calls = 0

    def entities(
        self,
        qids: list[str] | set[str],
        *,
        props: str,
    ) -> dict[str, dict[str, Any]]:
        self.entity_props.append(props)
        return super().entities(qids, props=props)

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        self.parse_revisions.append(revision_id)
        if revision_id == self.fail_revision:
            raise KeyboardInterrupt("simulated operator interruption")
        return super().parse_html(project, language, revision_id)

    def wikivoyage_document(
        self,
        qid: str,
        language: str,
        site: str,
        title: str,
    ) -> Document:
        self.voyage_calls += 1
        return super().wikivoyage_document(qid, language, site, title)


def _seed(root: Path) -> DataRoot:
    data_root = DataRoot(root)
    data_root.ensure()
    pq.write_table(
        pa.Table.from_pylist([article_row()]),
        data_root.processed_articles / "andorra-latest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"wikidata": "Q1"}]),
        data_root.processed_polygons / "andorra-latest.parquet",
    )
    return data_root


def test_interrupted_augmentation_reuses_completed_phases_and_section_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "SECTION_CHECKPOINT_BATCH_SIZE", 1)
    data_root = _seed(tmp_path / "seagate")
    first = _RecordingClient(fail_revision=40)

    with pytest.raises(KeyboardInterrupt, match="operator interruption"):
        augment_region(data_root, "andorra-latest", first)

    checkpoint_root = data_root.cache / "augmentation_checkpoints"
    assert checkpoint_root.is_dir()
    assert first.entity_props == ["sitelinks|claims"]
    assert first.voyage_calls == 1
    assert first.parse_revisions == [20, 40]

    resumed = _RecordingClient()
    augment_region(data_root, "andorra-latest", resumed)

    assert "sitelinks|claims" not in resumed.entity_props
    assert resumed.voyage_calls == 0
    assert resumed.parse_revisions == [40]
    assert not (checkpoint_root / "andorra-latest").exists()


def test_resumed_outputs_equal_uninterrupted_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "SECTION_CHECKPOINT_BATCH_SIZE", 1)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.augmentation.wikimedia.utc_now_iso",
        lambda: "2026-01-01T00:00:00Z",
    )
    resumed_root = _seed(tmp_path / "resumed")
    baseline_root = _seed(tmp_path / "baseline")

    with pytest.raises(KeyboardInterrupt):
        augment_region(
            resumed_root,
            "andorra-latest",
            _RecordingClient(fail_revision=40),
        )
    augment_region(resumed_root, "andorra-latest", _RecordingClient())
    augment_region(baseline_root, "andorra-latest", _RecordingClient())

    for resumed_path, baseline_path in zip(
        sidecar_paths(resumed_root, "andorra-latest"),
        sidecar_paths(baseline_root, "andorra-latest"),
        strict=True,
    ):
        resumed: pa.Table = pq.read_table(resumed_path)  # type: ignore[no-untyped-call]
        baseline: pa.Table = pq.read_table(baseline_path)  # type: ignore[no-untyped-call]
        assert resumed.equals(baseline)


def test_manifest_failure_preserves_checkpoints_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator_mod

    data_root = _seed(tmp_path / "seagate")

    def fail_manifest(*_args: object, **_kwargs: object) -> Path:
        raise OSError("manifest unavailable")

    monkeypatch.setattr(orchestrator_mod, "update_augmentation_manifest", fail_manifest)

    with pytest.raises(OSError, match="manifest unavailable"):
        augment_region(data_root, "andorra-latest", _RecordingClient())

    assert (data_root.cache / "augmentation_checkpoints" / "andorra-latest").is_dir()
