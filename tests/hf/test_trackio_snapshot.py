"""Contracts for the frozen public Trackio dataset snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from osm_polygon_wikidata_only.hf.trackio_snapshot import (
    DATASET_PRESENTATION_URL,
    FINAL_DATASET_SNAPSHOT,
    TRACKIO_RUN_NAME,
    publish_trackio_snapshot,
    render_snapshot_markdown,
)


class _FakeTrackio:
    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] | None = None
        self.logged: dict[str, object] | None = None
        self.finished = 0
        self.sync_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> object:
        self.init_kwargs = kwargs
        return SimpleNamespace()

    @staticmethod
    def Markdown(text: str) -> tuple[str, str]:
        return ("markdown", text)

    @staticmethod
    def Table(**kwargs: object) -> tuple[str, dict[str, object]]:
        return ("table", kwargs)

    @staticmethod
    def Image(value: object, caption: str | None = None) -> tuple[str, object, str | None]:
        return ("image", value, caption)

    def log(self, metrics: dict[str, object], *, step: int) -> None:
        self.logged = {"metrics": metrics, "step": step}

    def finish(self) -> None:
        self.finished += 1

    def sync(self, **kwargs: object) -> str:
        self.sync_kwargs = kwargs
        return "example/frozen-snapshot"


def test_snapshot_contains_only_requested_static_metrics() -> None:
    assert FINAL_DATASET_SNAPSHOT.metrics() == {
        "scale/polygons": 1_184_110,
        "scale/unique_wikidata_entities": 1_119_223,
        "corpus/documents": 2_288_170,
        "corpus/document_words": 801_528_334,
        "coverage/languages": 351,
        "coverage/geographic_regions": 375,
    }
    assert FINAL_DATASET_SNAPSHOT.other_wikipedia_languages == 1_257_718


def test_snapshot_markdown_is_public_and_excludes_runtime_telemetry() -> None:
    markdown = render_snapshot_markdown()
    assert "final-dataset-snapshot" in markdown
    assert "trackio" in markdown.lower()
    assert DATASET_PRESENTATION_URL in markdown
    assert "2,468,604" in markdown
    assert "19.2 GB" in markdown
    for forbidden in ("runtime", "API calls", "retries", "throughput", "processed regions"):
        assert forbidden.lower() not in markdown.lower()


def test_publish_logs_exactly_three_plots_and_no_runtime_metrics(tmp_path: Path) -> None:
    fake = _FakeTrackio()
    artifacts = publish_trackio_snapshot(
        output_dir=tmp_path / "trackio",
        space_id="example/frozen-snapshot",
        trackio_module=fake,
    )

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["name"] == TRACKIO_RUN_NAME
    assert fake.init_kwargs["space_id"] is None
    assert fake.init_kwargs["resume"] == "never"
    assert fake.init_kwargs["auto_log_cpu"] is False
    assert fake.init_kwargs["auto_log_gpu"] is False
    assert fake.logged is not None
    assert fake.logged["step"] == 0
    metrics = fake.logged["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {
        "scale/polygons",
        "scale/unique_wikidata_entities",
        "corpus/documents",
        "corpus/document_words",
        "coverage/languages",
        "coverage/geographic_regions",
        "report/summary",
        "report/table",
        "plot/text_coverage_funnel",
        "plot/top_10_wikipedia_languages",
        "plot/dataset_composition",
    }
    assert fake.finished == 1
    assert fake.sync_kwargs == {
        "project": "osm-polygon-wikidata-only",
        "space_id": "example/frozen-snapshot",
        "dataset_id": "NoeFlandre/osm-polygon-wikidata-only-trackio-data",
        "sdk": "static",
        "force": True,
    }
    assert len(artifacts.chart_paths) == 3
    assert all(path.is_file() for path in artifacts.chart_paths)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == TRACKIO_RUN_NAME
    assert manifest["table"] == [
        ["Wikipedia polygon-document links", "2,468,604"],
        ["Polygon/link-table storage", "9.9 GB"],
        ["Total Parquet storage", "19.2 GB"],
    ]
