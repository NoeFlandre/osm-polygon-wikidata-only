from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, scenario, then, when

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.ids import content_hash
from osm_polygon_wikidata_only.v2.sentence_runner import run_v2_sentence_split
from osm_polygon_wikidata_only.v2.storage import write_v2_region

_FEATURE = Path(__file__).with_name("sentence_pipeline.feature")
_STEM = "acceptance-region-latest"


@dataclass
class _FakeSegmenter:
    fail_on_call: int | None = None
    model_id: str = "segment-any-text/sat-3l-sm"
    version: str = "acceptance-fake"
    revision: str = "acceptance-model"
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def split(
        self,
        texts: Sequence[str],
        *,
        language: str,
    ) -> list[list[str]]:
        self.calls.append((language, tuple(texts)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected segmentation interruption")
        return [_split_at_first_period(text) for text in texts]


@dataclass
class _ScenarioState:
    workspace: Path
    data_root: DataRoot = field(init=False)
    source_path: Path = field(init=False)
    output_path: Path = field(init=False)
    sections: list[dict[str, Any]] = field(default_factory=list)
    source_before: bytes = b""
    failed_segmenter: _FakeSegmenter | None = None
    resumed_segmenter: _FakeSegmenter | None = None
    resumed_rows: list[dict[str, Any]] | None = None
    resumed_manifest: dict[str, Any] | None = None
    clean_rows: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.data_root = DataRoot(self.workspace / "resumed")
        self.source_path = self.data_root.processed_v2 / "wikipedia/sections" / f"{_STEM}.parquet"
        self.output_path = self.data_root.processed_v2 / "wikipedia/sentences" / f"{_STEM}.parquet"


def _split_at_first_period(text: str) -> list[str]:
    cut = text.index(".") + 1
    return [text[:cut], text[cut:]]


def _section(section_id: str, language: str, text: str) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "document_id": f"document-{section_id}",
        "article_id": f"article-{section_id}",
        "project": "wikipedia",
        "language": language,
        "text": text,
        "content_hash": content_hash(text),
    }


def _write_region(data_root: DataRoot, sections: list[dict[str, Any]]) -> None:
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        _STEM,
        polygons=[],
        documents=[],
        links=[],
        sections=sections,
    )


@pytest.fixture
def scenario_state(tmp_path: Path) -> _ScenarioState:
    return _ScenarioState(workspace=tmp_path)


@given("a local V2 region with supported and unsupported language sections")
def local_v2_region(scenario_state: _ScenarioState) -> None:
    sections = [
        _section("en-1", "en", "Alpha. Beta!"),
        _section("xx-1", "xx", "未知. Keep this row?"),
        _section("fr-1", "fr", "Été. Voilà?"),
    ]
    _write_region(scenario_state.data_root, sections)
    scenario_state.sections = sections
    scenario_state.source_before = scenario_state.source_path.read_bytes()


@when("sentence splitting fails during the second supported-language batch")
def fail_during_second_supported_batch(scenario_state: _ScenarioState) -> None:
    segmenter = _FakeSegmenter(fail_on_call=2)
    with pytest.raises(RuntimeError, match="injected segmentation interruption"):
        run_v2_sentence_split(scenario_state.data_root, segmenter=segmenter, batch_size=1)
    scenario_state.failed_segmenter = segmenter


@then("the interrupted run leaves the source unchanged, a durable checkpoint, and no final output")
def interrupted_run_preserves_durable_state(scenario_state: _ScenarioState) -> None:
    assert scenario_state.source_path.read_bytes() == scenario_state.source_before
    assert not scenario_state.output_path.exists()
    assert scenario_state.failed_segmenter is not None
    assert [language for language, _ in scenario_state.failed_segmenter.calls] == ["en", "fr"]


@when("I resume sentence splitting")
def resume_sentence_splitting(scenario_state: _ScenarioState) -> None:
    segmenter = _FakeSegmenter()
    result = run_v2_sentence_split(scenario_state.data_root, segmenter=segmenter, batch_size=1)
    scenario_state.resumed_segmenter = segmenter
    scenario_state.resumed_rows = pq.read_table(scenario_state.output_path).to_pylist()
    scenario_state.resumed_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))


@then("resumed rows preserve exact text, offsets, and language routing")
def resumed_rows_preserve_text_offsets_and_routing(
    scenario_state: _ScenarioState,
) -> None:
    assert scenario_state.resumed_rows is not None
    assert scenario_state.resumed_segmenter is not None
    rows = scenario_state.resumed_rows
    expected_ids = ["en-1", "en-1", "xx-1", "fr-1", "fr-1"]
    assert [row["section_id"] for row in rows] == expected_ids

    sections_by_id = {section["section_id"]: section for section in scenario_state.sections}
    for section_id in dict.fromkeys(expected_ids):
        section = sections_by_id[section_id]
        section_rows = [row for row in rows if row["section_id"] == section_id]
        assert "".join(row["text"] for row in section_rows) == section["text"]
        offset = 0
        for index, row in enumerate(section_rows):
            assert row["sentence_index"] == index
            assert row["start_char"] == offset
            assert row["end_char"] == offset + len(row["text"])
            assert row["text"] == section["text"][row["start_char"] : row["end_char"]]
            offset = row["end_char"]
        assert offset == len(section["text"])
    assert [language for language, _ in scenario_state.resumed_segmenter.calls] == ["fr"]
    unsupported = next(row for row in scenario_state.resumed_rows or [] if row["language"] == "xx")
    assert unsupported["segmentation_status"] == "unsupported_language"
    assert unsupported["segmenter"] == "unsplit"
    assert unsupported["start_char"] == 0
    assert unsupported["end_char"] == len(unsupported["text"])


@then("the manifest records supported and unsupported languages")
def manifest_records_language_routing(scenario_state: _ScenarioState) -> None:
    assert scenario_state.resumed_manifest is not None
    assert scenario_state.resumed_manifest["supported_languages"]
    assert scenario_state.resumed_manifest["unsupported_languages"] == ["xx"]
    region = scenario_state.resumed_manifest["regions"]
    assert region == [
        {
            "stem": _STEM,
            "project": "wikipedia",
            "sections": 3,
            "split_sections": 2,
            "unsplit_sections": 1,
            "sentence_rows": 5,
            "supported_languages": ["en", "fr"],
            "unsupported_languages": ["xx"],
        }
    ]


@when("I run the same input cleanly")
def run_same_input_cleanly(scenario_state: _ScenarioState) -> None:
    clean_root = DataRoot(scenario_state.workspace / "clean")
    _write_region(clean_root, scenario_state.sections)
    run_v2_sentence_split(clean_root, segmenter=_FakeSegmenter(), batch_size=1)
    output_path = clean_root.processed_v2 / "wikipedia/sentences" / f"{_STEM}.parquet"
    scenario_state.clean_rows = pq.read_table(output_path).to_pylist()


@then("the resumed and clean Parquet outputs are identical")
def resumed_and_clean_outputs_are_identical(scenario_state: _ScenarioState) -> None:
    assert scenario_state.resumed_rows == scenario_state.clean_rows


@scenario(
    str(_FEATURE),
    "Resume after a segmentation failure without losing language or offsets",
)
def test_resumable_sentence_pipeline() -> None:
    pass
