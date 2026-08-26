from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.ids import content_hash
from osm_polygon_wikidata_only.v2.sentence_checkpoints import SentenceCheckpoint
from osm_polygon_wikidata_only.v2.sentence_runner import _process_source, run_v2_sentence_split
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def _section(section_id: str, language: str, text: str) -> dict[str, object]:
    row = {field.name: None for field in section_schema()}
    row.update(
        {
            "section_id": section_id,
            "document_id": f"document-{section_id}",
            "article_id": f"article-{section_id}",
            "project": "wikipedia",
            "language": language,
            "site": f"{language}wiki",
            "page_id": 1,
            "revision_id": 1,
            "section_index": 0,
            "heading": "",
            "anchor": "",
            "level": 0,
            "parent_section_id": "",
            "section_path": "[]",
            "text": text,
            "text_length_chars": len(text),
            "text_length_words": len(text.split()),
            "text_length_tokens_estimate": max(1, len(text) // 4),
            "content_hash": content_hash(text),
            "license": "CC BY-SA 4.0",
            "attribution": "Wikipedia",
        }
    )
    return row


@dataclass
class _FakeSegmenter:
    fail_on_call: int | None = None
    model_id: str = "segment-any-text/sat-3l-sm"
    version: str = "test"
    revision: str = "model-a"

    def __post_init__(self) -> None:
        self.calls = 0

    def split(self, texts: list[str], *, language: str) -> list[list[str]]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("segmentation interruption")
        assert language == "en"
        return [[text[: text.index(".") + 2], text[text.index(".") + 2 :]] for text in texts]


def test_v2_sentence_split_resumes_without_rewriting_sections(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    sections = [
        _section("en-1", "en", "First. Second."),
        _section("xx-1", "xx", "Unknown. Language."),
        _section("en-2", "en", "Third. Fourth."),
    ]
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
        sections=sections,
    )
    source_path = data_root.processed_v2 / "wikipedia/sections/region-latest.parquet"
    source_before = source_path.read_bytes()

    with pytest.raises(RuntimeError, match="segmentation interruption"):
        run_v2_sentence_split(
            data_root,
            segmenter=_FakeSegmenter(fail_on_call=2),
            batch_size=1,
        )

    checkpoint_batches = sorted(
        (data_root.v2_cache / "sentence-checkpoints" / "region-latest" / "wikipedia").glob(
            "batch-*.parquet"
        )
    )
    assert [path.name for path in checkpoint_batches] == [
        "batch-00000000.parquet",
        "batch-00000001.parquet",
    ]
    assert not (data_root.processed_v2 / "wikipedia/sentences/region-latest.parquet").exists()

    resumed_segmenter = _FakeSegmenter()
    result = run_v2_sentence_split(data_root, segmenter=resumed_segmenter, batch_size=1)

    output_path = data_root.processed_v2 / "wikipedia/sentences/region-latest.parquet"
    output = pq.read_table(output_path).to_pylist()
    assert resumed_segmenter.calls == 1
    assert [row["text"] for row in output] == [
        "First. ",
        "Second.",
        "Unknown. Language.",
        "Third. ",
        "Fourth.",
    ]
    assert [row["segmentation_status"] for row in output] == [
        "split",
        "split",
        "unsupported_language",
        "split",
        "split",
    ]
    assert source_path.read_bytes() == source_before
    assert result.manifest_path == data_root.processed_v2 / "manifests/sentence_splitting.json"
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"unsupported_languages":["xx"]' in manifest
    assert '"model_id":"segment-any-text/sat-3l-sm"' in manifest


def test_sentence_source_processing_returns_typed_accounting(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
        sections=[_section("en-1", "en", "First. Second.")],
    )
    checkpoint = SentenceCheckpoint(
        data_root.v2_cache / "sentence-checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=1,
    )

    result = _process_source(
        data_root.processed_v2 / "wikipedia/sections/region-latest.parquet",
        checkpoint=checkpoint,
        segmenter=_FakeSegmenter(),
        batch_size=1,
        stem="region-latest",
        project="wikipedia",
    )

    assert is_dataclass(result)
    assert result.batch_count == 1
    assert result.row_count == 2
    assert result.summary.sentence_rows == 2
