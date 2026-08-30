from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.ids import content_hash
from osm_polygon_wikidata_only.v2.sentence_checkpoints import SentenceCheckpoint
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema
from osm_polygon_wikidata_only.v2.sentence_runner import (
    SentenceRegionSummary,
    _load_manifest_payload,
    _load_manifest_summaries,
    _manifest_metadata,
    _manifest_payload,
    _manifest_region_summary,
    _manifest_uses_segmenter,
    _process_source,
    _requested_stems,
    _run_project,
    _selected_stems,
    _summary_from_mapping,
    _write_checkpoint_batches,
    _write_output,
    run_v2_sentence_split,
)
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


def test_summary_decoder_preserves_manifest_values() -> None:
    summary = _summary_from_mapping(
        {
            "stem": "region-latest",
            "project": "wikipedia",
            "sections": 3,
            "split_sections": 2,
            "unsplit_sections": 1,
            "sentence_rows": 4,
            "supported_languages": ["en"],
            "unsupported_languages": ["xx"],
        },
        error="invalid summary",
    )

    assert summary == SentenceRegionSummary(
        stem="region-latest",
        project="wikipedia",
        sections=3,
        split_sections=2,
        unsplit_sections=1,
        sentence_rows=4,
        supported_languages=("en",),
        unsupported_languages=("xx",),
    )


def test_manifest_metadata_is_shared_by_writer_and_reader() -> None:
    assert _manifest_metadata(_FakeSegmenter()) == {
        "contract_version": "v2-sentence-splitting-v1",
        "segmenter": "sat-3l-sm",
        "model_id": "segment-any-text/sat-3l-sm",
        "model_revision": "model-a",
        "segmenter_version": "test",
    }


def test_requested_stems_are_sorted_and_deduplicated() -> None:
    manifest = {"b-latest": {}, "a-latest": {}}

    assert _requested_stems(manifest, ("b-latest", "a-latest", "b-latest")) == (
        "a-latest",
        "b-latest",
    )
    assert _requested_stems(manifest, None) == ("a-latest", "b-latest")


def test_manifest_region_summary_decodes_one_region() -> None:
    region = _manifest_region_summary(
        {
            "stem": "region-latest",
            "project": "wikipedia",
            "sections": 1,
            "split_sections": 1,
            "unsplit_sections": 0,
            "sentence_rows": 2,
            "supported_languages": ["en"],
            "unsupported_languages": [],
        },
        path=Path("manifest.json"),
    )

    assert region.stem == "region-latest"
    assert region.sentence_rows == 2


def test_manifest_payload_keeps_regions_and_language_policy() -> None:
    summary = SentenceRegionSummary(
        stem="region-latest",
        project="wikipedia",
        sections=1,
        split_sections=1,
        unsplit_sections=0,
        sentence_rows=2,
        supported_languages=("en",),
        unsupported_languages=("xx",),
    )

    payload = _manifest_payload(_FakeSegmenter(), [summary])

    assert payload["regions"] == [asdict(summary)]
    assert payload["unsupported_languages"] == ["xx"]
    assert payload["unsupported_language_policy"] == "one unsplit row; never passed to SaT"


def test_manifest_payload_loader_decodes_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "sentence_splitting.json"
    payload = {"regions": [], "segmenter": "sat-3l-sm"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_manifest_payload(path) == payload


def test_manifest_segmenter_identity_is_checked_against_all_metadata() -> None:
    payload = _manifest_metadata(_FakeSegmenter())

    assert _manifest_uses_segmenter(payload, _FakeSegmenter())
    assert not _manifest_uses_segmenter(
        {**payload, "model_revision": "other-revision"},
        _FakeSegmenter(),
    )


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

    cached_segmenter = _FakeSegmenter()
    cached_result = run_v2_sentence_split(
        data_root,
        segmenter=cached_segmenter,
        batch_size=1,
    )
    assert cached_segmenter.calls == 0
    assert cached_result.regions == result.regions


@pytest.mark.parametrize("batch_size", [0, -1])
def test_v2_sentence_split_rejects_non_positive_batch_size(
    tmp_path: Path,
    batch_size: int,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(ValueError, match="batch_size must be positive"):
        run_v2_sentence_split(data_root, segmenter=_FakeSegmenter(), batch_size=batch_size)


def test_v2_sentence_split_requires_supported_segmenter(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    segmenter = _FakeSegmenter(model_id="other/model")
    with pytest.raises(ValueError, match="Only sat-3l-sm is supported"):
        run_v2_sentence_split(data_root, segmenter=segmenter)


def test_selected_stems_rejects_missing_empty_and_invalid_values() -> None:
    with pytest.raises(FileNotFoundError, match="No finalized V2 regions"):
        _selected_stems({}, None)
    with pytest.raises(ValueError, match="not finalized"):
        _selected_stems({"region-latest": {}}, ("missing-latest",))
    with pytest.raises(ValueError, match="At least one"):
        _selected_stems({"region-latest": {}}, ())
    with pytest.raises(ValueError, match="Invalid V2 stem"):
        _selected_stems({".": {}}, (".",))


def test_run_project_rejects_missing_section_source(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(FileNotFoundError, match="V2 section source is missing"):
        _run_project(
            data_root,
            stem="region-latest",
            project="wikipedia",
            segmenter=_FakeSegmenter(),
            batch_size=1,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "Invalid sentence manifest"),
        ({"regions": {}}, "Invalid sentence manifest regions"),
        ({"regions": [None]}, "Invalid sentence manifest region"),
        ({"regions": [{}]}, "Invalid sentence manifest region"),
    ],
)
def test_sentence_manifest_rejects_invalid_payloads(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "sentence_splitting.json"
    if isinstance(payload, dict):
        payload = {**_manifest_metadata(_FakeSegmenter()), **payload}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_manifest_summaries(path, segmenter=_FakeSegmenter())


def test_sentence_manifest_rejects_different_segmenter(tmp_path: Path) -> None:
    path = tmp_path / "sentence_splitting.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": "v2-sentence-splitting-v1",
                "segmenter": "sat-3l-sm",
                "model_id": "other/model",
                "model_revision": "model-a",
                "segmenter_version": "test",
                "regions": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different segmenter"):
        _load_manifest_summaries(path, segmenter=_FakeSegmenter())


def test_sentence_output_cleans_temporary_file_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=1,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])

    def fail_validation(path: Path, schema: object) -> None:
        raise RuntimeError(f"invalid output: {path}")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.sentence_runner._validate_schema",
        fail_validation,
    )

    with pytest.raises(RuntimeError, match="invalid output"):
        _write_output(tmp_path / "sentences.parquet", checkpoint, batch_count=1)

    assert list(tmp_path.glob("*.tmp")) == []


def test_partial_sentence_run_preserves_previous_manifest_regions(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for stem in ("first-latest", "second-latest"):
        write_v2_region(
            data_root.processed_v2,
            stem,
            polygons=[],
            documents=[],
            links=[],
            sections=[_section(f"{stem}-en", "en", "First. Second.")],
        )

    segmenter = _FakeSegmenter()
    run_v2_sentence_split(data_root, segmenter=segmenter, batch_size=1, stems=("first-latest",))
    run_v2_sentence_split(data_root, segmenter=segmenter, batch_size=1, stems=("second-latest",))

    manifest = json.loads(
        (data_root.processed_v2 / "manifests/sentence_splitting.json").read_text(encoding="utf-8")
    )

    assert {region["stem"] for region in manifest["regions"]} == {
        "first-latest",
        "second-latest",
    }


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


def test_sentence_source_processing_reuses_batch_metadata_without_loading_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    write_v2_region(
        data_root.processed_v2,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
        sections=[_section("en-1", "en", "First.")],
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
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])
    monkeypatch.setattr(
        checkpoint,
        "load_batch_table",
        lambda index: pytest.fail(f"Rows were loaded for batch {index}"),
    )
    segmenter = _FakeSegmenter()

    result = _process_source(
        data_root.processed_v2 / "wikipedia/sections/region-latest.parquet",
        checkpoint=checkpoint,
        segmenter=segmenter,
        batch_size=1,
        stem="region-latest",
        project="wikipedia",
    )

    assert result.row_count == 1
    assert segmenter.calls == 0


def test_sentence_output_writes_validated_checkpoint_tables_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(
        0,
        [
            {"sentence_id": "sentence-1", "text": "First."},
            {"sentence_id": "sentence-2", "text": "Second."},
        ],
    )
    checkpoint.mark_complete(batch_count=1, row_count=2)
    monkeypatch.setattr(
        checkpoint,
        "load_batch",
        lambda index: pytest.fail(f"Python rows were materialized for batch {index}"),
    )

    output_path = tmp_path / "sentences.parquet"
    _write_output(output_path, checkpoint, batch_count=1)
    output = pq.read_table(output_path)

    assert output.schema.equals(sentence_schema(), check_metadata=True)
    assert output.column("sentence_id").to_pylist() == ["sentence-1", "sentence-2"]
    assert output.column("text").to_pylist() == ["First.", "Second."]


def test_checkpoint_batch_writer_skips_empty_tables(tmp_path: Path) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])
    checkpoint.write_batch(1, [])

    written_rows: list[int] = []

    class _Writer:
        def write_table(self, table: object) -> None:
            written_rows.append(table.num_rows)

    _write_checkpoint_batches(_Writer(), checkpoint, batch_count=2)

    assert written_rows == [1]
