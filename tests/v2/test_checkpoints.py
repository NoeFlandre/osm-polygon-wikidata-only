from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult, WikipediaArticle
from osm_polygon_wikidata_only.v2 import extractor
from osm_polygon_wikidata_only.v2.checkpoints import (
    ExtractionCheckpoint,
    RegionFetchCheckpoint,
)
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf, V2PbfStem, candidate_to_v2_row
from osm_polygon_wikidata_only.v2.reuse import merge_v2_region

_SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'


def _candidate(number: int) -> tuple[str, int, dict[str, str], str]:
    return "way", number, {"wikipedia": f"en:Page {number}"}, _SQUARE


def test_extraction_checkpoint_resumes_after_partial_pbf_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbf = tmp_path / "region-latest.osm.pbf"
    pbf.write_bytes(b"stable source")
    calls = 0

    class Reader:
        def __init__(self, _path: Path, *, include_wikipedia_tagged: bool) -> None:
            assert include_wikipedia_tagged

        def iter_polygon_candidates(self, callback) -> None:
            nonlocal calls
            calls += 1
            callback(_candidate(1))
            if calls == 1:
                raise RuntimeError("interrupted PBF")
            callback(_candidate(2))

    monkeypatch.setattr(extractor, "PBFReader", Reader)
    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="interrupted PBF"):
        extractor.extract_v2_pbf(
            pbf,
            settings=type("Settings", (), {"limit": None})(),
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=1,
        )

    resumed = extractor.extract_v2_pbf(
        pbf,
        settings=type("Settings", (), {"limit": None})(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
    )
    assert [row["osm_id"] for row in resumed.polygons] == [1, 2]
    assert calls == 2


def test_completed_extraction_checkpoint_skips_pbf_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pbf = tmp_path / "region-latest.osm.pbf"
    pbf.write_bytes(b"stable source")
    checkpoint_dir = tmp_path / "checkpoints"
    settings = type("Settings", (), {"limit": None})()

    class Reader:
        def __init__(self, _path: Path, *, include_wikipedia_tagged: bool) -> None:
            assert include_wikipedia_tagged

        def iter_polygon_candidates(self, callback) -> None:
            callback(_candidate(1))

    monkeypatch.setattr(extractor, "PBFReader", Reader)
    first = extractor.extract_v2_pbf(
        pbf,
        settings=settings,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
    )

    class UnexpectedReader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("a completed extraction must not rescan the PBF")

    monkeypatch.setattr(extractor, "PBFReader", UnexpectedReader)
    second = extractor.extract_v2_pbf(
        pbf,
        settings=settings,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
    )
    assert second.polygons == first.polygons


def test_extraction_checkpoint_is_invalidated_when_source_changes(tmp_path: Path) -> None:
    pbf = tmp_path / "region-latest.osm.pbf"
    pbf.write_bytes(b"source-a")
    checkpoint = ExtractionCheckpoint(tmp_path / "checkpoints", pbf)
    checkpoint.append(
        [
            candidate_to_v2_row(
                _candidate(1),
                source_pbf_stem="region-latest",
                region="region",
                source_pbf=pbf.name,
            )
        ]
    )
    pbf.write_bytes(b"source-b")
    changed = ExtractionCheckpoint(tmp_path / "checkpoints", pbf)
    assert changed.load_rows() == []
    assert not changed.complete


def test_complete_extraction_checkpoint_is_invalidated_when_chunk_is_missing(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "region-latest.osm.pbf"
    pbf.write_bytes(b"stable source")
    checkpoint = ExtractionCheckpoint(tmp_path / "checkpoints", pbf)
    row = candidate_to_v2_row(
        _candidate(1),
        source_pbf_stem="region-latest",
        region="region",
        source_pbf=pbf.name,
    )
    assert row is not None
    checkpoint.append([row])
    checkpoint.mark_complete()
    next(iter(checkpoint.root.glob("chunk-*.parquet"))).unlink()

    recovered = ExtractionCheckpoint(tmp_path / "checkpoints", pbf)
    assert recovered.load_rows() == []
    assert not recovered.complete


def test_fetch_checkpoint_discards_state_with_invalid_metadata(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = RegionFetchCheckpoint(root, "region-latest", input_fingerprint="fingerprint-a")
    checkpoint.save_sections(
        "document-1",
        [{"document_id": "document-1", "section_id": "section-1"}],
    )
    checkpoint.metadata_path.write_text("not json", encoding="utf-8")

    recovered = RegionFetchCheckpoint(root, "region-latest", input_fingerprint="fingerprint-a")
    assert recovered.load_sections() == []
    assert not recovered.has_work


def test_fetch_checkpoint_records_empty_section_results(tmp_path: Path) -> None:
    checkpoint = RegionFetchCheckpoint(tmp_path / "checkpoints", "region-latest")
    checkpoint.save_sections("document-1", [])
    assert checkpoint.load_sections() == []
    assert checkpoint.section_document_ids() == {"document-1"}


def test_fetch_checkpoint_invalid_section_payload_is_not_marked_complete(tmp_path: Path) -> None:
    checkpoint = RegionFetchCheckpoint(tmp_path / "checkpoints", "region-latest")
    checkpoint.save_sections("document-1", [])
    section_path = next(checkpoint.sections_root.glob("*.json"))
    section_path.write_text(
        '{"document_id":"document-1","rows":[{"document_id":"other"}]}',
        encoding="utf-8",
    )
    assert checkpoint.load_sections() == []
    assert checkpoint.section_document_ids() == set()


def test_fetch_checkpoint_is_invalidated_when_full_text_mode_changes(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = RegionFetchCheckpoint(root, "region-latest", fetch_full_text=True)
    checkpoint.save_sections("document-1", [])

    changed = RegionFetchCheckpoint(root, "region-latest", fetch_full_text=False)
    assert changed.load_sections() == []
    assert changed.section_document_ids() == set()


def _article(title: str) -> WikipediaArticle:
    return WikipediaArticle(
        language="en",
        site="enwiki",
        title=title,
        page_id=1,
        revision_id=2,
        revision_timestamp="2020-01-01T00:00:00Z",
        url="https://en.wikipedia.org/wiki/Page",
        lead_text="lead",
        extract="extract",
        full_text="full text",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA",
        attribution="",
        source_api="test",
        retrieved_at="2020-01-01T00:00:00Z",
    )


def test_region_fetch_checkpoint_reuses_article_after_section_failure(
    tmp_path: Path,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    row = candidate_to_v2_row(
        _candidate(1),
        source_pbf_stem="region-latest",
        region="region",
        source_pbf=pbf.name,
    )
    assert row is not None
    extracted = V2ExtractedPbf(
        V2PbfStem(pbf, "region-latest", "region"),
        (row,),
        0.0,
    )

    class Index:
        is_ready = False

        def by_title(self, _language: str, _title: str) -> tuple[dict, ...]:
            return ()

        def wait_until_ready(self) -> None:
            return None

    class Wikipedia:
        calls = 0

        def fetch_article(self, *_args: object, **_kwargs: object) -> FetchResult:
            self.calls += 1
            return FetchResult("ok", _article("Page 1"))

    class Sections:
        calls = 0
        fail = True

        def parse_html(self, _project: str, _language: str, _revision_id: int) -> str:
            self.calls += 1
            if self.fail:
                self.fail = False
                raise RuntimeError("section interruption")
            return "<p>section text</p>"

    client = Wikipedia()
    sections = Sections()
    with pytest.raises(RuntimeError, match="section interruption"):
        merge_v2_region(
            root,
            extracted,
            index=Index(),
            wikipedia_client=client,
            section_client=sections,
            wait_for_index=False,
            checkpoint_dir=root.v2_cache / "checkpoints",
        )

    result = merge_v2_region(
        root,
        extracted,
        index=Index(),
        wikipedia_client=client,
        section_client=sections,
        wait_for_index=False,
        checkpoint_dir=root.v2_cache / "checkpoints",
    )
    assert result
    assert client.calls == 1
    assert sections.calls == 2
    assert RegionFetchCheckpoint(root.v2_cache / "checkpoints", "region-latest").has_work
