"""Tests for the pipeline: extractor, processor, orchestrator, stats."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import polygon_id
from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    InMemoryWikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    FetchResult,
    InMemoryWikipediaClient,
    WikipediaArticle,
)
from osm_polygon_wikidata_only.io.pbf_reader import PolygonCandidate
from osm_polygon_wikidata_only.pipeline.extractor import candidate_to_polygon
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs, orchestrate
from osm_polygon_wikidata_only.pipeline.processor import (
    IncompleteEnrichmentError,
    PbfStem,
    process_pbf,
)
from osm_polygon_wikidata_only.pipeline.stats import StreamingStats


def _square_geom_json(lon: float, lat: float, d: float = 0.01) -> str:
    coords = [[[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]
    return json.dumps({"type": "Polygon", "coordinates": coords})


def _candidate(
    *,
    osm_type: str = "way",
    osm_id: int = 1,
    wikidata: str = "Q42",
    name: str = "Test",
    lon: float = 7.42,
    lat: float = 43.73,
    extra_tags: dict | None = None,
) -> PolygonCandidate:
    tags = {"wikidata": wikidata, "name": name, "landuse": "forest"}
    if extra_tags:
        tags.update(extra_tags)
    return (osm_type, osm_id, tags, _square_geom_json(lon, lat))


# --- candidate_to_polygon ----------------------------------------------


def test_candidate_to_polygon_builds_full_row() -> None:
    p = candidate_to_polygon(
        _candidate(wikidata="Q1", name="X"),
        source_pbf_stem="monaco-latest",
        region="monaco",
        source_pbf="monaco-latest.osm.pbf",
    )
    assert p is not None
    assert p.polygon_id == polygon_id("monaco-latest", "way", 1)
    assert p.wikidata == "Q1"
    assert p.name == "X"
    assert p.region == "monaco"
    assert p.osm_primary_tag == "landuse=forest"
    assert p.area_bucket  # non-empty
    assert p.bbox  # JSON list
    assert p.geometry
    assert '"coordinates"' in p.geometry


def test_candidate_to_polygon_skips_invalid_geometry() -> None:
    candidate: PolygonCandidate = ("way", 1, {"wikidata": "Q1"}, "{not json")
    assert (
        candidate_to_polygon(
            candidate,
            source_pbf_stem="x",
            region="x",
            source_pbf="x-latest.osm.pbf",
        )
        is None
    )


def test_candidate_to_polygon_skips_missing_wikidata() -> None:
    candidate: PolygonCandidate = ("way", 1, {"name": "x"}, _square_geom_json(7, 43))
    assert (
        candidate_to_polygon(
            candidate,
            source_pbf_stem="x",
            region="x",
            source_pbf="x-latest.osm.pbf",
        )
        is None
    )


# --- PbfStem -----------------------------------------------------------


def test_pbf_stem_parses_geofabrik_filename() -> None:
    s = PbfStem.from_path(Path("monaco-latest.osm.pbf"))
    assert s.stem == "monaco-latest"
    assert s.region == "monaco"


# --- process_pbf end-to-end with fake clients --------------------------


def _make_article(language: str, body: str) -> WikipediaArticle:
    return WikipediaArticle(
        language=language,
        site=f"{language}wiki",
        title="X",
        page_id=10,
        revision_id=100,
        revision_timestamp="2026-01-01T00:00:00Z",
        url=f"https://{language}.wikipedia.org/wiki/X",
        lead_text=body,
        extract=body,
        full_text=body,
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
        source_api="mediawiki_action_api",
        retrieved_at="2026-01-01T00:00:00Z",
    )


class _FakePbf:
    """Tiny stand-in for a PBF file: just emit pre-built candidates."""

    def __init__(self, path: Path, candidates: list[PolygonCandidate]) -> None:
        self.path = path
        self.candidates = candidates

    def collect(self) -> list[PolygonCandidate]:
        return list(self.candidates)


@pytest.fixture()
def tiny_pbf(tmp_path: Path) -> Path:
    """Create a placeholder PBF file (the contents are ignored in the test)."""
    p = tmp_path / "tiny-latest.osm.pbf"
    p.write_bytes(b"")
    return p


def test_process_pbf_writes_three_parquet_and_manifest(
    tiny_pbf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [
        _candidate(osm_id=1, wikidata="Q1", name="A"),
        _candidate(osm_id=2, wikidata="Q2", name="B"),
        _candidate(osm_id=3, wikidata="Q1", name="A2"),
    ]

    # Monkeypatch PBFReader to return our canned candidates.
    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = Path(pbf_path)

        @property
        def region_name(self) -> str:
            return "tiny"

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return list(candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)

    # Fake Wikidata + Wikipedia clients.
    wd = InMemoryWikidataClient(
        {
            "Q1": WikidataEntity(
                qid="Q1",
                sitelinks={"enwiki": "A", "frwiki": "A_fr"},
                labels={"en": "A label"},
                descriptions={"en": "A desc"},
            ),
            "Q2": WikidataEntity(
                qid="Q2",
                sitelinks={"enwiki": "B"},
                labels={"en": "B label"},
            ),
        }
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient

    wiki = InMemoryWikipediaClient(
        {
            ("enwiki", "A"): FetchResult("ok", _make_article("en", "en body for A")),
            ("frwiki", "A_fr"): FetchResult("ok", _make_article("fr", "fr body for A")),
            ("enwiki", "B"): FetchResult("ok", _make_article("en", "en body for B")),
        }
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()

    settings = Settings()
    result = process_pbf(
        tiny_pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=settings,
    )
    assert result.polygon_count == 3
    # 3 unique articles: en A, fr A_fr (both for Q1) and en B (for Q2).
    assert result.article_count == 3
    # 2 Q1 polygons x 2 articles + 1 Q2 polygon x 1 article = 5 links.
    assert result.link_count == 5
    assert result.polygons_path.exists()
    assert result.articles_path.exists()
    assert result.polygon_articles_path.exists()
    assert result.manifest_path.exists()
    assert set(result.stage_timings_s) == {
        "extract",
        "enrich",
        "build_rows",
        "write_parquet",
        "manifest",
    }
    assert all(seconds >= 0 for seconds in result.stage_timings_s.values())

    # Read back polygons parquet and check schema.
    table = pq.read_table(result.polygons_path)
    assert table.num_rows == 3
    assert "wikipedia_languages" in table.column_names

    # Manifest entry exists.
    text = result.manifest_path.read_text()
    assert "tiny-latest.osm.pbf" in text
    assert "polygon_count" in text


def test_process_pbf_dedups_repeated_qids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [
        _candidate(osm_id=1, wikidata="Q1", name="A"),
        _candidate(osm_id=2, wikidata="Q1", name="A"),
        _candidate(osm_id=3, wikidata="Q1", name="A"),
    ]

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = Path(pbf_path)

        @property
        def region_name(self) -> str:
            return "tiny"

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return list(candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)

    wd = InMemoryWikidataClient(
        {"Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "A"}, labels={"en": "A"})}
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient

    wiki = InMemoryWikipediaClient(
        {("enwiki", "A"): FetchResult("ok", _make_article("en", "body"))}
    )

    pbf = tmp_path / "tiny-latest.osm.pbf"
    pbf.write_bytes(b"")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    settings = Settings()
    import osm_polygon_wikidata_only.pipeline.processor as processor

    word_count_calls = 0
    original_count_words = processor.count_words

    def count_words_once(text: str) -> int:
        nonlocal word_count_calls
        word_count_calls += 1
        return original_count_words(text)

    monkeypatch.setattr(processor, "count_words", count_words_once)
    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=settings,
    )
    # One polygon row per candidate, but only one article row and
    # three link rows (one per polygon).
    assert result.polygon_count == 3
    assert result.article_count == 1
    assert result.link_count == 3
    assert word_count_calls == 1


def test_process_pbf_keeps_polygons_with_no_wikipedia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [
        _candidate(osm_id=1, wikidata="Q1", name="A"),
        _candidate(osm_id=2, wikidata="Q-NO-SITELINKS", name="B"),
    ]

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = Path(pbf_path)

        @property
        def region_name(self) -> str:
            return "tiny"

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return list(candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)

    wd = InMemoryWikidataClient(
        {
            "Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "A"}, labels={"en": "A"}),
            # Q-NO-SITELINKS is invalid so the linker will reject it.
        }
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient

    wiki = InMemoryWikipediaClient(
        {("enwiki", "A"): FetchResult("ok", _make_article("en", "body"))}
    )

    pbf = tmp_path / "tiny-latest.osm.pbf"
    pbf.write_bytes(b"")
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    settings = Settings()
    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=settings,
    )
    assert result.polygon_count == 2
    # First polygon has Wikipedia, second does not (invalid QID).
    table = pq.read_table(result.polygons_path)
    by_id = {row["polygon_id"]: row for row in table.to_pylist()}
    assert by_id["tiny-latest:way:1"]["has_wikipedia"] is True
    assert by_id["tiny-latest:way:2"]["has_wikipedia"] is False


def test_process_pbf_does_not_publish_when_an_expected_article_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return [_candidate(osm_id=1, wikidata="Q1", name="A")]

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)
    pbf = tmp_path / "tiny-latest.osm.pbf"
    pbf.write_bytes(b"")
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    wd = InMemoryWikidataClient({"Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "A"})})
    wiki = InMemoryWikipediaClient(
        {("enwiki", "A"): FetchResult("rate_limited", None, "retry later")}
    )

    with pytest.raises(IncompleteEnrichmentError, match="enwiki"):
        process_pbf(
            pbf,
            data_root=data_root,
            wikidata_client=wd,
            wikipedia_client=wiki,
            settings=Settings(),
        )

    assert not list(data_root.processed.rglob("*.parquet"))
    assert not (data_root.processed_manifests / "processed_pbfs.json").exists()


def test_process_pbf_publishes_empty_text_articles_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Option C: empty_text results become rows, not PBF-level failures."""
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [
        _candidate(osm_id=1, wikidata="Q1", name="A"),
    ]

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return list(candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)

    wd = InMemoryWikidataClient(
        {
            "Q1": WikidataEntity(
                qid="Q1",
                sitelinks={"enwiki": "A", "frwiki": "A_fr"},
                labels={"en": "A label"},
            ),
        }
    )

    empty_stub = _make_article("fr", "")
    wiki = InMemoryWikipediaClient(
        {
            ("enwiki", "A"): FetchResult("ok", _make_article("en", "en body")),
            (
                "frwiki",
                "A_fr",
            ): FetchResult(
                "empty_text",
                empty_stub,
                "extract and exact-revision parse were empty",
            ),
        }
    )

    pbf = tmp_path / "tiny-latest.osm.pbf"
    pbf.write_bytes(b"")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    # Must NOT raise: empty_text is data, not an infrastructure failure.
    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=Settings(),
    )
    assert result.polygon_count == 1
    assert result.article_count == 2  # en OK + fr empty_text both linked
    assert result.link_count == 2  # polygon -> both articles

    art_table = pq.read_table(result.articles_path)
    rows = {row["language"]: row for row in art_table.to_pylist()}
    assert rows["en"]["fetch_status"] == "ok"
    assert rows["en"]["full_text"] == "en body"
    assert rows["fr"]["fetch_status"] == "empty_text"
    assert rows["fr"]["full_text"] == ""
    assert rows["fr"]["page_id"] == 10
    assert rows["fr"]["revision_id"] == 100

    link_table = pq.read_table(result.polygon_articles_path)
    assert sorted(row["language"] for row in link_table.to_pylist()) == ["en", "fr"]


def test_process_pbf_still_raises_for_transient_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Option C must NOT swallow rate_limited / http_error / parse_error."""
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [_candidate(osm_id=1, wikidata="Q1", name="A")]

    class _StubReader:
        def __init__(self, pbf_path: Path) -> None:
            self.pbf_path = pbf_path

        def collect_polygon_candidates(self) -> list[PolygonCandidate]:
            return list(candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _StubReader)

    wd = InMemoryWikidataClient({"Q1": WikidataEntity(qid="Q1", sitelinks={"enwiki": "A"})})
    wiki = InMemoryWikipediaClient(
        {("enwiki", "A"): FetchResult("rate_limited", None, "503 Service Unavailable")}
    )

    pbf = tmp_path / "tiny-latest.osm.pbf"
    pbf.write_bytes(b"")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(IncompleteEnrichmentError, match="enwiki"):
        process_pbf(
            pbf,
            data_root=data_root,
            wikidata_client=wd,
            wikipedia_client=wiki,
            settings=Settings(),
        )
    assert not list(data_root.processed.rglob("*.parquet"))


# --- orchestrator ------------------------------------------------------


def test_collect_pbfs_expands_directories(tmp_path: Path) -> None:
    (tmp_path / "a-latest.osm.pbf").write_bytes(b"")
    (tmp_path / "b-latest.osm.pbf").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c-latest.osm.pbf").write_bytes(b"")
    pbfs = collect_pbfs([tmp_path, sub])
    names = sorted(p.name for p in pbfs)
    assert names == ["a-latest.osm.pbf", "b-latest.osm.pbf", "c-latest.osm.pbf"]


def test_orchestrate_submits_each_result_before_processing_next_pbf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.osm.pbf"
    second = tmp_path / "b.osm.pbf"
    first.write_bytes(b"")
    second.write_bytes(b"")
    events: list[str] = []

    def fake_process(path: Path, **_: object) -> object:
        events.append(f"process:{path.name}")
        return type("Result", (), {"manifest_entry": {"source_pbf": path.name}})()

    monkeypatch.setattr("osm_polygon_wikidata_only.pipeline.orchestrator.process_pbf", fake_process)
    data_root = DataRoot(tmp_path / "data")
    data_root.ensure()
    orchestrate(
        [first, second],
        data_root=data_root,
        settings=Settings(),
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        on_complete=lambda result: events.append(f"submit:{result.manifest_entry['source_pbf']}"),
    )
    assert events == [
        "process:a.osm.pbf",
        "submit:a.osm.pbf",
        "process:b.osm.pbf",
        "submit:b.osm.pbf",
    ]


# --- stats -------------------------------------------------------------


def test_streaming_stats_aggregates_correctly() -> None:
    stats = StreamingStats()
    poly = Polygon.make(
        source_pbf_stem="x",
        region="x",
        source_pbf="x.osm.pbf",
        osm_type="way",
        osm_id=1,
        wikidata="Q1",
        name="",
        tags="{}",
        tag_keys='["landuse"]',
        tag_count=1,
        osm_primary_tag="landuse=forest",
        centroid='{"type":"Point"}',
        lat=0,
        lon=0,
        bbox="[0,0,0,0]",
        area_m2=1000,
        area_km2=0.001,
        area_bucket="100m2-1k_m2",
        has_name=False,
        has_wikidata=True,
        extraction_version="0.1.0",
        extracted_at="2026-01-01T00:00:00Z",
    )
    stats.add_polygon(poly)
    final = stats.finalize()
    assert final.polygon_count == 1
    assert final.unique_wikidata_count == 1
    assert final.area_bucket_counts == {"100m2-1k_m2": 1}
    assert final.top_tag_keys == {"landuse": 1}


def test_candidate_to_polygon_includes_geometry() -> None:
    row = candidate_to_polygon(
        (
            "way",
            1,
            {"wikidata": "Q1", "name": "x"},
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}',
        ),
        source_pbf_stem="test-latest",
        region="test",
        source_pbf="test-latest.osm.pbf",
        extracted_at="2026-01-01T00:00:00Z",
    )

    assert row is not None
    assert '"type":"Polygon"' in row.geometry or '"type": "Polygon"' in row.geometry
    assert "coordinates" in row.geometry
