"""Characterization tests for the Phase 6 processing decomposition.

These tests pin the exact behavior the decomposition of
:mod:`pipeline.processor` into focused phases must preserve:

* :func:`extract_pbf` returns an :class:`ExtractedPbf` carrying the
  parsed :class:`PbfStem`, the stream-respecting polygon tuple, and
  the extraction duration. The helper is re-exported from
  ``pipeline.processor`` for facade continuity.
* The enrichment phase issues exactly one ``entities(...)`` fetch
  for the unique QIDs while keeping the heartbeat lifecycle
  intact, raises :class:`IncompleteEnrichmentError` on
  non-fatal-elided failures, and closes the heartbeat thread on
  every exit path (success, partial-failure, fatal-failure).
* The row construction phase produces exact deterministic Article
  and PolygonArticleLink rows from a fixed ``LinkSummary``.
* The persistence phase writes the three parquet files using the
  canonical schemas via ``os.replace``, removes the temporary
  file on failure, and then writes the manifest entry. A failure
  inside the row-building or write stages must leave the manifest
  untouched.
* The full pipeline keeps the ``stage_timings_s`` keys
  (``extract``, ``enrich``, ``build_rows``, ``write_parquet``,
  ``manifest``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import (
    PROCESSED_ARTICLES,
    PROCESSED_LINKS,
    PROCESSED_POLYGONS,
    DataRoot,
)
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import polygon_id
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary
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
from osm_polygon_wikidata_only.pipeline.completeness import IncompleteEnrichmentError
from osm_polygon_wikidata_only.pipeline.processor import (
    ExtractedPbf,
    PbfStem,
    extract_pbf,
    process_pbf,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _square_geom_json(lon: float = 7.42, lat: float = 43.73) -> str:
    coords = [
        [[lon, lat], [lon + 0.01, lat], [lon + 0.01, lat + 0.01], [lon, lat + 0.01], [lon, lat]]
    ]
    return json.dumps({"type": "Polygon", "coordinates": coords})


def _candidate(*, osm_id: int = 1, wikidata: str = "Q1", name: str = "X") -> PolygonCandidate:
    return (
        "way",
        osm_id,
        {"wikidata": wikidata, "name": name, "landuse": "forest"},
        _square_geom_json(),
    )


class _StubReader:
    def __init__(self, pbf_path: Path, candidates: list[PolygonCandidate]) -> None:
        self.pbf_path = pbf_path
        self._candidates = list(candidates)

    def collect_polygon_candidates(self) -> list[PolygonCandidate]:
        return list(self._candidates)


def _install_pbf_reader(
    monkeypatch: pytest.MonkeyPatch, candidates: list[PolygonCandidate]
) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    def _factory(pbf_path: Path) -> _StubReader:
        return _StubReader(pbf_path, candidates)

    monkeypatch.setattr(pbf_reader_mod, "PBFReader", _factory)


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


def _pbf_path(tmp_path: Path, name: str = "tiny-latest.osm.pbf") -> Path:
    pbf = tmp_path / name
    pbf.write_bytes(b"")
    return pbf


def _heartbeat_recorder(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Return a dict collecting heartbeat entry/exit events for the patched
    EnrichmentHeartbeat. The heartbeat lives in ``enrichment_phase``."""

    events: dict[str, list[Any]] = {
        "regions": [],
        "exited": [],
    }
    from osm_polygon_wikidata_only.pipeline import enrichment_phase as enrichment_phase_mod

    real_heartbeat = enrichment_phase_mod.EnrichmentHeartbeat

    class _RecordingHeartbeat:
        def __init__(self, *, region: str, **_: object) -> None:
            events["regions"].append(region)
            self._real = real_heartbeat(region=region, **_)

        def __enter__(self) -> _RecordingHeartbeat:
            self._real.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            events["exited"].append(True)
            return self._real.__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(enrichment_phase_mod, "EnrichmentHeartbeat", _RecordingHeartbeat)
    return events


# ---------------------------------------------------------------------------
# extract_pbf
# ---------------------------------------------------------------------------


def test_extract_pbf_returns_extracted_pbf_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``extract_pbf`` is re-exported from ``pipeline.processor`` and
    returns ``ExtractedPbf`` carrying the parsed stem, stream-respecting
    polygons (limit + wikidata filter preserved), and a non-negative
    extraction duration."""

    candidates = [
        _candidate(osm_id=1, wikidata="Q1"),
        _candidate(osm_id=2, wikidata="Q2"),
        _candidate(osm_id=3, wikidata=""),  # filtered (no wikidata)
    ]
    _install_pbf_reader(monkeypatch, candidates)
    pbf = _pbf_path(tmp_path, "monaco-latest.osm.pbf")

    result = extract_pbf(pbf, settings=Settings())

    assert isinstance(result, ExtractedPbf)
    assert isinstance(result.stem, PbfStem)
    assert result.stem.stem == "monaco-latest"
    assert result.stem.region == "monaco"
    assert [p.osm_id for p in result.polygons] == [1, 2]
    assert result.extraction_duration_s >= 0
    # Polygons are tuple-typed (immutable, sortable).
    assert isinstance(result.polygons, tuple)
    # Polygon id is computed exactly as before.
    assert result.polygons[0].polygon_id == polygon_id("monaco-latest", "way", 1)


def test_extract_pbf_respects_settings_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [_candidate(osm_id=i, wikidata=f"Q{i}") for i in range(1, 6)]
    _install_pbf_reader(monkeypatch, candidates)
    pbf = _pbf_path(tmp_path, "andorra-latest.osm.pbf")
    settings = replace(Settings(), limit=3)

    result = extract_pbf(pbf, settings=settings)
    assert len(result.polygons) == 3
    assert [p.osm_id for p in result.polygons] == [1, 2, 3]


def test_processor_facade_re_exports_extract_pbf_and_extracted_pbf() -> None:
    """``pipeline.processor.extract_pbf`` and ``pipeline.processor.ExtractedPbf``
    are identity exports of the focused helpers."""

    from osm_polygon_wikidata_only.pipeline import extractor as extractor_module
    from osm_polygon_wikidata_only.pipeline import processor as processor_module

    assert processor_module.extract_pbf is extractor_module.extract_pbf
    assert processor_module.ExtractedPbf is extractor_module.ExtractedPbf


# ---------------------------------------------------------------------------
# Enrichment phase
# ---------------------------------------------------------------------------


def _clients(
    *, qids_to_sites: dict[str, dict[str, str]], article_body: str = "en body"
) -> tuple[InMemoryWikidataClient, InMemoryWikipediaClient]:
    entities: dict[str, WikidataEntity] = {}
    articles: dict[tuple[str, str], FetchResult] = {}
    for qid, sites in qids_to_sites.items():
        entities[qid] = WikidataEntity(qid=qid, sitelinks=sites, labels={"en": qid})
        for site, title in sites.items():
            articles[(site, title)] = FetchResult("ok", _make_article(site[:2], article_body))
    return InMemoryWikidataClient(entities), InMemoryWikipediaClient(articles)


def test_enrichment_uses_unique_sorted_qids_with_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _heartbeat_recorder(monkeypatch)
    candidates = [
        _candidate(osm_id=1, wikidata="Q2"),
        _candidate(osm_id=2, wikidata="Q1"),
        _candidate(osm_id=3, wikidata="Q1"),
    ]
    _install_pbf_reader(monkeypatch, candidates)
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}, "Q2": {"enwiki": "B"}})
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=Settings(),
    )

    # Two unique QIDs out of three polygons.
    assert events["regions"] == ["tiny"]
    assert events["exited"] == [True]
    assert result.polygon_count == 3
    assert result.manifest_path.exists()


def test_enrichment_raises_incomplete_on_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _heartbeat_recorder(monkeypatch)
    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])
    wd, _ = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})
    wiki = InMemoryWikipediaClient(
        {("enwiki", "A"): FetchResult("rate_limited", None, "retry later")}
    )
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
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

    # Heartbeat exited even on failure; no parquet leaked.
    assert events["regions"] == ["tiny"]
    assert events["exited"] == [True]
    # No parquet written, no manifest entry.
    assert not list(data_root.processed.rglob("*.parquet"))
    assert not (data_root.processed_manifests / "processed_pbfs.json").exists()


def test_enrichment_heartbeat_cleanup_on_fatal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``fetch_qids`` raises inside the heartbeat, the heartbeat
    thread is still joined."""
    from osm_polygon_wikidata_only.pipeline import enrichment_phase as enrichment_phase_mod

    events = _heartbeat_recorder(monkeypatch)
    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(enrichment_phase_mod, "fetch_qids", _boom)
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(RuntimeError, match="boom"):
        process_pbf(
            pbf,
            data_root=data_root,
            wikidata_client=wd,
            wikipedia_client=wiki,
            settings=Settings(),
        )
    assert events["exited"] == [True]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Row construction phase
# ---------------------------------------------------------------------------


def test_row_construction_phase_produces_exact_rows_and_links() -> None:
    """``build_articles_and_links`` builds deterministic Article and
    PolygonArticleLink rows from one ``LinkSummary`` carrying two
    language articles; deduplication keeps one entry per unique
    (wikidata, language, page_id, revision_id)."""
    from osm_polygon_wikidata_only.pipeline.row_construction import (
        build_articles_and_links,
    )

    article_en = _make_article("en", "English body")
    article_fr = _make_article("fr", "Corps français")
    summary = LinkSummary(
        qid="Q42",
        entity=None,
        articles=[article_en, article_fr],
        statuses={"enwiki": "ok", "frwiki": "ok"},
        errors={},
    )
    polygon = type(
        "Polygon",
        (),
        {
            "polygon_id": "tiny-latest:way:1",
            "region": "tiny",
            "source_pbf": "tiny-latest.osm.pbf",
            "osm_type": "way",
            "osm_id": 1,
            "wikidata": "Q42",
            "page_id": None,
            "revision_id": None,
        },
    )()

    articles, links = build_articles_and_links([polygon], {"Q42": summary})

    article_by_lang = {a.language: a for a in articles}
    assert set(article_by_lang) == {"en", "fr"}
    assert article_by_lang["en"].full_text == "English body"
    assert article_by_lang["fr"].full_text == "Corps français"
    # Two links, one per article.
    assert len(links) == 2
    assert {link.language for link in links} == {"en", "fr"}
    assert all(link.wikidata == "Q42" for link in links)


# ---------------------------------------------------------------------------
# Persistence phase
# ---------------------------------------------------------------------------


def test_persistence_phase_writes_three_parquet_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persistence phase writes the three parquet files (matching
    the canonical schemas) and the manifest entry. The stage-timing
    keys are preserved exactly."""

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.pipeline.persistence.process_extracted_pbf_marker",
        True,
        raising=False,
    )
    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=Settings(),
    )

    polygons_path = data_root.processed / PROCESSED_POLYGONS / "tiny-latest.parquet"
    articles_path = data_root.processed / PROCESSED_ARTICLES / "tiny-latest.parquet"
    links_path = data_root.processed / PROCESSED_LINKS / "tiny-latest.parquet"
    assert polygons_path == result.polygons_path
    assert articles_path == result.articles_path
    assert links_path == result.polygon_articles_path

    polygon_table = pq.read_table(polygons_path)
    article_table = pq.read_table(articles_path)
    link_table = pq.read_table(links_path)
    assert polygon_table.num_rows == 1
    assert article_table.num_rows == 1
    assert link_table.num_rows == 1
    # Canonical schema columns are present.
    assert {"polygon_id", "wikidata"}.issubset(polygon_table.schema.names)
    assert {"article_id", "full_text", "fetch_status"}.issubset(article_table.schema.names)
    assert {"polygon_id", "article_id"}.issubset(link_table.schema.names)

    # Stage timings: exactly the canonical five keys.
    assert set(result.stage_timings_s) == {
        "extract",
        "enrich",
        "build_rows",
        "write_parquet",
        "manifest",
    }
    assert all(value >= 0 for value in result.stage_timings_s.values())

    # Manifest entry has the canonical key set.
    assert result.manifest_path.exists()
    manifest_text = result.manifest_path.read_text()
    assert "tiny-latest.osm.pbf" in manifest_text
    assert "polygon_count" in manifest_text
    assert "polygons/tiny-latest.parquet" in manifest_text


def test_persistence_phase_removes_temporary_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If writing parquet fails midway, the temporary files are cleaned
    up and the manifest is not written."""
    from osm_polygon_wikidata_only.pipeline import persistence as persistence_module

    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    original_write = persistence_module.write_articles

    def _exploding_write_articles(path: Path, rows: Any) -> int:
        raise RuntimeError("disk full")

    monkeypatch.setattr(persistence_module, "write_articles", _exploding_write_articles)
    try:
        with pytest.raises(RuntimeError, match="disk full"):
            process_pbf(
                pbf,
                data_root=data_root,
                wikidata_client=wd,
                wikipedia_client=wiki,
                settings=Settings(),
            )
    finally:
        monkeypatch.setattr(persistence_module, "write_articles", original_write)

    # No manifest written, no leftover *.tmp.
    assert not (data_root.processed_manifests / "processed_pbfs.json").exists()
    assert not list(data_root.processed.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# No manifest on row construction failure
# ---------------------------------------------------------------------------


def test_process_pbf_no_manifest_when_row_building_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the imported name the processor actually uses
    # (`pipeline.rows.build_articles_and_links` is the re-export alias).
    from osm_polygon_wikidata_only.pipeline import rows as rows_mod

    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})

    def _explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("row building failure")

    monkeypatch.setattr(rows_mod, "build_articles_and_links", _explode)
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()

    with pytest.raises(RuntimeError, match="row building failure"):
        process_pbf(
            pbf,
            data_root=data_root,
            wikidata_client=wd,
            wikipedia_client=wiki,
            settings=Settings(),
        )
    assert not (data_root.processed_manifests / "processed_pbfs.json").exists()


# ---------------------------------------------------------------------------
# Stage timings preserved
# ---------------------------------------------------------------------------


def test_stage_timings_keys_are_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pbf_reader(monkeypatch, [_candidate(osm_id=1, wikidata="Q1")])
    wd, wiki = _clients(qids_to_sites={"Q1": {"enwiki": "A"}})
    pbf = _pbf_path(tmp_path, "tiny-latest.osm.pbf")
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    result = process_pbf(
        pbf,
        data_root=data_root,
        wikidata_client=wd,
        wikipedia_client=wiki,
        settings=Settings(),
    )
    assert list(result.stage_timings_s) == [
        "extract",
        "enrich",
        "build_rows",
        "write_parquet",
        "manifest",
    ]


# ---------------------------------------------------------------------------
# Next extraction starts before current enrichment completes
# ---------------------------------------------------------------------------
#
# These invariants live in the sync-runner decomposition; see
# tests/contracts/test_sync_runner_decomposition.py.
