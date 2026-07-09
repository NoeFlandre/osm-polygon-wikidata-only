"""End-to-end smoke test that drives the new pipeline with mocked
HTTP. Exercises the extractor, both enrichment clients, the
processor, the manifest writer, and the HF stub upload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import (
    InMemoryWikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
    FetchResult,
    InMemoryWikipediaClient,
    WikipediaArticle,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_ARTICLES_DIR,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
)
from osm_polygon_wikidata_only.hf.uploader import (
    StubHfHub,
    upload_card,
    upload_manifest,
    upload_parquet,
)
from osm_polygon_wikidata_only.io.pbf_reader import PolygonCandidate
from osm_polygon_wikidata_only.pipeline.processor import process_pbf


def _square_geom_json(lon: float, lat: float, d: float = 0.01) -> str:
    coords = [
        [
            [lon, lat],
            [lon + d, lat],
            [lon + d, lat + d],
            [lon, lat + d],
            [lon, lat],
        ]
    ]
    return json.dumps({"type": "Polygon", "coordinates": coords})


def _candidate(
    *, osm_id: int, qid: str, name: str, lon: float = 7.42, lat: float = 43.73
) -> PolygonCandidate:
    return (
        "way",
        osm_id,
        {"wikidata": qid, "name": name, "boundary": "national_park"},
        _square_geom_json(lon, lat),
    )


def _wiki_article(lang: str, body: str) -> FetchResult:
    return FetchResult(
        "ok",
        WikipediaArticle(
            language=lang,
            site=f"{lang}wiki",
            title="X",
            page_id=10,
            revision_id=100,
            revision_timestamp="2026-01-01T00:00:00Z",
            url="https://wikipedia.org",
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
        ),
    )


def test_end_to_end_pbf_to_parquet_to_manifest_to_hf_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osm_polygon_wikidata_only.io import pbf_reader as pbf_reader_mod

    candidates = [
        _candidate(osm_id=1, qid="Q1", name="A", lon=7.41, lat=43.72),
        _candidate(osm_id=2, qid="Q2", name="B", lon=7.42, lat=43.73),
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
            "Q1": WikidataEntity(
                qid="Q1",
                sitelinks={"enwiki": "A", "frwiki": "A_fr"},
                labels={"en": "A"},
                descriptions={"en": "first"},
            ),
            "Q2": WikidataEntity(qid="Q2", sitelinks={"enwiki": "B"}, labels={"en": "B"}),
        }
    )
    wiki = InMemoryWikipediaClient(
        {
            ("enwiki", "A"): _wiki_article("en", "en body A"),
            ("frwiki", "A_fr"): _wiki_article("fr", "fr body A"),
            ("enwiki", "B"): _wiki_article("en", "en body B"),
        }
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

    # All three parquet files exist with rows.
    poly_t = pq.read_table(result.polygons_path)
    art_t = pq.read_table(result.articles_path)
    link_t = pq.read_table(result.polygon_articles_path)
    assert poly_t.num_rows == 2
    assert art_t.num_rows == 3  # en A, fr A_fr, en B
    assert link_t.num_rows == 3  # 2 articles for Q1 + 1 article for Q2

    # Manifest has the entry.
    text = result.manifest_path.read_text()
    assert "tiny-latest.osm.pbf" in text

    # Render and "upload" the card + per-file parquet + manifest via stub.
    stub = StubHfHub()
    card_md = render_dataset_card(
        repo_id=settings.repo_id,
        stats={"polygon_count": 2, "article_count": 3, "unique_wikidata_count": 2},
        polygon_columns=[f.name for f in poly_t.schema],
        polygon_descriptions={c: c for c in poly_t.schema.names},
        article_columns=[f.name for f in art_t.schema],
        article_descriptions={c: c for c in art_t.schema.names},
        link_columns=[f.name for f in link_t.schema],
        link_descriptions={c: c for c in link_t.schema.names},
    )
    upload_card(settings.repo_id, card_md, hub=stub)
    for path, sub in [
        (result.polygons_path, REMOTE_POLYGONS_DIR),
        (result.articles_path, REMOTE_ARTICLES_DIR),
        (result.polygon_articles_path, REMOTE_LINKS_DIR),
    ]:
        upload_parquet(
            settings.repo_id,
            path,
            path_in_repo=f"{sub}/{path.stem}.parquet",
            hub=stub,
        )
    upload_manifest(
        settings.repo_id, result.manifest_path, path_in_repo=REMOTE_MANIFEST_FILE, hub=stub
    )

    paths = [u["path_in_repo"] for u in stub.uploads]
    assert "README.md" in paths
    assert "polygons/tiny-latest.parquet" in paths
    assert "articles/tiny-latest.parquet" in paths
    assert "polygon_articles/tiny-latest.parquet" in paths
    assert REMOTE_MANIFEST_FILE in paths
