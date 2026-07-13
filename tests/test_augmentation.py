"""Tests for additive Wikimedia augmentation sidecars."""

from __future__ import annotations

import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.models import (
    Document,
    document_from_article_row,
    document_id,
)
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    augmentation_is_current,
    sidecar_paths,
)
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
)
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.wikimedia import (
    discover_wikivoyage_sitelinks,
    normalize_facts,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.io.cache import JsonFileCache


class ThrottleThenSuccessSession:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, request: object) -> tuple[bytes, str]:
        self.calls += 1
        if self.calls == 1:
            headers = Message()
            headers["Retry-After"] = "0"
            raise urllib.error.HTTPError(
                "https://en.wikipedia.org/w/api.php", 429, "limited", headers, None
            )
        return b'{"parse":{"text":"ok"}}', ""


def article_row() -> dict[str, object]:
    return {
        "article_id": "Q1:en:10:20",
        "wikidata": "Q1",
        "language": "en",
        "site": "enwiki",
        "title": "Andorra",
        "url": "https://en.wikipedia.org/wiki/Andorra",
        "page_id": 10,
        "revision_id": 20,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-02T00:00:00Z",
        "full_text": "Lead. History text.",
        "full_text_format": "plain_text",
        "article_length_chars": 19,
        "article_length_words": 3,
        "article_length_tokens_estimate": 4,
        "license": "CC BY-SA 4.0",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "abc",
    }


def test_sidecar_columns_are_explicit_and_joinable() -> None:
    assert DOCUMENT_COLUMNS[:4] == ("document_id", "article_id", "wikidata", "project")
    assert SECTION_COLUMNS[:4] == ("section_id", "document_id", "article_id", "wikidata")
    assert FACT_COLUMNS[:4] == ("fact_id", "wikidata", "property_id", "property_label_en")
    assert "property_labels" in FACT_COLUMNS
    assert "value_labels" in FACT_COLUMNS


def test_augmentation_transport_retries_after_wikimedia_429(tmp_path) -> None:
    client = AugmentationWikimediaClient(
        Settings(request_max_retries=2, request_base_delay_s=0),
        JsonFileCache(tmp_path),
        environ={},
    )
    session = ThrottleThenSuccessSession()
    client._session = session

    result = client.get_json("https://retry.example.org/w/api.php?action=parse", key="retry-test")

    assert result == {"parse": {"text": "ok"}}
    assert session.calls == 2


def test_existing_article_becomes_wikipedia_document_without_data_loss() -> None:
    row = article_row()
    document = document_from_article_row(row)

    assert document.document_id == document_id("Q1", "wikipedia", "en", 10, 20)
    assert document.article_id == row["article_id"]
    assert document.project == "wikipedia"
    assert document.full_text == row["full_text"]
    assert document.revision_id == row["revision_id"]
    assert document.license == row["license"]


def test_section_parser_preserves_lead_and_nested_hierarchy() -> None:
    document = document_from_article_row(article_row())
    html = """
      <div class="mw-parser-output"><p>Lead text.</p>
      <h2><span id="History">History</span></h2><p>History text.</p>
      <h3><span id="Modern">Modern era</span></h3><p>Modern text.</p>
      <h2><span id="References">References</span></h2><ol><li>Ignored citation</li></ol>
      </div>
    """

    sections = parse_sections(document, html)

    assert [(section.heading, section.level, section.text) for section in sections] == [
        ("", 0, "Lead text."),
        ("History", 2, "History text."),
        ("Modern era", 3, "Modern text."),
    ]
    assert json.loads(sections[2].section_path) == ["History", "Modern era"]
    assert sections[2].parent_section_id == sections[1].section_id


def test_wikivoyage_discovery_keeps_every_language() -> None:
    entity = {
        "sitelinks": {
            "enwikivoyage": {"title": "Andorra"},
            "frwikivoyage": {"title": "Andorre"},
            "enwiki": {"title": "Andorra"},
        }
    }

    assert discover_wikivoyage_sitelinks(entity) == [
        ("en", "enwikivoyage", "Andorra"),
        ("fr", "frwikivoyage", "Andorre"),
    ]


def test_fact_normalization_always_keeps_english_and_extra_labels() -> None:
    entity = {
        "id": "Q1",
        "claims": {
            "P31": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datatype": "wikibase-item",
                        "datavalue": {"value": {"id": "Q6256"}},
                    },
                    "qualifiers": {},
                    "references": [],
                }
            ]
        },
    }
    labels = {
        "P31": {"en": "instance of", "fr": "nature de l'élément"},
        "Q6256": {"en": "country", "ca": "país"},
    }

    facts = normalize_facts(entity, labels)

    assert len(facts) == 1
    fact = facts[0]
    assert fact.property_label_en == "instance of"
    assert json.loads(fact.property_labels) == labels["P31"]
    assert fact.value_label_en == "country"
    assert json.loads(fact.value_labels) == labels["Q6256"]


def test_sidecar_paths_match_the_approved_additive_layout(tmp_path) -> None:
    paths = sidecar_paths(DataRoot(tmp_path), "andorra-latest")
    assert [str(path.relative_to(tmp_path / "processed")) for path in paths] == [
        "wikipedia/documents/andorra-latest.parquet",
        "wikipedia/sections/andorra-latest.parquet",
        "wikivoyage/documents/andorra-latest.parquet",
        "wikivoyage/sections/andorra-latest.parquet",
        "wikidata/facts/andorra-latest.parquet",
    ]


class FakeAugmentationClient:
    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]:
        if props == "labels":
            return {qid: {"id": qid, "labels": {"en": {"value": f"English {qid}"}}} for qid in qids}
        return {
            "Q1": {
                "id": "Q1",
                "sitelinks": {"frwikivoyage": {"title": "Andorre"}},
                "claims": {
                    "P31": [
                        {
                            "rank": "normal",
                            "mainsnak": {
                                "snaktype": "value",
                                "datatype": "wikibase-item",
                                "datavalue": {"value": {"id": "Q6256"}},
                            },
                        }
                    ]
                },
            }
        }

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        return '<div class="mw-parser-output"><p>Lead.</p><h2>History</h2><p>Past.</p></div>'

    def wikivoyage_document(self, qid: str, language: str, site: str, title: str) -> Document:
        row = article_row()
        row.update(
            article_id="",
            language=language,
            site=site,
            title=title,
            page_id=30,
            revision_id=40,
            full_text="Travel text.",
        )
        source = document_from_article_row(row)
        values = source.to_dict()
        values.update(
            document_id=document_id(qid, "wikivoyage", language, 30, 40),
            project="wikivoyage",
        )
        return Document(**values)


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_augment_region_writes_five_sidecars_without_modifying_core(tmp_path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    articles_path = data_root.processed_articles / "andorra-latest.parquet"
    polygons_path = data_root.processed_polygons / "andorra-latest.parquet"
    pq.write_table(pa.Table.from_pylist([article_row()]), articles_path)
    pq.write_table(pa.Table.from_pylist([{"wikidata": "Q1"}]), polygons_path)
    core_before = (_bytes(articles_path), _bytes(polygons_path))

    result = augment_region(data_root, "andorra-latest", FakeAugmentationClient())

    assert all(path.exists() for path in sidecar_paths(data_root, "andorra-latest"))
    assert (_bytes(articles_path), _bytes(polygons_path)) == core_before
    assert result.counts == {
        "wikipedia_documents": 1,
        "wikipedia_sections": 2,
        "wikivoyage_documents": 1,
        "wikivoyage_sections": 2,
        "wikidata_facts": 1,
    }
    assert (
        pq.read_table(result.wikidata_facts_path).to_pylist()[0]["value_label_en"]
        == "English Q6256"
    )

    assert augmentation_is_current(data_root, "andorra-latest") is True
    pq.write_table(pa.Table.from_pylist([{"wikidata": "Q2"}]), polygons_path)
    assert augmentation_is_current(data_root, "andorra-latest") is False


def test_augment_region_reports_final_phase_progress(tmp_path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    pq.write_table(
        pa.Table.from_pylist([article_row()]),
        data_root.processed_articles / "andorra-latest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"wikidata": "Q1"}]),
        data_root.processed_polygons / "andorra-latest.parquet",
    )
    progress = AugmentationProgress()

    augment_region(data_root, "andorra-latest", FakeAugmentationClient(), progress=progress)

    snapshot = progress.snapshot()
    assert snapshot.phase == "Writing sidecars"
    assert snapshot.completed == snapshot.total == 5
