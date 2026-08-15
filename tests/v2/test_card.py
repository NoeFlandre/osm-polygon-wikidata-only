from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2 import card
from osm_polygon_wikidata_only.v2.card import (
    compute_v2_card_stats,
    render_v2_card,
    write_v2_card,
)
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def test_card_is_factual_and_deterministic(tmp_path: Path) -> None:
    write_v2_region(
        tmp_path,
        "region-latest",
        polygons=[{"polygon_id": "p1", "has_wikidata": False}],
        documents=[],
        links=[],
    )
    first = render_v2_card(tmp_path)
    second = render_v2_card(tmp_path)
    assert first == second
    assert "Wikipedia-tag-only polygons:** 1" in first
    assert "exact V1 22-column section schema" in first
    assert "--dataset-version v2" in first
    assert "external drive" not in first
    assert first.rstrip().endswith(
        "Download the dataset citation metadata from [`CITATION.cff`](CITATION.cff)."
    )
    assert first.index("## Citation") > first.index("## Reproducibility")
    assert write_v2_card(tmp_path).read_text(encoding="utf-8") == first


def test_write_v2_card_preserves_previous_card_when_atomic_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed card write must leave the previous README intact."""
    target = tmp_path / "README.md"
    target.write_text("previous card\n", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(card, "atomic_write_text", fail, raising=False)

    with pytest.raises(OSError, match="disk full"):
        write_v2_card(tmp_path)

    assert target.read_text(encoding="utf-8") == "previous card\n"
    assert not list(tmp_path.glob(".README.md.*.tmp"))


def _empty_row(schema: pa.Schema) -> dict[str, object]:
    return {field.name: None for field in schema}


def test_v2_card_reports_word_and_section_deltas_from_v1(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"

    v1_document_schema = wikipedia_document_schema()
    v1_document = _empty_row(v1_document_schema)
    v1_document.update({"document_id": "v1-doc", "article_length_words": 4})
    v1_document_path = v1 / "wikipedia" / "documents" / "region-latest.parquet"
    v1_document_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_document], schema=v1_document_schema),
        v1_document_path,
    )

    v1_section_schema = section_schema()
    v1_section = _empty_row(v1_section_schema)
    v1_section.update({"section_id": "v1-section", "document_id": "v1-doc"})
    v1_section_path = v1 / "wikipedia" / "sections" / "region-latest.parquet"
    v1_section_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([v1_section], schema=v1_section_schema), v1_section_path)

    v2_document_schema = wikipedia_document_v2_schema()
    v2_document = _empty_row(v2_document_schema)
    v2_document.update({"document_id": "v2-doc", "article_length_words": 10})
    v2_section = _empty_row(v1_section_schema)
    v2_section.update({"section_id": "v2-section", "document_id": "v2-doc"})
    write_v2_region(
        v2,
        "region-latest",
        polygons=[],
        documents=[v2_document],
        links=[],
        sections=[v2_section],
    )

    stats = compute_v2_card_stats(v2, v1_processed=v1)

    assert stats.additional_document_words_vs_v1 == 6
    assert stats.additional_sections_vs_v1 == 0
    card_text = render_v2_card(v2, v1_processed=v1, stats=stats)
    assert "**Additional document-row words in V2 (Wikipedia + Wikivoyage):** 6" in card_text
    assert "**Additional section rows in V2 (Wikipedia + Wikivoyage):** 0" in card_text


def test_v2_card_reports_source_split_and_unique_content_deltas(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"

    v1_polygon = empty_row(tuple(field.name for field in polygon_schema()))
    v1_polygon.update({"polygon_id": "region-latest:way:1"})
    v1_polygon_path = v1 / "polygons" / "region-latest.parquet"
    v1_polygon_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_polygon], schema=polygon_schema()),
        v1_polygon_path,
    )

    v1_document_schema = wikipedia_document_schema()
    v1_document = _empty_row(v1_document_schema)
    v1_document.update(
        {
            "document_id": "old-doc",
            "article_length_words": 4,
            "content_hash": "shared-content",
        }
    )
    v1_document_path = v1 / "wikipedia" / "documents" / "region-latest.parquet"
    v1_document_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_document], schema=v1_document_schema),
        v1_document_path,
    )

    v1_section_schema = section_schema()
    v1_section = _empty_row(v1_section_schema)
    v1_section.update({"section_id": "old-section", "document_id": "old-doc"})
    v1_section_path = v1 / "wikipedia" / "sections" / "region-latest.parquet"
    v1_section_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_section], schema=v1_section_schema),
        v1_section_path,
    )

    def v2_polygon(polygon_id: str, sources: str) -> dict[str, object]:
        row = empty_row(tuple(field.name for field in polygon_schema()))
        row.update(
            {
                "polygon_id": polygon_id,
                "discovery_sources": sources,
                "has_wikidata": "wikidata" in sources,
            }
        )
        return row

    v2_document_schema = wikipedia_document_v2_schema()
    v2_document = _empty_row(v2_document_schema)
    v2_document.update(
        {
            "document_id": "new-doc",
            "article_length_words": 10,
            "content_hash": "shared-content",
        }
    )
    v2_section = _empty_row(v1_section_schema)
    v2_section.update({"section_id": "new-section", "document_id": "new-doc"})
    write_v2_region(
        v2,
        "region-latest",
        polygons=[
            v2_polygon("region-latest:way:1", '["wikidata"]'),
            v2_polygon("region-latest:way:2", '["wikipedia_tag"]'),
            v2_polygon("region-latest:way:3", '["wikidata","wikipedia_tag"]'),
            v2_polygon("region-latest:way:4", '["wikidata"]'),
        ],
        documents=[v2_document],
        links=[
            {
                "polygon_id": "region-latest:way:3",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            }
        ],
        sections=[v2_section],
    )

    stats = compute_v2_card_stats(v2, v1_processed=v1)
    assert stats.new_polygons_vs_v1 == 3
    assert stats.new_polygons_wikipedia_tag_vs_v1 == 2
    assert stats.new_polygons_wikidata_only_vs_v1 == 1
    assert stats.new_wikipedia_tag_polygons_without_document == 1
    assert stats.new_wikipedia_document_identity_words_vs_v1 == 10
    assert stats.new_wikipedia_documents_sharing_v1_content == 1
    assert stats.additional_unique_sections_vs_v1 == 1

    card_text = render_v2_card(v2, v1_processed=v1, stats=stats)
    assert "**Additional polygon identities:** 3" in card_text
    assert "**Of those, polygons with a Wikipedia tag:** 2" in card_text
    assert "**Of those, Wikidata-only polygons:** 1" in card_text
    assert "**New Wikipedia-tag polygons without a matching page at the snapshot:** 1" in card_text
    assert "**Words in newly added Wikipedia document identities:** 10" in card_text
    assert "**New Wikipedia document identities sharing content with V1:** 1" in card_text
    assert "**Additional unique section identities:** 1" in card_text
