from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
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
    assert "**Additional document words in V2 (Wikipedia + Wikivoyage):** 6" in card_text
    assert "**Additional document sections in V2 (Wikipedia + Wikivoyage):** 0" in card_text
