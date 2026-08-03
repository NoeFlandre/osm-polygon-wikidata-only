from pathlib import Path

from osm_polygon_wikidata_only.v2.card import render_v2_card, write_v2_card
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
