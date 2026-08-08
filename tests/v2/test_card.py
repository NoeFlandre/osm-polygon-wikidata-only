from pathlib import Path

import pytest

from osm_polygon_wikidata_only.v2 import card
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
