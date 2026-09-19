"""Contracts for the shared public dataset-card summary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from osm_polygon_wikidata_only.hf.minimal_card import (
    ContinentCoverage,
    MinimalCardSnapshot,
    render_minimal_card,
)


def _snapshot() -> MinimalCardSnapshot:
    return MinimalCardSnapshot(
        front_matter="---\nlicense: odbl\n---\n",
        repo_id="NoeFlandre/example",
        title="Example dataset",
        description="A compact public dataset description.",
        polygon_rows=10,
        unique_polygon_identities=8,
        polygons_with_text=6,
        documents=12,
        sections=20,
        languages=4,
        regions=2,
        total_parquet_bytes=2_000_000,
        continent_rows=(ContinentCoverage("Europe", 8, 7, 2, 5, 6),),
    )


def test_minimal_card_has_the_shared_eight_row_snapshot_and_maps() -> None:
    card = render_minimal_card(_snapshot())

    assert card.startswith("---\nlicense: odbl\n---\n")
    snapshot_table = card.split("## Dataset snapshot\n", 1)[1].split(
        "\n\nPolygon rows", 1
    )[0]
    assert sum(line.startswith("| ") for line in snapshot_table.splitlines()) == 10
    assert card.count("| Metric | Value |") == 1
    assert "| Unique polygon identities (osm_type, osm_id) | 8 |" in card
    assert "| Polygons with successful non-empty text (unique OSM identities) | 6 |" in card
    assert "assets/coverage_map.png" in card
    assert "assets/geographic_text_presence.png" in card
    assert "assets/geographic_text_density.png" in card
    assert "## Geographic distribution by continent" in card
    assert "[`stats.json`](stats.json)" in card


def test_minimal_card_body_is_below_the_public_size_limit() -> None:
    card = render_minimal_card(_snapshot())
    body = card.split("\n---\n", 2)[-1]

    assert len(body.encode("utf-8")) < 8192


def test_minimal_card_rejects_an_oversized_description() -> None:
    snapshot = _snapshot()
    oversized = replace(snapshot, description="x" * 8192)

    with pytest.raises(ValueError, match="8 KiB"):
        render_minimal_card(oversized)
