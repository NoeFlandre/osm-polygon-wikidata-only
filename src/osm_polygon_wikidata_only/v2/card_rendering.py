"""Deterministic V2 card rendering primitives."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.hf.minimal_card import render_minimal_card
from osm_polygon_wikidata_only.v2.card_front_matter import (
    has_parquet,
    render_front_matter,
)
from osm_polygon_wikidata_only.v2.card_models import V2CardStats
from osm_polygon_wikidata_only.v2.card_release import build_minimal_v2_release_snapshot


def render_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
    generated_on: str | None = None,
) -> str:
    """Render a concise, viewer-compatible card from V2 files on disk."""
    prepared = build_minimal_v2_release_snapshot(
        processed_v2,
        v1_processed=v1_processed,
        generated_on=generated_on,
        stats=stats,
    )
    return render_minimal_card(prepared.card)


__all__ = ["has_parquet", "render_front_matter", "render_v2_card"]
