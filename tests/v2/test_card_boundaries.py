"""Structural seams for the V2 card facade."""

from __future__ import annotations

import inspect

from osm_polygon_wikidata_only.v2 import card, card_metrics, card_models, card_rendering


def test_card_facade_delegates_metrics_and_rendering_to_focused_modules() -> None:
    assert card.V2CardStats is card_models.V2CardStats
    assert card._CardFiles is card_models._CardFiles
    assert inspect.getmodule(card.compute_v2_card_stats) is card_metrics
    assert inspect.getmodule(card.render_v2_card) is card_rendering

