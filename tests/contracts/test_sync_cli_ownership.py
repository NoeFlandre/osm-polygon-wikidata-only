"""Focused ownership contracts for the stable sync CLI composition root."""

from osm_polygon_wikidata_only.cli import run_sync
from osm_polygon_wikidata_only.cli._sync import retirement


def test_sync_cli_uses_focused_retirement_pairing() -> None:
    assert run_sync._paired_retirement_stems is retirement.paired_retirement_stems
