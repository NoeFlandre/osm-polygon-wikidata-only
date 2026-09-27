"""Step definitions for ``dataset_card.feature`` (#40, #115).

The fixtures are the raw ``compute_continent_stats`` tuples that both release
cards consume, so the card is checked against the statistics it was built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf.minimal_card import (
    MinimalCardSnapshot,
    continent_coverage_rows,
    render_minimal_card,
)
from osm_polygon_wikidata_only.hf.stats_release import (
    RELEASE_ASSET_FILES,
    StatsReleaseError,
    _require_consistent_text_coverage,
)
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

scenarios("dataset_card.feature")

# (continent, polygons, wikipedia docs, wikivoyage docs, wikipedia-text polygons, text polygons)
_STATS: dict[str, tuple[tuple[str, int, int, int, int, int], ...]] = {
    "V1": (("Europe", 1_200, 900, 40, 700, 720), ("Asia", 800, 500, 10, 300, 305)),
    "V2": (
        ("Africa", 50, 30, 0, 20, 20),
        ("Europe", 2_000, 1_800, 90, 1_500, 1_540),
        ("Oceania", 25, 12, 3, 10, 11),
    ),
}
_REPO = {"V1": DEFAULT_REPO_ID, "V2": V2_REPO_ID}


@dataclass
class _State:
    version: str = ""
    stats: tuple[tuple[str, int, int, int, int, int], ...] = ()
    card_path: Path | None = None
    card: str = ""
    errors: list[Exception] = field(default_factory=list)


def _snapshot(version: str, stats: tuple[tuple[str, int, int, int, int, int], ...]):
    rows = continent_coverage_rows(stats)
    return MinimalCardSnapshot(
        front_matter="---\nlicense: odbl\n---\n",
        repo_id=_REPO[version],
        title=f"{version} acceptance dataset",
        description="Offline acceptance fixture.",
        polygon_rows=sum(row.polygons for row in rows),
        unique_polygon_identities=sum(row.polygons for row in rows),
        polygons_with_text=sum(row.text_polygons for row in rows),
        documents=sum(row.wikipedia_documents + row.wikivoyage_documents for row in rows),
        sections=0,
        languages=2,
        regions=len(rows),
        total_parquet_bytes=1_000,
        continent_rows=rows,
    )


@pytest.fixture
def state() -> _State:
    return _State()


@given(parsers.parse("release statistics fixtures for the {version} dataset"))
def stats_fixture(state: _State, version: str) -> None:
    state.version = version
    state.stats = _STATS[version]


@when(parsers.parse("I render the {version} dataset card"))
def render_card(state: _State, version: str, tmp_path: Path) -> None:
    assert version == state.version
    state.card = render_minimal_card(_snapshot(version, state.stats))
    state.card_path = tmp_path / "README.md"
    state.card_path.write_text(state.card, encoding="utf-8")


@when("the rendered card headline disagrees with its continent table")
def contradicting_card(state: _State, tmp_path: Path) -> None:
    snapshot = _snapshot(state.version, state.stats)
    card = render_minimal_card(snapshot)
    headline = f"| {snapshot.polygons_with_text:,} |"
    assert headline in card
    card = card.replace(headline, f"| {snapshot.polygons_with_text + 1:,} |", 1)
    state.card_path = tmp_path / "README.md"
    state.card_path.write_text(card, encoding="utf-8")
    try:
        _require_consistent_text_coverage(state.card_path)
    except StatsReleaseError as error:
        state.errors.append(error)


@then("the card continent table lists every continent from the stats")
def continent_rows_present(state: _State) -> None:
    table = state.card.split("## Geographic distribution by continent", 1)[1]
    for name, polygons, wikipedia, wikivoyage, wikipedia_text, text in state.stats:
        rate = text / polygons
        expected = (
            f"| {name} | {polygons:,} | {wikipedia:,} | {wikivoyage:,} | "
            f"{wikipedia_text:,} | {text:,} | {rate:.1%} |"
        )
        assert expected in table


@then("the continent text coverage sums to the headline figure")
def coverage_consistent(state: _State) -> None:
    assert state.card_path is not None
    _require_consistent_text_coverage(state.card_path)
    total = sum(row[5] for row in state.stats)
    assert (
        f"| Polygons with successful non-empty text (unique OSM identities) | {total:,} |"
        in state.card
    )


@then("the card references every released map asset")
def assets_referenced(state: _State) -> None:
    for asset in RELEASE_ASSET_FILES:
        assert asset in state.card


@then("the release consistency guard refuses the card")
def guard_refuses(state: _State) -> None:
    assert len(state.errors) == 1
    assert "continent table sums to" in str(state.errors[0])
