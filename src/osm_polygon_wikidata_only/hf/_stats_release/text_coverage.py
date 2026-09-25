"""Consistency check between headline and continent text coverage."""

from __future__ import annotations

import re
from pathlib import Path

from osm_polygon_wikidata_only.hf._stats_release.models import StatsReleaseError

_HEADLINE_TEXT_COVERAGE = re.compile(
    r"(?:\|\s*Polygons with successful non-empty text \(unique OSM identities\)\s*\|"
    r"|-\s*\*\*Polygons with non-empty Wikipedia or Wikivoyage text:\*\*)"
    r"\s*([\d,]+)"
)
_CONTINENT_TABLE_ROW = re.compile(r"^\|([^|\n]+)\|([^\n]*)\|\s*$", re.MULTILINE)


def require_consistent_text_coverage(card_path: Path) -> None:
    """Fail closed when the card states two different text-coverage totals.

    The headline figure and the per-continent breakdown are computed by
    separate renderers. They must agree, otherwise the published card
    contradicts itself -- and so does the map caption rendered from the
    same snapshot as the headline.
    """
    card = card_path.read_text(encoding="utf-8")
    headline = _headline_text_coverage(card)
    continent_total = _continent_text_coverage_total(card)
    if headline is None or continent_total is None:
        return
    if headline != continent_total:
        raise StatsReleaseError(
            "dataset card states inconsistent text-coverage totals: headline "
            f"{headline:,} but the continent table sums to {continent_total:,}; "
            "no release was published"
        )


def _headline_text_coverage(card: str) -> int | None:
    match = _HEADLINE_TEXT_COVERAGE.search(card)
    return int(match.group(1).replace(",", "")) if match else None


def _continent_text_coverage_total(card: str) -> int | None:
    section = _continent_section(card)
    if section is None:
        return None
    total = 0
    counted = False
    for line in section.splitlines():
        value = _continent_row_combined(line)
        if value is not None:
            total += value
            counted = True
    return total if counted else None


def _continent_section(card: str) -> str | None:
    heading = "## Geographic distribution by continent"
    start = card.find(heading)
    if start < 0:
        return None
    end = card.find("\n## ", start + len(heading))
    return card[start:] if end < 0 else card[start:end]


def _continent_row_combined(line: str) -> int | None:
    """Return the combined text-covered count from one continent data row."""
    cells = _continent_data_cells(line)
    return _parse_grouped_int(cells[5]) if cells else None


def _continent_data_cells(line: str) -> list[str] | None:
    """Return the cells of a continent data row, or None for any other line."""
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) != 7 or not cells[-1].endswith("%"):
        return None
    return cells


def _parse_grouped_int(value: str) -> int | None:
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None
