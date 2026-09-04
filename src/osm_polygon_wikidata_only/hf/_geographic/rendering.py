"""Genuinely shared rendering primitives for the geographic visualizations.

This module owns:

* ``atomic_save_png``: publish a matplotlib figure through the shared
  :func:`osm_polygon_wikidata_only.io.atomic.atomic_replacement` ritual,
  so a partial render is never visible at the output path.
* ``format_percent_tick`` / ``format_count_tick``: colorbar tick
  formatters used by the coverage and count visualizations.

The figure layout constants and the world-extent axis initialization
live in :mod:`.basemap`; the per-visualization styling lives in
:mod:`.coverage` and :mod:`.polygon_count`. Nothing visualization-
specific (colormap, alpha, threshold, caption) belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_replacement


def atomic_save_png(fig: Any, output_path: Path) -> None:
    """Save ``fig`` to ``output_path`` via a temporary file then atomic rename."""
    with atomic_replacement(output_path) as temporary:
        fig.savefig(
            str(temporary),
            format="png",
            facecolor="white",
            metadata={"Software": "osm-polygon-wikidata-only"},
        )


def format_percent_tick(value: float, _position: int | None = None) -> str:
    """Format a [0, 1] colorbar value as an integer percentage label."""
    return f"{round(value * 100)}%"


def format_count_tick(value: float, _position: int | None = None) -> str:
    """Format a polygon-count colorbar value as a human-readable integer label."""
    count = round(value)
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        thousands = count / 1_000.0
        return f"{thousands:.0f}k" if thousands.is_integer() else f"{thousands:.1f}k"
    millions = count / 1_000_000.0
    return f"{millions:.1f}M"


__all__ = [
    "atomic_save_png",
    "format_count_tick",
    "format_percent_tick",
]
