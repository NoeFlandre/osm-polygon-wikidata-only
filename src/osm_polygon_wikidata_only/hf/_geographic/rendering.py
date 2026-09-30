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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from osm_polygon_wikidata_only.io.atomic import atomic_replacement

from .basemap import DPI, FIGSIZE, draw_landmasses, init_axes


def create_map_axes(land_features: Sequence[Any] | None) -> tuple[Any, Any]:
    """Create the shared map figure and initialize its geographic axes."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.set_facecolor("white")
    init_axes(ax)
    if land_features:
        draw_landmasses(ax, land_features)
    return fig, ax


def save_map_figure(fig: Any, output_path: Path) -> None:
    """Fit, atomically save, and close a map figure on every exit path."""
    try:
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        atomic_save_png(fig, output_path)
    finally:
        plt.close(fig)


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
    "create_map_axes",
    "format_count_tick",
    "format_percent_tick",
    "save_map_figure",
]
