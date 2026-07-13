"""Genuinely shared rendering primitives for the geographic visualizations.

This module owns:

* ``atomic_save_png``: write a matplotlib figure to disk via a
  temporary file then atomic rename, with cleanup on success and
  failure.
* ``format_percent_tick`` / ``format_count_tick``: colorbar tick
  formatters used by the coverage and count visualizations.

The figure layout constants and the world-extent axis initialization
live in :mod:`.basemap`; the per-visualization styling lives in
:mod:`.coverage` and :mod:`.polygon_count`. Nothing visualization-
specific (colormap, alpha, threshold, caption) belongs here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_save_png(fig: Any, output_path: Path) -> None:
    """Save ``fig`` to ``output_path`` via a temporary file then atomic rename."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        fig.savefig(
            str(tmp_path),
            format="png",
            facecolor="white",
            metadata={"Software": "osm-polygon-wikidata-only"},
        )
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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
