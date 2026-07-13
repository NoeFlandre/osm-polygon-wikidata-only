"""Polygon-count-specific rendering pipeline.

Owns the polygon-count-only visual constants, the per-cell renderer,
the caption text, the logarithmic normalization, and the colourbar for
the polygon density visualization. Shares axis initialization, atomic
save, and the cell-rings / antimeridian geometry with :mod:`.coverage`
through :mod:`.basemap`, :mod:`.h3_geometry`, and :mod:`.rendering`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .aggregation import aggregate_geographic_polygon_count
from .basemap import _DPI, _FIGSIZE, draw_landmasses, init_axes
from .h3_geometry import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MIN_POLYGONS_PER_CELL,
    cell_rings,
)
from .models import CoverageMapError, PolygonCountCell, RenderResult
from .rendering import atomic_save_png, format_count_tick

LOGGER = logging.getLogger(__name__)


# Polygon-count-only visual constants.
_COUNT_COLORMAP_NAME = "magma"
_COUNT_ALPHA = 0.95


def coerce_count_cells(cells: Sequence[PolygonCountCell]) -> list[PolygonCountCell]:
    """Validate and copy ``cells`` into a deterministic list."""
    coerced: list[PolygonCountCell] = []
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, PolygonCountCell):
            raise CoverageMapError(
                f"All entries must be PolygonCountCell instances; got {type(cell).__name__}."
            )
        if cell.h3_cell in seen:
            raise CoverageMapError(f"Duplicate H3 cell id supplied to renderer: {cell.h3_cell}")
        seen.add(cell.h3_cell)
        coerced.append(cell)
    coerced.sort(key=lambda entry: entry.h3_cell)
    return coerced


def draw_count_cell(
    ax: Any,
    cell: PolygonCountCell,
    *,
    cmap: mcolors.Colormap,
    norm: mcolors.LogNorm,
) -> None:
    """Draw a single H3 cell on ``ax`` for the polygon count map.

    All cells use the colormap -- including low-sample cells, because
    low counts are the metric and must remain visible. Opacity is not
    used as a second data encoding.
    """
    safe_count = max(int(cell.polygon_count), 1)
    facecolor: Any = cmap(norm(safe_count))
    for ring in cell_rings(cell):
        patch = mpatches.Polygon(
            ring,
            closed=True,
            facecolor=facecolor,
            edgecolor="#333333",
            linewidth=0.2,
            alpha=_COUNT_ALPHA,
            zorder=3,
        )
        ax.add_patch(patch)


def render_geographic_polygon_count(
    cells: Sequence[PolygonCountCell],
    output_path: Path,
    *,
    land_features: Sequence[Any] | None = None,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> RenderResult:
    """Render the polygon count PNG and atomically write it to ``output_path``."""
    coerced = coerce_count_cells(cells)
    if not coerced:
        raise CoverageMapError("Cannot render polygon count map: no H3 cells supplied.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    fig.set_facecolor("white")
    init_axes(ax)
    if land_features:
        draw_landmasses(ax, land_features)

    counts = [cell.polygon_count for cell in coerced]
    minimum = max(min(counts), 1)
    maximum = max(max(counts), minimum + 1)
    cmap = plt.get_cmap(_COUNT_COLORMAP_NAME)
    norm = mcolors.LogNorm(vmin=minimum, vmax=maximum)
    for cell in coerced:
        draw_count_cell(ax, cell, cmap=cmap, norm=norm)

    total_polygons = sum(counts)
    low_sample_count = sum(1 for c in coerced if c.is_low_sample)
    caption = (
        "Geographic Polygon Density (Wikidata-tagged). Colour encodes the "
        "number of dataset polygons per H3 cell on a logarithmic scale. "
        "Each dataset polygon (already conditional on an OSM "
        "`wikidata=*` tag) is counted exactly once. "
        f"Grey cells (none here) are reserved for sub-threshold maps; on "
        f"this map every cell is coloured. {total_polygons:,} polygons across "
        f"{len(coerced):,} H3 cells ({low_sample_count:,} with fewer than "
        f"{min_polygons_per_cell} polygons)."
    )
    fig.suptitle(
        "Geographic Polygon Density",
        fontsize=14,
        color="#222222",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#444444",
        wrap=True,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Polygons per H3 cell", fontsize=8, color="#333333")
    colorbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_count_tick))
    colorbar.ax.tick_params(labelsize=7)

    try:
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        atomic_save_png(fig, output_path)
    finally:
        plt.close(fig)

    LOGGER.info("Wrote geographic polygon count map to %s", output_path)
    return RenderResult(output_path=output_path, caption=caption)


def generate_geographic_polygon_count(
    processed_root: Path,
    output_path: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
    land_cache_dir: Path | None = None,
) -> RenderResult:
    """Aggregate polygon counts and render the PNG to ``output_path``."""
    cells = aggregate_geographic_polygon_count(
        processed_root,
        h3_resolution=h3_resolution,
        min_polygons_per_cell=min_polygons_per_cell,
    )
    from .basemap import load_land_basemap

    land_features = load_land_basemap(land_cache_dir) if land_cache_dir else None
    return render_geographic_polygon_count(
        cells,
        output_path,
        land_features=land_features,
        min_polygons_per_cell=min_polygons_per_cell,
    )


__all__ = [
    "generate_geographic_polygon_count",
    "render_geographic_polygon_count",
]
