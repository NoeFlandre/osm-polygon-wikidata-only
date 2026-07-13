"""Coverage-specific rendering pipeline.

Owns the coverage-only visual constants, the per-cell renderer, the
caption text, the normalization, and the colourbar for the Wikipedia
text coverage visualization. Shares axis initialization, atomic save,
and the cell-rings / antimeridian geometry with :mod:`.polygon_count`
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

from .aggregation import aggregate_geographic_text_coverage
from .basemap import _DPI, _FIGSIZE, draw_landmasses, init_axes
from .h3_geometry import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MIN_POLYGONS_PER_CELL,
    cell_rings,
)
from .models import CoverageCell, CoverageMapError, RenderResult
from .parquet_inputs import (
    sorted_parquets,  # noqa: F401  (kept for downstream consumer compatibility)
)
from .rendering import atomic_save_png, format_percent_tick

LOGGER = logging.getLogger(__name__)


# Coverage-only visual constants: do not vary across runs to keep the
# output deterministic and visually consistent in the published README.
_LOW_SAMPLE_COLOR = "#bdbdbd"
_LOW_SAMPLE_EDGE = "#8a8a8a"
_COVERAGE_COLORMAP_NAME = "viridis"
_VMIN = 0.0
_VMAX = 1.0
_COVERAGE_ALPHA = 1.0
_LOW_SAMPLE_ALPHA = 0.7


def coerce_coverage_cells(cells: Sequence[CoverageCell]) -> list[CoverageCell]:
    """Validate and copy ``cells`` into a deterministic list."""
    coerced: list[CoverageCell] = []
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, CoverageCell):
            raise CoverageMapError(
                f"All entries must be CoverageCell instances; got {type(cell).__name__}."
            )
        if cell.h3_cell in seen:
            raise CoverageMapError(f"Duplicate H3 cell id supplied to renderer: {cell.h3_cell}")
        seen.add(cell.h3_cell)
        coerced.append(cell)
    coerced.sort(key=lambda entry: entry.h3_cell)
    return coerced


def draw_coverage_cell(
    ax: Any,
    cell: CoverageCell,
    *,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
) -> None:
    """Draw a single H3 cell on ``ax`` for the coverage map.

    Eligible (non-low-sample) cells use the configured full alpha so
    polygon count is not encoded as opacity. Grey is reserved for
    low-sample cells flagged by the threshold.
    """
    facecolor: Any
    edgecolor: str
    alpha: float
    if cell.is_low_sample:
        facecolor = _LOW_SAMPLE_COLOR
        edgecolor = _LOW_SAMPLE_EDGE
        alpha = _LOW_SAMPLE_ALPHA
    else:
        facecolor = cmap(norm(cell.coverage_rate))
        edgecolor = "#333333"
        alpha = _COVERAGE_ALPHA
    for ring in cell_rings(cell):
        patch = mpatches.Polygon(
            ring,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.2,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(patch)


def render_geographic_text_coverage(
    cells: Sequence[CoverageCell],
    output_path: Path,
    *,
    land_features: Sequence[Any] | None = None,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> RenderResult:
    """Render the coverage PNG and atomically write it to ``output_path``."""
    coerced = coerce_coverage_cells(cells)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    fig.set_facecolor("white")
    init_axes(ax)
    if land_features:
        draw_landmasses(ax, land_features)

    cmap = plt.get_cmap(_COVERAGE_COLORMAP_NAME)
    norm = mcolors.Normalize(vmin=_VMIN, vmax=_VMAX)
    for cell in coerced:
        draw_coverage_cell(ax, cell, cmap=cmap, norm=norm)

    covered_total = sum(c.covered_polygon_count for c in coerced)
    polygon_total = sum(c.polygon_count for c in coerced)
    overall_pct = 100.0 * covered_total / polygon_total if polygon_total > 0 else 0.0
    low_sample_count = sum(1 for c in coerced if c.is_low_sample)
    caption = (
        "Geographic Wikipedia Text Coverage. Colour encodes the share of "
        "dataset polygons (already conditional on an OSM `wikidata=*` tag) "
        "linked to at least one Wikipedia article with non-empty text, "
        f"from 0% to 100%. Grey cells hold fewer than {min_polygons_per_cell} "
        "polygons and are not statistically meaningful. "
        f"{polygon_total:,} polygons across {len(coerced):,} H3 cells "
        f"({covered_total:,} covered, {overall_pct:.1f}% overall; "
        f"{low_sample_count:,} low-sample)."
    )
    fig.suptitle(
        "Geographic Wikipedia Text Coverage",
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
    colorbar.set_label("Polygons with non-empty Wikipedia text (%)", fontsize=8, color="#333333")
    colorbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_percent_tick))
    colorbar.ax.tick_params(labelsize=7)

    try:
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        atomic_save_png(fig, output_path)
    finally:
        plt.close(fig)

    LOGGER.info("Wrote geographic Wikipedia text coverage map to %s", output_path)
    return RenderResult(output_path=output_path, caption=caption)


def generate_geographic_text_coverage(
    processed_root: Path,
    output_path: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
    land_cache_dir: Path | None = None,
) -> RenderResult:
    """Aggregate coverage and render the PNG to ``output_path``."""
    cells = aggregate_geographic_text_coverage(
        processed_root,
        h3_resolution=h3_resolution,
        min_polygons_per_cell=min_polygons_per_cell,
    )
    from .basemap import load_land_basemap

    land_features = load_land_basemap(land_cache_dir) if land_cache_dir else None
    return render_geographic_text_coverage(
        cells,
        output_path,
        land_features=land_features,
        min_polygons_per_cell=min_polygons_per_cell,
    )


__all__ = [
    "generate_geographic_text_coverage",
    "render_geographic_text_coverage",
]
