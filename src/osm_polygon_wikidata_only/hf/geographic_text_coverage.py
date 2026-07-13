"""Generate a deterministic H3-aggregated geographic Wikipedia text coverage map.

The visualization answers a single, precisely-defined question:

    For each H3 cell at the configured resolution, what fraction of
    dataset polygons are linked to at least one Wikipedia article with
    non-empty full text?

The denominator counts every polygon row under
``processed/polygons/*.parquet`` exactly once. Polygons in the dataset
already require an OSM ``wikidata=*`` tag; that qualification is
inherited from the upstream schema and is documented in the rendered
caption.

The numerator counts unique polygons (never polygon-article links).
A polygon qualifies when at least one linked article's ``full_text``
is not null, not empty, and not whitespace-only.

Outputs are sorted by H3 cell id, the colormap scale is fixed at
``[0, 1]``, and the figure layout is fully deterministic. The function
is independent of CLI parsing and the network: it reads local Parquet
files and writes a PNG with no external dependencies at render time.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h3
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)


# Defaults and tunables --------------------------------------------------


DEFAULT_H3_RESOLUTION: int = 3
DEFAULT_MIN_POLYGONS_PER_CELL: int = 20
REMOTE_ASSET_PATH: str = "assets/geographic_wikipedia_text_coverage.png"
LOCAL_ASSET_PATH: str = "assets/geographic_wikipedia_text_coverage.png"

# Fixed visual constants: do not vary across runs to keep the output
# deterministic and visually consistent in the published README.
_FIGSIZE = (16, 8)
_DPI = 100
_OCEAN_COLOR = "#cfe2f3"
_LAND_COLOR = "#e8e0d0"
_LAND_EDGE = "#b8aa90"
_LOW_SAMPLE_COLOR = "#bdbdbd"
_LOW_SAMPLE_EDGE = "#8a8a8a"
_COLORMAP_NAME = "viridis"
_VMIN = 0.0
_VMAX = 1.0


class CoverageMapError(RuntimeError):
    """Raised for invalid inputs or missing required dataset artifacts."""


@dataclass(frozen=True, slots=True)
class CoverageCell:
    """One H3 cell's aggregated coverage statistics."""

    h3_cell: str
    polygon_count: int
    covered_polygon_count: int
    coverage_rate: float
    is_low_sample: bool


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Outcome of :func:`render_geographic_text_coverage`.

    The PNG is written to ``output_path`` and the exact caption text
    rendered onto the figure is exposed here so callers and tests can
    introspect it without parsing the rasterized image.
    """

    output_path: Path
    caption: str


# --- helpers ------------------------------------------------------------


def assign_h3_cell(lat: float, lon: float, *, resolution: int = DEFAULT_H3_RESOLUTION) -> str:
    """Map a centroid to its H3 cell id at the requested resolution.

    Raises :class:`CoverageMapError` on null, NaN, out-of-range, or
    non-finite coordinates or on an invalid resolution.
    """
    if lat is None or lon is None:
        raise CoverageMapError("Latitude and longitude must not be null.")
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError) as error:
        raise CoverageMapError(
            f"Latitude and longitude must be numeric; got lat={lat!r}, lon={lon!r}."
        ) from error
    if not (math_isfinite(lat_value) and math_isfinite(lon_value)):
        raise CoverageMapError(
            f"Latitude and longitude must be finite; got lat={lat_value!r}, lon={lon_value!r}."
        )
    if not (-90.0 <= lat_value <= 90.0):
        raise CoverageMapError(f"Latitude {lat_value} is outside the [-90, 90] range.")
    if not (-180.0 <= lon_value <= 180.0):
        raise CoverageMapError(f"Longitude {lon_value} is outside the [-180, 180] range.")
    if not isinstance(resolution, int) or not (0 <= resolution <= 15):
        raise CoverageMapError(f"H3 resolution must be an int in [0, 15]; got {resolution!r}.")
    try:
        return str(h3.latlng_to_cell(lat_value, lon_value, resolution))
    except (ValueError, h3.H3ValueError) as error:
        raise CoverageMapError(
            f"Could not assign H3 cell for ({lat_value}, {lon_value}) "
            f"at resolution {resolution}: {error}"
        ) from error


def math_isfinite(value: float) -> bool:
    """Local re-export so the function above is self-contained."""
    import math

    return math.isfinite(value)


# --- I/O ---------------------------------------------------------------


def _sorted_parquets(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def _read_required_columns(
    parquet_path: Path,
    columns: tuple[str, ...],
    *,
    label: str,
) -> list[dict[str, Any]]:
    import pyarrow as pa

    actual: set[str] = set()
    try:
        metadata = pq.read_metadata(parquet_path)  # type: ignore[no-untyped-call]
        actual = set(metadata.schema.names) - _PYARROW_INTERNAL_COLUMNS
    except Exception:
        actual = set()
    try:
        table = pq.read_table(parquet_path, columns=list(columns))  # type: ignore[no-untyped-call]
    except pa.ArrowInvalid as error:
        missing = sorted(set(columns) - actual)
        raise CoverageMapError(
            f"{label} parquet {parquet_path} is missing required columns: {missing}"
        ) from error
    except KeyError as error:
        missing = sorted(set(columns) - actual)
        raise CoverageMapError(
            f"{label} parquet {parquet_path} is missing required columns: {missing}"
        ) from error
    except OSError as error:
        raise CoverageMapError(f"Could not read {label} parquet {parquet_path}: {error}") from error
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


_PYARROW_INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {"__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
)


def _load_qualifying_article_ids(articles_dir: Path) -> set[str]:
    """Return the set of article IDs whose ``full_text`` is non-empty and non-whitespace."""
    qualifying: set[str] = set()
    for parquet_path in _sorted_parquets(articles_dir):
        for row in _read_required_columns(
            parquet_path, ("article_id", "full_text"), label="articles"
        ):
            text = row.get("full_text")
            if text is None:
                continue
            if not isinstance(text, str):
                continue
            if not text.strip():
                continue
            article_id = row.get("article_id")
            if article_id:
                qualifying.add(str(article_id))
    return qualifying


def _load_covered_polygon_ids(
    links_dir: Path,
    qualifying_article_ids: set[str],
) -> set[str]:
    """Return the set of polygon IDs linked to at least one qualifying article."""
    covered: set[str] = set()
    for parquet_path in _sorted_parquets(links_dir):
        for row in _read_required_columns(
            parquet_path, ("polygon_id", "article_id"), label="polygon_articles"
        ):
            article_id = row.get("article_id")
            if article_id is None:
                continue
            if str(article_id) not in qualifying_article_ids:
                continue
            polygon_id = row.get("polygon_id")
            if polygon_id:
                covered.add(str(polygon_id))
    return covered


def _load_polygon_cells(
    polygons_dir: Path,
    *,
    h3_resolution: int,
) -> list[tuple[str, str]]:
    """Return a sorted list of ``(polygon_id, h3_cell)`` tuples.

    Every polygon row in ``polygons/*.parquet`` must contribute to the
    denominator, so missing, null, non-finite, or out-of-range
    coordinates are never silently skipped. Invalid rows raise
    :class:`CoverageMapError` identifying the source parquet path and
    the offending polygon id so the operator can fix the data.
    """
    rows: list[tuple[str, str]] = []
    for parquet_path in _sorted_parquets(polygons_dir):
        table_rows = _read_required_columns(
            parquet_path, ("polygon_id", "lat", "lon"), label="polygons"
        )
        for row_index, row in enumerate(table_rows):
            polygon_id = row.get("polygon_id")
            lat = row.get("lat")
            lon = row.get("lon")
            if not polygon_id:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} is missing "
                    f"polygon_id; cannot include it in the visualization denominator."
                )
            if lat is None or lon is None:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} (polygon_id="
                    f"{polygon_id}) has null lat or lon; cannot include it in the "
                    f"visualization denominator."
                )
            try:
                cell = assign_h3_cell(lat, lon, resolution=h3_resolution)
            except CoverageMapError as error:
                raise CoverageMapError(
                    f"polygons parquet {parquet_path} row {row_index} (polygon_id="
                    f"{polygon_id}) has invalid coordinates (lat={lat}, lon={lon}): "
                    f"{error}"
                ) from error
            rows.append((str(polygon_id), cell))
    rows.sort(key=lambda pair: pair[0])
    return rows


def _require_directory(path: Path, *, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise CoverageMapError(
            f"Required {label} directory does not exist: {path}. "
            f"Run a complete PBF processing pass first."
        )
    return path


# --- Aggregation -------------------------------------------------------


def aggregate_geographic_text_coverage(
    processed_root: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> list[CoverageCell]:
    """Aggregate coverage statistics per H3 cell from local Parquet artifacts."""
    if min_polygons_per_cell < 1:
        raise CoverageMapError(f"min_polygons_per_cell must be >= 1; got {min_polygons_per_cell}")
    polygons_dir = _require_directory(processed_root / "polygons", label="polygons")
    articles_dir = _require_directory(processed_root / "articles", label="articles")
    links_dir = _require_directory(processed_root / "polygon_articles", label="polygon_articles")

    qualifying_article_ids = _load_qualifying_article_ids(articles_dir)
    covered_polygon_ids = _load_covered_polygon_ids(links_dir, qualifying_article_ids)
    polygon_cells = _load_polygon_cells(polygons_dir, h3_resolution=h3_resolution)

    counts: dict[str, int] = {}
    covered_counts: dict[str, int] = {}
    for polygon_id, cell in polygon_cells:
        counts[cell] = counts.get(cell, 0) + 1
        if polygon_id in covered_polygon_ids:
            covered_counts[cell] = covered_counts.get(cell, 0) + 1

    cells: list[CoverageCell] = []
    for h3_cell in sorted(counts):
        polygon_count = counts[h3_cell]
        covered_polygon_count = covered_counts.get(h3_cell, 0)
        coverage_rate = covered_polygon_count / polygon_count if polygon_count else 0.0
        cells.append(
            CoverageCell(
                h3_cell=h3_cell,
                polygon_count=polygon_count,
                covered_polygon_count=covered_polygon_count,
                coverage_rate=coverage_rate,
                is_low_sample=polygon_count < min_polygons_per_cell,
            )
        )
    LOGGER.info(
        "Aggregated %d H3 cell(s); %d covered polygon(s) of %d total.",
        len(cells),
        sum(c.covered_polygon_count for c in cells),
        sum(c.polygon_count for c in cells),
    )
    return cells


# --- Rendering ---------------------------------------------------------


def _coerce_cells(cells: Sequence[CoverageCell]) -> list[CoverageCell]:
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


def _opacity_for_count(count: int) -> float:
    """Deterministic log-scaled opacity in (0, 1] from polygon count."""
    if count <= 0:
        return 0.1
    import math

    # log10(1 + count) clamped to [0, 1], with a visible floor.
    scaled = math.log10(1.0 + count) / 4.0  # log10(10_000+1) ~= 4
    return max(0.15, min(1.0, scaled))


def _draw_cell_polygon(
    ax: Any, cell: CoverageCell, *, cmap: mcolors.Colormap, norm: mcolors.Normalize
) -> None:
    """Draw a single H3 cell on ``ax``, splitting antimeridian crossings."""
    try:
        boundary = h3.cell_to_boundary(cell.h3_cell)
    except (ValueError, h3.H3ValueError):
        LOGGER.warning("Could not fetch boundary for %s", cell.h3_cell)
        return
    if not boundary:
        return

    facecolor: Any
    edgecolor: str
    alpha: float
    if cell.is_low_sample:
        facecolor = _LOW_SAMPLE_COLOR
        edgecolor = _LOW_SAMPLE_EDGE
        alpha = 0.7
    else:
        facecolor = cmap(norm(cell.coverage_rate))
        edgecolor = "#333333"
        alpha = _opacity_for_count(cell.polygon_count)

    # ``boundary`` is a sequence of (lat, lon) tuples in h3 4.x. Split
    # rings at antimeridian crossings so we never draw across the world.
    boundary_pairs = list(boundary)
    raw_points: list[tuple[float, float]] = []
    for pair in boundary_pairs:
        if len(pair) >= 2:
            raw_points.append((float(pair[0]), float(pair[1])))
    points: list[tuple[float, float]] = [(lon, lat) for lat, lon in raw_points]
    for ring in _split_antimeridian(points):
        if len(ring) < 3:
            continue
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


def _split_antimeridian(
    points: Sequence[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Split a polygon ring at longitudinal jumps larger than 180 degrees."""
    if len(points) < 4:
        return [list(points)]
    from itertools import pairwise

    rings: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    for prev, curr in pairwise(points):
        if abs(curr[0] - prev[0]) > 180.0:
            rings.append(current)
            current = [curr]
        else:
            current.append(curr)
    if len(current) >= 3:
        rings.append(current)
    elif current:
        # Carry a degenerate tail onto the previous ring so we never
        # produce zero-area patches.
        if rings:
            rings[-1].extend(current)
        else:
            rings.append(current)
    return rings


def _load_land_basemap(cache_dir: Path) -> list[Any] | None:
    """Load the cached Natural Earth 110m landmass GeoJSON, if available.

    Returns the parsed ``features`` list, or ``None`` if the cache is
    missing. We intentionally do not download anything here; rendering
    without landmasses is acceptable and the surrounding module is
    documented as network-free.
    """
    candidate = cache_dir / "ne_110m_land.geojson"
    if not candidate.exists() or candidate.stat().st_size == 0:
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.warning("Could not read cached land GeoJSON: %s", error)
        return None
    if not isinstance(data, dict):
        return None
    return data.get("features") or []


def _draw_landmasses(ax: Any, features: Sequence[Any]) -> None:
    """Draw Natural Earth landmasses on ``ax``."""
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        coords = geom.get("coordinates")
        gtype = geom.get("type")
        if gtype == "Polygon" and coords:
            _draw_land_ring(ax, coords[0])
        elif gtype == "MultiPolygon" and coords:
            for polygon in coords:
                if polygon:
                    _draw_land_ring(ax, polygon[0])


def _draw_land_ring(ax: Any, ring: Sequence[Sequence[float]]) -> None:
    if not ring or len(ring) < 3:
        return
    patch = mpatches.Polygon(
        [(float(lon), float(lat)) for lon, lat in ring],
        closed=True,
        facecolor=_LAND_COLOR,
        edgecolor=_LAND_EDGE,
        linewidth=0.2,
        zorder=1,
    )
    ax.add_patch(patch)


def render_geographic_text_coverage(
    cells: Sequence[CoverageCell],
    output_path: Path,
    *,
    land_features: Sequence[Any] | None = None,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> RenderResult:
    """Render the coverage PNG and atomically write it to ``output_path``.

    The :class:`RenderResult` exposes the exact caption text drawn onto
    the figure so callers (and tests) can audit it without parsing the
    rasterized PNG.
    """
    coerced = _coerce_cells(cells)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    fig.set_facecolor("white")
    ax.set_facecolor(_OCEAN_COLOR)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(True, color="#ffffff", linewidth=0.3, alpha=0.4)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_aspect("equal", adjustable="box")

    if land_features:
        _draw_landmasses(ax, land_features)

    cmap = plt.get_cmap(_COLORMAP_NAME)
    norm = mcolors.Normalize(vmin=_VMIN, vmax=_VMAX)
    for cell in coerced:
        _draw_cell_polygon(ax, cell, cmap=cmap, norm=norm)

    covered_total = sum(c.covered_polygon_count for c in coerced)
    polygon_total = sum(c.polygon_count for c in coerced)
    overall_pct = 100.0 * covered_total / polygon_total if polygon_total > 0 else 0.0
    low_sample_count = sum(1 for c in coerced if c.is_low_sample)
    caption = (
        "Geographic Wikipedia Text Coverage. Colour shows the share of dataset "
        "polygons (already conditional on an OSM `wikidata=*` tag) linked to at "
        "least one Wikipedia article with non-empty text. Cell fill encodes "
        "coverage (0-100%); cell opacity encodes polygon count on a log scale. "
        f"Grey cells hold fewer than {min_polygons_per_cell} polygons and "
        "are not statistically meaningful. "
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
    # The underlying normalization stays in [0, 1]; the colorbar ticks are
    # formatted as percentages so the legend reads "0% ... 100%" instead of
    # "0.0 ... 1.0".
    colorbar.set_label("Polygons with non-empty Wikipedia text (%)", fontsize=8, color="#333333")
    colorbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(_format_percent_tick))
    colorbar.ax.tick_params(labelsize=7)

    try:
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
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
    finally:
        plt.close(fig)

    LOGGER.info("Wrote geographic Wikipedia text coverage map to %s", output_path)
    return RenderResult(output_path=output_path, caption=caption)


def _format_percent_tick(value: float, _position: int | None = None) -> str:
    """Format a [0, 1] colorbar value as an integer percentage label."""
    return f"{round(value * 100)}%"


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
    land_features = _load_land_basemap(land_cache_dir) if land_cache_dir else None
    return render_geographic_text_coverage(
        cells,
        output_path,
        land_features=land_features,
        min_polygons_per_cell=min_polygons_per_cell,
    )


__all__ = [
    "DEFAULT_H3_RESOLUTION",
    "DEFAULT_MIN_POLYGONS_PER_CELL",
    "LOCAL_ASSET_PATH",
    "REMOTE_ASSET_PATH",
    "CoverageCell",
    "CoverageMapError",
    "RenderResult",
    "aggregate_geographic_text_coverage",
    "assign_h3_cell",
    "generate_geographic_text_coverage",
    "render_geographic_text_coverage",
]
