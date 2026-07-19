"""Generate a world coverage map from polygon centroids.

Reads ``lat``/``lon`` columns from all processed polygon parquet files
and renders a world map PNG with actual landmasses (Natural Earth
110m simplified data) overlaid with one point per polygon. Designed
to scale to worldwide coverage: parquet columnar reads fetch only the
two columns needed and matplotlib scatter handles millions of points.
"""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

WORLD_LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_land.geojson"
)
WORLD_LAND_FILENAME = "ne_110m_land.geojson"
WORLD_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
WORLD_COUNTRIES_FILENAME = "ne_110m_admin_0_countries.geojson"

_OCEAN_COLOR = "#cfe2f3"
_LAND_COLOR = "#e8e0d0"
_LAND_EDGE = "#b8aa90"
_POINT_COLOR = "#e8743b"
_POINT_EDGE = "#c0392b"
_POINT_SIZE = 1.5
_POINT_ALPHA = 0.5


def load_centroids_from_parquet(polygons_dir: Path) -> tuple[list[float], list[float]]:
    """Read polygon centroids from all parquet files in ``polygons_dir``.

    Returns ``(lons, lats)`` parallel lists. Only the ``lon`` and ``lat``
    columns are read (columnar pruning), so this is fast even for
    hundreds of processed PBFs. ``None``/missing values are skipped.
    """
    lons: list[float] = []
    lats: list[float] = []
    if not polygons_dir.exists():
        return lons, lats
    for parquet_path in sorted(polygons_dir.glob("*.parquet")):
        try:
            table = pq.read_table(parquet_path, columns=["lon", "lat"])  # type: ignore[no-untyped-call]
        except (OSError, KeyError) as e:
            LOGGER.warning("Skipping %s: %s", parquet_path, e)
            continue
        for row_lon, row_lat in zip(
            table.column("lon").to_pylist(), table.column("lat").to_pylist(), strict=True
        ):
            if row_lon is None or row_lat is None:
                continue
            lons.append(float(row_lon))
            lats.append(float(row_lat))
    return lons, lats


def ensure_world_land(cache_dir: Path) -> Path:
    """Download and cache the Natural Earth 110m land GeoJSON.

    Returns the path to the cached file. If the file already exists
    and is non-empty, it is reused without re-downloading.
    """
    cache_path = cache_dir / WORLD_LAND_FILENAME
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading world land GeoJSON from %s", WORLD_LAND_URL)
    urllib.request.urlretrieve(WORLD_LAND_URL, cache_path)
    LOGGER.info("Cached world land GeoJSON (%d bytes)", cache_path.stat().st_size)
    return cache_path


def ensure_world_countries(cache_dir: Path) -> Path:
    """Copy the bundled Natural Earth countries into the runtime cache.

    Bundling the small 110m reference makes dataset-card generation
    deterministic and offline; it also prevents a metadata-only
    publication from unexpectedly depending on GitHub availability.
    """
    cache_path = cache_dir / WORLD_COUNTRIES_FILENAME
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(__file__).with_name(WORLD_COUNTRIES_FILENAME)
    if not bundled.is_file():
        raise FileNotFoundError(
            f"Bundled Natural Earth country reference is missing: {bundled}. "
            "Reinstall the package before publishing the dataset card."
        )
    shutil.copyfile(bundled, cache_path)
    LOGGER.info("Cached world country GeoJSON (%d bytes)", cache_path.stat().st_size)
    return cache_path


def generate_coverage_map(
    lons: list[float],
    lats: list[float],
    output_path: Path,
    *,
    land_geojson_path: Path | None = None,
    title: str = "Dataset Coverage",
    figsize: tuple[float, float] = (16, 8),
    dpi: int = 100,
    point_size: float = _POINT_SIZE,
) -> Path:
    """Render the coverage map PNG with one scatter point per polygon.

    Landmasses from the Natural Earth 110m GeoJSON provide geographic
    context (optional; pass ``land_geojson_path`` to enable). Points
    are drawn on top with small markers and moderate transparency so
    dense clusters darken naturally.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.set_facecolor("white")
    ax.set_facecolor(_OCEAN_COLOR)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(True, color="#ffffff", linewidth=0.3, alpha=0.4)
    ax.tick_params(colors="#666666", labelsize=7)

    if land_geojson_path is not None and land_geojson_path.exists():
        _draw_landmasses(ax, land_geojson_path)

    if lons and lats:
        ax.scatter(
            lons,
            lats,
            s=point_size,
            c=_POINT_COLOR,
            edgecolors=_POINT_EDGE,
            linewidths=0.1,
            alpha=_POINT_ALPHA,
            zorder=3,
        )

    polygon_label = "polygon" if len(lons) == 1 else "polygons"
    ax.set_title(
        f"{title} - {len(lons):,} {polygon_label} plotted",
        fontsize=11,
        color="#333333",
        pad=10,
    )
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(str(output_path), format="png", facecolor="white")
    plt.close(fig)

    LOGGER.info("Wrote coverage map (%d points) to %s", len(lons), output_path)
    return output_path


def _draw_landmasses(ax: Any, geojson_path: Path) -> None:
    """Parse a GeoJSON file and draw filled landmass polygons."""
    data: dict[str, Any] = json.loads(geojson_path.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon" and coords:
            _draw_polygon_rings(ax, coords)
        elif gtype == "MultiPolygon" and coords:
            for polygon_coords in coords:
                _draw_polygon_rings(ax, polygon_coords)


def _draw_polygon_rings(ax: Any, rings: list[list[list[float]]]) -> None:
    """Draw a single polygon (list of coordinate rings) on the axes.

    The first ring is the outer boundary; subsequent rings are holes
    (rare at 110m resolution, but handled for correctness).
    """
    if not rings:
        return
    patch = mpatches.Polygon(
        rings[0],
        closed=True,
        facecolor=_LAND_COLOR,
        edgecolor=_LAND_EDGE,
        linewidth=0.3,
        zorder=1,
    )
    ax.add_patch(patch)
    for hole in rings[1:]:
        hole_patch = mpatches.Polygon(
            hole,
            closed=True,
            facecolor=_OCEAN_COLOR,
            edgecolor=_LAND_EDGE,
            linewidth=0.3,
            zorder=2,
        )
        ax.add_patch(hole_patch)


__all__ = [
    "WORLD_LAND_FILENAME",
    "ensure_world_land",
    "generate_coverage_map",
    "load_centroids_from_parquet",
]
