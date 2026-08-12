"""V2 geographic assets derived from the isolated V2 artifact tree."""

from __future__ import annotations

import logging
from pathlib import Path

from osm_polygon_wikidata_only.hf.coverage_map import (
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)
from osm_polygon_wikidata_only.hf.geographic_text_density import (
    generate_geographic_text_density,
)
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    generate_geographic_text_presence,
    load_text_presence,
)

LOGGER = logging.getLogger(__name__)


def generate_v2_map_assets(
    processed_v2: Path,
    output_dir: Path,
    *,
    land_geojson_path: Path | None = None,
    land_cache_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Render the three public V2 map views from V2 Parquet files.

    The all-polygons map, text-presence map, and H3 text-density map use the
    same rendering contracts as V1 but read only ``processed_v2``. No V1
    files are consulted, which keeps the two published cards independent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if land_geojson_path is None and land_cache_dir is not None:
        try:
            land_geojson_path = ensure_world_land(land_cache_dir)
        except Exception as error:
            LOGGER.warning("V2 maps will omit Natural Earth land context: %s", error)

    coverage_path = output_dir / "coverage_map.png"
    lons, lats = load_centroids_from_parquet(processed_v2 / "polygons")
    generate_coverage_map(
        lons,
        lats,
        coverage_path,
        land_geojson_path=land_geojson_path,
        title="V2 dataset coverage",
    )

    links_dir = processed_v2 / "polygon_document_links"
    presence_snapshot = load_text_presence(processed_v2, links_dir=links_dir)

    presence_path = output_dir / "geographic_text_presence.png"
    generate_geographic_text_presence(
        processed_v2,
        presence_path,
        land_geojson_path=land_geojson_path,
        snapshot=presence_snapshot,
    )

    density_path = output_dir / "geographic_text_density.png"
    generate_geographic_text_density(
        processed_v2,
        density_path,
        land_cache_dir=land_cache_dir,
        snapshot=presence_snapshot,
    )
    return coverage_path, presence_path, density_path


__all__ = ["generate_v2_map_assets"]
