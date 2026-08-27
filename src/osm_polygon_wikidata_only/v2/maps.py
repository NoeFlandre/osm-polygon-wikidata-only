"""V2 geographic assets derived from the isolated V2 artifact tree."""

from __future__ import annotations

import logging
from pathlib import Path

from osm_polygon_wikidata_only.hf.coverage_map import (
    WORLD_LAND_FILENAME,
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
from osm_polygon_wikidata_only.v2.comparison import (
    select_v2_added_wikipedia_tag_document_polygon_ids,
)
from osm_polygon_wikidata_only.v2.config import V2_ADDED_WIKIPEDIA_TAG_MAP_FILENAME

LOGGER = logging.getLogger(__name__)


def generate_v2_map_assets(
    processed_v2: Path,
    output_dir: Path,
    *,
    v1_processed: Path | None = None,
    land_geojson_path: Path | None = None,
    land_cache_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Render the three public V2 map views from V2 Parquet files.

    The all-polygons map, text-presence map, and H3 text-density map use the
    same rendering contracts as V1 but read only ``processed_v2``. When
    ``v1_processed`` is supplied, the V2-only Wikipedia-tag comparison map
    is written alongside them.

    When called directly with a standard data-root layout, an existing
    ``<data-root>/cache/ne_110m_land.geojson`` is discovered automatically.
    This keeps manually refreshed V2 assets consistent with the sync path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    land_geojson_path, land_cache_dir = _resolve_land_context(
        processed_v2, land_geojson_path, land_cache_dir
    )
    coverage_path = _render_coverage(processed_v2, output_dir, land_geojson_path)
    presence_path, density_path = _render_text_maps(
        processed_v2, output_dir, land_geojson_path, land_cache_dir
    )
    if v1_processed is not None:
        generate_v2_added_wikipedia_tag_map(
            processed_v2,
            v1_processed,
            output_dir / V2_ADDED_WIKIPEDIA_TAG_MAP_FILENAME,
            land_geojson_path=land_geojson_path,
        )
    return coverage_path, presence_path, density_path


def generate_v2_added_wikipedia_tag_map(
    processed_v2: Path,
    v1_processed: Path,
    output_path: Path,
    *,
    land_geojson_path: Path | None = None,
) -> Path:
    """Render only V2 polygons added through newly retrieved Wikipedia tags."""
    polygon_ids = select_v2_added_wikipedia_tag_document_polygon_ids(
        processed_v2,
        v1_processed,
    )
    lons, lats = load_centroids_from_parquet(
        processed_v2 / "polygons",
        polygon_ids=polygon_ids,
    )
    return generate_coverage_map(
        lons,
        lats,
        output_path,
        land_geojson_path=land_geojson_path,
        title="V2-added Wikipedia-tag document polygons",
    )


def _resolve_land_context(
    processed_v2: Path,
    land_geojson_path: Path | None,
    land_cache_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    if land_geojson_path is not None:
        return land_geojson_path, land_cache_dir
    if land_cache_dir is None:
        sibling = _existing_sibling_land(processed_v2)
        if sibling is not None:
            return sibling, sibling.parent
        return None, None
    try:
        return ensure_world_land(land_cache_dir), land_cache_dir
    except Exception as error:
        LOGGER.warning("V2 maps will omit Natural Earth land context: %s", error)
        return None, land_cache_dir


def _existing_sibling_land(processed_v2: Path) -> Path | None:
    sibling_cache = processed_v2.parent / "cache"
    sibling_land = sibling_cache / WORLD_LAND_FILENAME
    if sibling_land.is_file() and sibling_land.stat().st_size > 0:
        return sibling_land
    return None


def _render_coverage(processed_v2: Path, output_dir: Path, land_geojson_path: Path | None) -> Path:
    coverage_path = output_dir / "coverage_map.png"
    lons, lats = load_centroids_from_parquet(processed_v2 / "polygons")
    generate_coverage_map(
        lons,
        lats,
        coverage_path,
        land_geojson_path=land_geojson_path,
        title="V2 dataset coverage",
    )
    return coverage_path


def _render_text_maps(
    processed_v2: Path,
    output_dir: Path,
    land_geojson_path: Path | None,
    land_cache_dir: Path | None,
) -> tuple[Path, Path]:
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
    return presence_path, density_path


__all__ = ["generate_v2_added_wikipedia_tag_map", "generate_v2_map_assets"]
