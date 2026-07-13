"""Deterministic Hugging Face repo paths for the OSM/Wikidata dataset."""

from __future__ import annotations

from pathlib import Path

# Top-level directories inside the HF dataset repo. These must
# match the column lists in ``domain.schema``.
REMOTE_POLYGONS_DIR = "polygons"
REMOTE_ARTICLES_DIR = "articles"
REMOTE_LINKS_DIR = "polygon_articles"
REMOTE_MANIFESTS_DIR = "manifests"

# The per-file manifest, one entry per source PBF.
REMOTE_MANIFEST_FILE = "manifests/processed_pbfs.json"

# The coverage map PNG embedded in the dataset README.
REMOTE_COVERAGE_MAP_FILE = "coverage_map.png"

# The geographic Wikipedia text coverage PNG embedded in the dataset README.
REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE = "assets/geographic_wikipedia_text_coverage.png"

# The geographic polygon-density PNG embedded in the dataset README.
REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE = "assets/geographic_polygon_count.png"


def remote_parquet_path(subdir: str, stem: str) -> str:
    """Build the deterministic remote path for a per-PBF parquet file."""
    return f"{subdir}/{stem}.parquet"


def remote_dataset_card_path() -> str:
    """Default path for the dataset README card."""
    return "README.md"


def local_to_remote(local_path: Path, processed_subdir: str) -> str:
    """Convert a local path under the processed dir to its remote equivalent."""
    parts = local_path.parts
    # Last two: <subdir>/<stem>.parquet
    return remote_parquet_path(parts[-2], local_path.stem)


__all__ = [
    "REMOTE_ARTICLES_DIR",
    "REMOTE_COVERAGE_MAP_FILE",
    "REMOTE_LINKS_DIR",
    "REMOTE_MANIFESTS_DIR",
    "REMOTE_MANIFEST_FILE",
    "REMOTE_POLYGONS_DIR",
    "local_to_remote",
    "remote_dataset_card_path",
    "remote_parquet_path",
]
