"""Deterministic Hugging Face repo paths for the OSM/Wikidata dataset.

ALL remote paths published by this codebase MUST be defined here.
Production code MUST NOT scatter literal remote-path strings across
CLI or pipeline code. The single exception is
:data:`LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE`, an explicitly-named
constant used solely by the migration commit that unifies the
augmentation manifest under ``manifests/``.
"""

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

# Canonical remote augmentation manifest path. Lives next to the
# core manifest under ``manifests/`` so the dataset has ONE
# ``manifests/`` directory and no obsolete ``augmentation/`` tree.
REMOTE_AUGMENTATION_MANIFEST_FILE = "manifests/augmentation_manifest.json"

# Legacy remote augmentation manifest path. Named explicitly so the
# first publication after this change can DELETE it in the SAME
# atomic Hub commit as the canonical upload. After one publication
# succeeds this path is no longer touched; the canonical reference
# is :data:`REMOTE_AUGMENTATION_MANIFEST_FILE`.
LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE = "augmentation/manifests/augmentation_manifest.json"

# The coverage map PNG embedded in the dataset README.
REMOTE_COVERAGE_MAP_FILE = "assets/coverage_map.png"
LEGACY_REMOTE_COVERAGE_MAP_FILE = "coverage_map.png"

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
    "LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE",
    "LEGACY_REMOTE_COVERAGE_MAP_FILE",
    "REMOTE_ARTICLES_DIR",
    "REMOTE_AUGMENTATION_MANIFEST_FILE",
    "REMOTE_COVERAGE_MAP_FILE",
    "REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE",
    "REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE",
    "REMOTE_LINKS_DIR",
    "REMOTE_MANIFESTS_DIR",
    "REMOTE_MANIFEST_FILE",
    "REMOTE_POLYGONS_DIR",
    "local_to_remote",
    "remote_dataset_card_path",
    "remote_parquet_path",
]
