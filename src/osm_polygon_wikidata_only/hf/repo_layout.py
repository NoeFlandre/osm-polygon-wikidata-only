"""Deterministic Hugging Face repo paths for the OSM/Wikidata dataset.

ALL remote paths published by this codebase MUST be defined here.
Production code MUST NOT scatter literal remote-path strings across
CLI or pipeline code. Explicitly-named constants:
* :data:`LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE`: used solely by the migration
  commit that unifies the augmentation manifest under ``manifests/``.
* :data:`LEGACY_REMOTE_COVERAGE_MAP_FILE`: used solely by the migration
  commit that unifies the coverage map under ``assets/``.
* :data:`LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE` and
  :data:`LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE`: used solely by
  the migration commit that replaces the two old H3 maps with combined
  Wikipedia/Wikivoyage text density.
"""

from __future__ import annotations

from pathlib import Path

# Top-level directories inside the HF dataset repo. These must
# match the column lists in ``domain.schema``.
REMOTE_POLYGONS_DIR = "polygons"
# Canonical Wikipedia corpus. ``articles/`` is retained only as the
# explicitly named legacy path during the atomic remote migration.
REMOTE_WIKIPEDIA_DOCUMENTS_DIR = "wikipedia/documents"
LEGACY_REMOTE_ARTICLES_DIR = "articles"
# Backward-compatible import alias. New publication code must use the
# explicitly named legacy constant instead.
REMOTE_ARTICLES_DIR = LEGACY_REMOTE_ARTICLES_DIR
REMOTE_LINKS_DIR = "polygon_articles"
REMOTE_MANIFESTS_DIR = "manifests"

# The per-file manifest, one entry per source PBF.
REMOTE_MANIFEST_FILE = "manifests/processed_pbfs.json"

# Canonical remote augmentation manifest path. Lives next to the
# core manifest under ``manifests/`` so the dataset has ONE
# ``manifests/`` directory and no obsolete ``augmentation/`` tree.
REMOTE_AUGMENTATION_MANIFEST_FILE = "manifests/augmentation_manifest.json"
REMOTE_CONTAINMENT_RETIREMENT_FILE = "manifests/containment_retirements.json"

# Legacy remote augmentation manifest path. Named explicitly so the
# first publication after this change can DELETE it in the SAME
# atomic Hub commit as the canonical upload. After one publication
# succeeds this path is no longer touched; the canonical reference
# is :data:`REMOTE_AUGMENTATION_MANIFEST_FILE`.
LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE = "augmentation/manifests/augmentation_manifest.json"

# The coverage map PNG embedded in the dataset README.
REMOTE_COVERAGE_MAP_FILE = "assets/coverage_map.png"
LEGACY_REMOTE_COVERAGE_MAP_FILE = "coverage_map.png"

# Superseded geographic Wikipedia coverage path, retained for compatibility
# imports and its atomic remote deletion.
REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE = "assets/geographic_wikipedia_text_coverage.png"
LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE = REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE

# Superseded all-polygon H3 density path, retained for compatibility imports
# and its atomic remote deletion.
REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE = "assets/geographic_polygon_count.png"
LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE = REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE

# Every polygon with non-empty Wikipedia or Wikivoyage document text.
REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE = "assets/geographic_text_presence.png"

# Canonical H3 density of polygons with Wikipedia or Wikivoyage text.
REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE = "assets/geographic_text_density.png"


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


def canonical_region_paths(stem: str) -> dict[str, str]:
    """Return mapping of expected local relative paths (under processed/) to remote paths for a region."""
    return {
        f"polygons/{stem}.parquet": f"polygons/{stem}.parquet",
        f"polygon_articles/{stem}.parquet": f"polygon_articles/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet": f"wikipedia/documents/{stem}.parquet",
        f"wikipedia/sections/{stem}.parquet": f"wikipedia/sections/{stem}.parquet",
        f"wikivoyage/documents/{stem}.parquet": f"wikivoyage/documents/{stem}.parquet",
        f"wikivoyage/sections/{stem}.parquet": f"wikivoyage/sections/{stem}.parquet",
        f"wikidata/facts/{stem}.parquet": f"wikidata/facts/{stem}.parquet",
    }


__all__ = [
    "LEGACY_REMOTE_ARTICLES_DIR",
    "LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE",
    "LEGACY_REMOTE_COVERAGE_MAP_FILE",
    "LEGACY_REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE",
    "LEGACY_REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE",
    "REMOTE_ARTICLES_DIR",
    "REMOTE_AUGMENTATION_MANIFEST_FILE",
    "REMOTE_CONTAINMENT_RETIREMENT_FILE",
    "REMOTE_COVERAGE_MAP_FILE",
    "REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE",
    "REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE",
    "REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE",
    "REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE",
    "REMOTE_LINKS_DIR",
    "REMOTE_MANIFESTS_DIR",
    "REMOTE_MANIFEST_FILE",
    "REMOTE_POLYGONS_DIR",
    "REMOTE_WIKIPEDIA_DOCUMENTS_DIR",
    "canonical_region_paths",
    "local_to_remote",
    "remote_dataset_card_path",
    "remote_parquet_path",
]
