"""Explicit V2 dataset constants and version selection."""

from __future__ import annotations

from enum import StrEnum

V2_CONTRACT_VERSION = "wikipedia-tags-v2"
# Hugging Face Dataset Viewer requires ``dataset_info.version`` to use
# semantic-version syntax.  Keep the storage contract above separate from
# this public card version so the viewer can load the dataset reliably.
V2_DATASET_CARD_VERSION = "2.0.0"
V2_CACHE_CONTRACT_VERSION = "wikipedia-tags-v2-cache"
V2_REPO_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
V2_TRACKIO_PROJECT = "osm-polygon-wikidata-and-wikipedia"
V2_TRACKIO_RUN_NAME = "final-dataset-snapshot-v2"
V2_TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia-trackio"
V2_TRACKIO_DATASET_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia-trackio-data"
V2_TRACKIO_SPACE_URL = f"https://huggingface.co/spaces/{V2_TRACKIO_SPACE_ID}"
V2_GITHUB_URL = "https://github.com/NoeFlandre/osm-polygon-wikidata-only"
V1_REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"
V1_DATASET_URL = f"https://huggingface.co/datasets/{V1_REPO_ID}"
V2_ASSET_PATHS: tuple[str, ...] = (
    "assets/coverage_map.png",
    "assets/geographic_text_presence.png",
    "assets/geographic_text_density.png",
)


class DatasetVersion(StrEnum):
    """Published dataset contracts supported by the application."""

    V1 = "v1"
    V2 = "v2"


__all__ = [
    "V1_DATASET_URL",
    "V1_REPO_ID",
    "V2_ASSET_PATHS",
    "V2_CACHE_CONTRACT_VERSION",
    "V2_CONTRACT_VERSION",
    "V2_DATASET_CARD_VERSION",
    "V2_GITHUB_URL",
    "V2_REPO_ID",
    "V2_TRACKIO_DATASET_ID",
    "V2_TRACKIO_PROJECT",
    "V2_TRACKIO_RUN_NAME",
    "V2_TRACKIO_SPACE_ID",
    "V2_TRACKIO_SPACE_URL",
    "DatasetVersion",
]
