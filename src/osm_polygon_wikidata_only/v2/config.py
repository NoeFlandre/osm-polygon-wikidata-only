"""Explicit V2 dataset constants and version selection."""

from __future__ import annotations

from enum import StrEnum

V2_CONTRACT_VERSION = "wikipedia-tags-v2"
V2_CACHE_CONTRACT_VERSION = "wikipedia-tags-v2-cache"
V2_REPO_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"


class DatasetVersion(StrEnum):
    """Published dataset contracts supported by the application."""

    V1 = "v1"
    V2 = "v2"
