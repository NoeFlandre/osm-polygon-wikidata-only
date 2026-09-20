"""YAML front matter for the V2 dataset card.

The front matter declares the Dataset Viewer configurations, so it is owned by
the release snapshot rather than by the Markdown body renderer. Keeping it in
its own module lets the snapshot builder and the card renderer share it without
importing one another.
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.v2.card_models import V2CardStats
from osm_polygon_wikidata_only.v2.config import (
    V2_CONTRACT_VERSION,
    V2_DATASET_CARD_VERSION,
)

_BASE_CONFIGS: tuple[tuple[str, str, str], ...] = (
    ("polygons", "polygons", "polygons/*.parquet"),
    ("polygon_document_links", "polygon_document_links", "polygon_document_links/*.parquet"),
    ("wikipedia_documents", "wikipedia_documents", "wikipedia/documents/*.parquet"),
    ("wikipedia_sections", "wikipedia_sections", "wikipedia/sections/*.parquet"),
    ("wikivoyage_documents", "wikivoyage_documents", "wikivoyage/documents/*.parquet"),
    ("wikivoyage_sections", "wikivoyage_sections", "wikivoyage/sections/*.parquet"),
    ("wikidata_facts", "wikidata_facts", "wikidata/facts/*.parquet"),
)
_SENTENCE_CONFIGS: tuple[tuple[str, str, str], ...] = (
    ("wikipedia_sentences", "wikipedia_sentences", "wikipedia/sentences/*.parquet"),
    ("wikivoyage_sentences", "wikivoyage_sentences", "wikivoyage/sentences/*.parquet"),
)


def _render_front_matter(snapshot: V2CardStats, *, processed_v2: Path) -> str:
    """Render the card front matter, declaring one config per published table."""
    configs = [
        *_BASE_CONFIGS,
        *(
            config
            for config in _SENTENCE_CONFIGS
            if _has_parquet(processed_v2 / config[2].rsplit("/", 1)[0])
        ),
    ]
    lines = [
        "---",
        "license: odbl",
        "language:",
        "  - en",
        "tags:",
        "  - openstreetmap",
        "  - wikidata",
        "  - wikipedia",
        "  - wikivoyage",
        "  - geospatial",
        "configs:",
    ]
    for config_name, split, path in configs:
        lines.extend(
            [
                f"  - config_name: {config_name}",
                "    data_files:",
                f"      - split: {split}",
                f"        path: {path}",
            ]
        )
    lines.extend(
        [
            "dataset_info:",
            f"  version: {V2_DATASET_CARD_VERSION}",
            f"  regions: {snapshot.regions}",
            f"  polygons: {snapshot.polygons}",
            f"  documents: {snapshot.documents}",
            f"dataset_contract: {V2_CONTRACT_VERSION}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def _has_parquet(directory: Path) -> bool:
    return any(directory.glob("*.parquet"))


# Public collaborator spellings used by release preparation and the facade.
render_front_matter = _render_front_matter
has_parquet = _has_parquet
