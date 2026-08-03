"""Deterministic public dataset card for the V2 artifact tree."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION, V2_REPO_ID
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest


def render_v2_card(processed_v2: Path) -> str:
    """Render a concise, factual card from the V2 files on disk."""
    manifest = load_v2_manifest(processed_v2)
    polygon_count = _rows(processed_v2 / "polygons", manifest)
    wikipedia_document_count = _rows(processed_v2 / "wikipedia/documents", manifest)
    wikipedia_section_count = _rows(processed_v2 / "wikipedia/sections", manifest)
    wikivoyage_document_count = _rows(processed_v2 / "wikivoyage/documents", manifest)
    wikivoyage_section_count = _rows(processed_v2 / "wikivoyage/sections", manifest)
    wikidata_fact_count = _rows(processed_v2 / "wikidata/facts", manifest)
    link_count = _rows(processed_v2 / "polygon_document_links", manifest)
    wikipedia_only = sum(
        _wikipedia_only_count(processed_v2 / "polygons" / f"{stem}.parquet")
        for stem in manifest
        if (processed_v2 / "polygons" / f"{stem}.parquet").is_file()
    )
    return "\n".join(
        [
            "---",
            f"dataset_info:\n  config_name: default\n  version: {V2_CONTRACT_VERSION}",
            "---",
            "# OSM Polygon Wikidata + Wikipedia",
            "",
            "Version 2 keeps every V1 polygon and adds polygons discovered through valid multilingual OSM `wikipedia=*` tags, including polygons without a Wikidata QID.",
            "",
            "The build reuses finalized V1 documents and sidecars. It fetches only direct Wikipedia pages that are not already present in V1, so reruns are resumable and do not refetch the existing corpus.",
            "",
            "## Snapshot",
            "",
            f"- **Hugging Face dataset:** [{V2_REPO_ID}](https://huggingface.co/datasets/{V2_REPO_ID})",
            f"- **Regions:** {len(manifest):,}",
            f"- **Polygons:** {polygon_count:,}",
            f"- **Wikipedia documents:** {wikipedia_document_count:,}",
            f"- **Wikipedia sections:** {wikipedia_section_count:,}",
            f"- **Wikivoyage documents:** {wikivoyage_document_count:,}",
            f"- **Wikivoyage sections:** {wikivoyage_section_count:,}",
            f"- **Wikidata facts:** {wikidata_fact_count:,}",
            f"- **Polygon-document links:** {link_count:,}",
            f"- **Wikipedia-tag-only polygons:** {wikipedia_only:,}",
            "",
            "## Discovery and provenance",
            "",
            "Each polygon row records `discovery_sources` (`wikidata`, `wikipedia_tag`, or both), normalized multilingual references, and structured rejections. The `polygon_document_links` table stores both V1 Wikidata-sitelink relationships and V2 direct-tag relationships in one schema, with `link_sources` identifying how each relationship was found.",
            "",
            "## Repository layout",
            "",
            "- `polygons/` — one row for every retained OSM polygon.",
            "- `wikipedia/documents/` — canonical 32-column Wikipedia document rows; direct-only rows may have a null `wikidata`.",
            "- `polygon_document_links/` — unified polygon-to-document relationships for Wikipedia and Wikivoyage.",
            "- `wikipedia/sections/` — section-level Wikipedia text using the exact V1 22-column section schema.",
            "- `wikivoyage/documents/`, `wikivoyage/sections/`, `wikidata/facts/` — reused V1 sidecars.",
            "- `manifests/processed_pbfs.json` — versioned, per-region row counts and file hashes.",
            "",
            "## Reproducibility",
            "",
            "The V2 build is selected explicitly with `sync-dir --dataset-version v2`. V1 commands and the V1 artifact tree remain unchanged.",
            "",
        ]
    )


def write_v2_card(processed_v2: Path) -> Path:
    """Write the deterministic card atomically and return its path."""
    path = processed_v2 / "README.md"
    temporary = path.with_suffix(".md.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(render_v2_card(processed_v2), encoding="utf-8")
    temporary.replace(path)
    return path


def _rows(directory: Path, manifest: dict[str, dict]) -> int:
    total = 0
    for stem in sorted(manifest):
        path = directory / f"{stem}.parquet"
        if path.is_file():
            total += pq.read_metadata(path).num_rows
    return total


def _wikipedia_only_count(path: Path) -> int:
    table = pq.read_table(path, columns=["has_wikidata"])
    return sum(1 for value in table.column("has_wikidata").to_pylist() if not value)


__all__ = ["render_v2_card", "write_v2_card"]
