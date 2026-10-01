"""Dataset-card (README) snapshot rendering for the V1 publication."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.integrity import INTEGRITY_CONTRACT_VERSION
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._dataset_stats.augmentation import compute_augmentation_stats
from osm_polygon_wikidata_only.hf.continent_stats import compute_continent_stats
from osm_polygon_wikidata_only.hf.coverage_map import ensure_world_countries
from osm_polygon_wikidata_only.hf.dataset_card import render_front_matter
from osm_polygon_wikidata_only.hf.dataset_stats import compute_dataset_stats
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    TextPresenceSnapshot,
    load_text_presence,
)
from osm_polygon_wikidata_only.hf.minimal_card import (
    MinimalCardSnapshot,
    continent_coverage_rows,
    render_minimal_card,
)
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import load_polygon_geometry_stats
from osm_polygon_wikidata_only.io.atomic import atomic_write_text


@dataclass(frozen=True, slots=True)
class MinimalV1ReleaseSnapshot:
    """Prepared V1 card values shared by the card, report, and maps."""

    card: MinimalCardSnapshot
    report_extra: dict[str, object]
    text_presence: TextPresenceSnapshot


def build_minimal_v1_release_snapshot(
    data_root: DataRoot,
    repo_id: str,
    *,
    generated_on: str | None = None,
    text_presence: TextPresenceSnapshot | None = None,
) -> MinimalV1ReleaseSnapshot:
    """Compute the V1 public summary once for all release artifacts."""
    core_stats = compute_dataset_stats(data_root.processed)
    augmentation_stats = compute_augmentation_stats(
        data_root.processed,
        cache_index_dir=data_root.cache,
    )
    geometry_stats = load_polygon_geometry_stats(data_root.processed)
    presence = text_presence or load_text_presence(data_root.processed)
    countries_path = ensure_world_countries(data_root.cache)
    continent_rows = compute_continent_stats(data_root.processed, countries_path)
    front_matter = render_front_matter(
        repo_id=repo_id,
        license="odbl",
        primary_lang="en",
        polygon_count=core_stats.polygon_count,
        article_count=core_stats.article_count,
        unique_wikidata_count=core_stats.unique_wikidata_count,
    )
    rows = continent_coverage_rows(continent_rows)
    documents, sections, languages = _corpus_totals(augmentation_stats, core_stats)
    snapshot = MinimalCardSnapshot(
        front_matter=front_matter,
        repo_id=repo_id,
        title="OSM Polygon Wikidata, Wikipedia and Wikivoyage",
        description=(
            "OSM polygons carrying `wikidata=*`, enriched with multilingual Wikipedia "
            "and Wikivoyage documents. The published tables preserve regional records "
            "and provenance."
        ),
        polygon_rows=core_stats.polygon_count,
        unique_polygon_identities=core_stats.unique_polygon_identities or presence.polygon_count,
        polygons_with_text=len(presence.combined_polygon_identities),
        documents=documents,
        sections=sections,
        languages=languages,
        regions=core_stats.region_count,
        total_parquet_bytes=augmentation_stats.total_parquet_bytes,
        total_area_m2=geometry_stats.area.total_m2,
        median_area_m2=geometry_stats.area.median_m2,
        document_words=(
            augmentation_stats.wikipedia_documents.total_words
            + augmentation_stats.wikivoyage_documents.total_words
        ),
        continent_rows=rows,
        generated_on=generated_on,
        viewer_url=f"https://huggingface.co/datasets/{repo_id}/viewer",
    )
    report_extra = {
        "card_contract": "minimal-v1",
        "dataset_stats": asdict(core_stats),
        "augmentation_stats": asdict(augmentation_stats),
        "continent_stats": [asdict(row) for row in rows],
        "join_integrity_audit": integrity_audit(data_root),
        "text_coverage": {
            "polygon_identities": presence.polygon_count,
            "wikipedia_text_polygon_identities": len(presence.wikipedia_polygon_identities),
            "combined_text_polygon_identities": len(presence.combined_polygon_identities),
            "wikipedia_documents": len(presence.wikipedia_document_ids),
            "wikivoyage_documents": len(presence.wikivoyage_document_ids),
        },
    }
    return MinimalV1ReleaseSnapshot(snapshot, report_extra, presence)


def integrity_audit(data_root: DataRoot) -> dict[str, Any] | None:
    """Return the join-integrity audit payload published in ``stats.json``.

    The compact card no longer renders the audit table, so the report is the
    only place these counts remain available.
    """
    audit_path = data_root.processed / "integrity" / "integrity_audit.json"
    if not audit_path.is_file():
        return None
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        **payload,
        "contract_version": str(payload.get("contract_version", INTEGRITY_CONTRACT_VERSION)),
    }


def _corpus_totals(augmentation_stats: Any, core_stats: Any) -> tuple[int, int, int]:
    """Return the document, section, and language totals shown on the V1 card.

    The combined-language index is authoritative when it has been built; the
    per-corpus row counts are the fallback for roots predating it.
    """
    documents = (
        augmentation_stats.combined_languages.document_count
        or augmentation_stats.wikipedia_documents.rows
        + augmentation_stats.wikivoyage_documents.rows
    )
    sections = (
        augmentation_stats.wikipedia_sections.rows + augmentation_stats.wikivoyage_sections.rows
    )
    languages = augmentation_stats.combined_languages.language_count or core_stats.language_count
    return documents, sections, languages


def write_dataset_card_snapshot(
    data_root: DataRoot,
    repo_id: str,
    destination: Path,
    *,
    generated_on: str | None,
) -> None:
    """Render the canonical dataset README from current local artifacts.

    The README is recomputed by:

    1. Computing the core :class:`DatasetStats` snapshot from the
       finalized Parquet tables via
       :func:`compute_dataset_stats`.
    2. Computing the augmentation :class:`AugmentationStats` snapshot
       via :func:`compute_augmentation_stats`. The per-file summary
       cache lives under ``data_root.cache``, so a warm refresh
       performs zero Parquet table reads.
    3. Rendering the polygon surface and geometry block from the same
       snapshot that :func:`write_polygon_stats_snapshot` publishes as
       ``stats.json``, so the card and the report never disagree.
    4. Computing public continent statistics from polygon centroids and
       the bundled Natural Earth Admin-0 reference.
    5. Rendering the public snapshot, Wikipedia and Wikivoyage corpora,
       Wikidata facts, storage accounting, and continent distribution.

    The README must be written AFTER every other snapshot so a
    partial core upload never reaches the Hub. The destination is
    written atomically via
    :func:`osm_polygon_wikidata_only.io.atomic.atomic_write_text`.
    """
    prepared = build_minimal_v1_release_snapshot(data_root, repo_id, generated_on=generated_on)
    atomic_write_text(
        destination,
        render_minimal_card(prepared.card),
    )
