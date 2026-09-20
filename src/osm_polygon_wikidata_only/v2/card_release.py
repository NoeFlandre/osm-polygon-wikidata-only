"""Release-card adapter for the shared V2 publication snapshot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.hf._geographic.models import CoverageMapError
from osm_polygon_wikidata_only.hf.continent_stats import compute_continent_stats
from osm_polygon_wikidata_only.hf.coverage_map import ensure_world_countries
from osm_polygon_wikidata_only.hf.geographic_text_presence import (
    TextPresenceSnapshot,
    load_text_presence,
)
from osm_polygon_wikidata_only.hf.minimal_card import (
    ContinentCoverage,
    MinimalCardSnapshot,
    SentenceCoverage,
    continent_coverage_rows,
)
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import load_polygon_geometry_stats
from osm_polygon_wikidata_only.v2.card_front_matter import (
    render_front_matter as _render_front_matter,
)
from osm_polygon_wikidata_only.v2.card_metrics import compute_v2_card_stats
from osm_polygon_wikidata_only.v2.card_models import V2CardStats
from osm_polygon_wikidata_only.v2.config import V1_DATASET_URL, V2_GITHUB_URL, V2_REPO_ID


@dataclass(frozen=True, slots=True)
class MinimalV2ReleaseSnapshot:
    """Prepared V2 card values shared by the card, report, and maps."""

    card: MinimalCardSnapshot
    report_extra: dict[str, object]
    text_presence: TextPresenceSnapshot


def build_minimal_v2_release_snapshot(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    cache_dir: Path | None = None,
    generated_on: str | None = None,
    text_presence: TextPresenceSnapshot | None = None,
    stats: V2CardStats | None = None,
) -> MinimalV2ReleaseSnapshot:
    """Compute the V2 public summary once for all release artifacts."""
    stats = stats or compute_v2_card_stats(processed_v2, v1_processed=v1_processed)
    links_dir = processed_v2 / "polygon_document_links"
    presence = text_presence or _load_presence(processed_v2, links_dir)
    geometry_area = _geometry_area(processed_v2)
    countries_path = ensure_world_countries(cache_dir or processed_v2.parent / "cache")
    rows = _continent_rows(processed_v2, countries_path, links_dir)
    front_matter = _render_front_matter(stats, processed_v2=processed_v2)
    snapshot = MinimalCardSnapshot(
        front_matter=front_matter,
        repo_id=V2_REPO_ID,
        title="OSM Polygon Wikidata + Wikipedia, V2",
        description=(
            f"V2 builds on the [V1 Wikidata-only dataset]({V1_DATASET_URL}) with "
            "multilingual Wikipedia and Wikivoyage text. It also retains valid "
            "multilingual `wikipedia=*` references, including polygons without a "
            "Wikidata QID."
        ),
        polygon_rows=stats.polygons,
        unique_polygon_identities=stats.unique_polygon_identities or presence.polygon_count,
        polygons_with_text=_polygons_with_text(stats, presence),
        documents=stats.documents,
        sections=stats.wikipedia_sections + stats.wikivoyage_sections,
        languages=stats.languages,
        regions=stats.regions,
        total_parquet_bytes=stats.total_parquet_storage_bytes,
        total_area_m2=geometry_area[0],
        median_area_m2=geometry_area[1],
        document_words=stats.document_words,
        sentence_rows=_sentence_rows(stats.sentence_stats),
        sentence_coverage=_sentence_coverage(stats.sentence_stats),
        continent_rows=rows,
        generated_on=generated_on,
        source_url=V2_GITHUB_URL,
        viewer_url=f"https://huggingface.co/datasets/{V2_REPO_ID}/viewer",
    )
    return MinimalV2ReleaseSnapshot(snapshot, _report_extra(stats, rows, presence), presence)


_EMPTY_PRESENCE = TextPresenceSnapshot(
    polygon_count=0,
    wikipedia_covered_polygon_ids=frozenset(),
    combined_covered_polygon_ids=frozenset(),
    wikipedia_document_ids=frozenset(),
    wikivoyage_document_ids=frozenset(),
    covered_points=(),
)


def _load_presence(processed_v2: Path, links_dir: Path) -> TextPresenceSnapshot:
    """Return the text-presence snapshot, or an empty one when it cannot be scanned."""
    try:
        return load_text_presence(processed_v2, links_dir=links_dir)
    except (CoverageMapError, OSError, ValueError):
        return _EMPTY_PRESENCE


def _geometry_area(processed_v2: Path) -> tuple[float | None, float | None]:
    """Return the total and median polygon area, or ``None`` when unavailable."""
    try:
        area = load_polygon_geometry_stats(processed_v2).area
    except (CoverageMapError, OSError, ValueError):
        return None, None
    return area.total_m2, area.median_m2


def _continent_rows(
    processed_v2: Path,
    countries_path: Path,
    links_dir: Path,
) -> tuple[ContinentCoverage, ...]:
    """Return typed continent rows, or none when coverage cannot be computed."""
    try:
        rows = compute_continent_stats(processed_v2, countries_path, links_dir=links_dir)
    except (CoverageMapError, OSError, ValueError):
        return ()
    return continent_coverage_rows(rows)


def _polygons_with_text(stats: Any, presence: TextPresenceSnapshot) -> int:
    """Prefer the scanned text-polygon count, falling back to the presence map."""
    if stats.non_empty_text_polygons is not None:
        return int(stats.non_empty_text_polygons)
    return len(presence.combined_polygon_identities)


def _sentence_rows(stats: Any) -> int | None:
    """Return the sentence-row total, or ``None`` without sentence sidecars."""
    return None if stats is None else int(stats.total_rows)


def _sentence_coverage(stats: Any) -> SentenceCoverage | None:
    if stats is None:
        return None
    return SentenceCoverage(
        eligible_units=stats.eligible_units,
        supported_units=stats.supported_units,
        unsupported_units=stats.unsupported_units,
        top_unsupported_languages=stats.top_unsupported_languages,
    )


def _report_extra(
    stats: Any,
    rows: tuple[ContinentCoverage, ...],
    presence: TextPresenceSnapshot,
) -> dict[str, object]:
    """Assemble the machine-readable report payload for the V2 release."""
    return {
        "card_contract": "minimal-v2",
        "v2_card_stats": _jsonable(asdict(stats)),
        "continent_stats": [asdict(row) for row in rows],
        "text_coverage": {
            "polygon_identities": presence.polygon_count,
            "wikipedia_text_polygon_identities": len(presence.wikipedia_polygon_identities),
            "combined_text_polygon_identities": len(presence.combined_polygon_identities),
            "wikipedia_documents": len(presence.wikipedia_document_ids),
            "wikivoyage_documents": len(presence.wikivoyage_document_ids),
        },
    }


def _jsonable(value: Any) -> Any:
    """Convert nested tuple values from dataclasses to JSON-native values."""
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


jsonable = _jsonable
