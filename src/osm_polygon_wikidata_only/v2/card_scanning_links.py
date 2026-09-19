"""Link and text-presence scans used by the V2 card metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet

_SUCCESSFUL_FETCH_STATUS = "ok"
_TEXT_DOCUMENT_PROJECTS = frozenset({"wikipedia", "wikivoyage"})


def _text_metrics_from_scanned(
    document_languages: dict[tuple[str, str], str],
    wikipedia_language_counts: Counter[str],
    link_paths: Iterable[Path],
    polygon_index: PolygonIndex,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Build text metrics from document columns already scanned once."""
    languages_by_polygon = _polygon_languages(link_paths, document_languages, polygon_index)
    return (
        _text_funnel(languages_by_polygon),
        tuple(wikipedia_language_counts.most_common(10)),
    )


def _count_linked_non_empty_text_polygons(
    paths: Iterable[Path],
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
) -> int:
    """Count unique OSM identities linked to successful non-empty documents."""
    polygon_identities: set[PolygonIdentity] = set()
    if not eligible_document_keys:
        return 0
    for path in paths:
        _collect_linked_non_empty_text_polygons(
            path,
            eligible_document_keys,
            polygon_index,
            polygon_identities,
        )
    return len(polygon_identities)


def _collect_linked_non_empty_text_polygons(
    path: Path,
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
    polygon_identities: set[PolygonIdentity],
) -> None:
    with open_parquet(path) as parquet_file:
        columns = {"polygon_id", "document_id", "project"}
        if not columns.issubset(parquet_file.schema_arrow.names):
            return
        for batch in iter_record_batches(
            parquet_file,
            columns=["polygon_id", "document_id", "project"],
            batch_size=65_536,
        ):
            _merge_linked_non_empty_text_polygons(
                batch,
                eligible_document_keys,
                polygon_index,
                polygon_identities,
            )


def _merge_linked_non_empty_text_polygons(
    batch: Any,
    eligible_document_keys: set[tuple[str, str]],
    polygon_index: PolygonIndex,
    polygon_identities: set[PolygonIdentity],
) -> None:
    for polygon_id, document_id, project in zip(
        batch.column(0).to_pylist(),
        batch.column(1).to_pylist(),
        batch.column(2).to_pylist(),
        strict=True,
    ):
        document_key = (str(project), str(document_id))
        if document_key not in eligible_document_keys:
            continue
        identity = polygon_index.by_polygon_id.get(str(polygon_id))
        if identity is not None:
            polygon_identities.add(identity)


def _polygon_languages(
    paths: Iterable[Path],
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
) -> defaultdict[PolygonIdentity, set[str]]:
    languages_by_polygon: defaultdict[PolygonIdentity, set[str]] = defaultdict(set)
    for path in paths:
        languages_by_path = _polygon_languages_file(path, document_languages, polygon_index)
        for identity, languages in languages_by_path.items():
            languages_by_polygon[identity].update(languages)
    return languages_by_polygon


def _polygon_languages_file(
    path: Path,
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
) -> dict[PolygonIdentity, set[str]]:
    values: defaultdict[PolygonIdentity, set[str]] = defaultdict(set)
    with open_parquet(path) as parquet_file:
        if not {"polygon_id", "document_id"}.issubset(parquet_file.schema_arrow.names):
            return values
        columns = ["polygon_id", "document_id"]
        has_project = "project" in parquet_file.schema_arrow.names
        if has_project:
            columns.append("project")
        for batch in iter_record_batches(parquet_file, columns=columns, batch_size=65_536):
            _merge_polygon_languages(
                values,
                batch,
                document_languages,
                polygon_index,
                has_project=has_project,
            )
    return values


def _merge_polygon_languages(
    values: defaultdict[PolygonIdentity, set[str]],
    batch: Any,
    document_languages: dict[tuple[str, str], str],
    polygon_index: PolygonIndex,
    *,
    has_project: bool,
) -> None:
    projects = batch.column(2).to_pylist() if has_project else ["wikipedia"] * batch.num_rows
    for polygon_id, document_id, project in zip(
        batch.column(0).to_pylist(),
        batch.column(1).to_pylist(),
        projects,
        strict=True,
    ):
        identity = polygon_index.by_polygon_id.get(str(polygon_id))
        language = document_languages.get((str(project), str(document_id)))
        if identity is not None and language is not None:
            values[identity].add(language)


def _text_funnel(
    languages_by_polygon: dict[PolygonIdentity, set[str]],
) -> tuple[tuple[str, int], ...]:
    all_text = len(languages_by_polygon)
    english = sum("en" in languages for languages in languages_by_polygon.values())
    return (
        ("All polygons", 0),
        ("With non-empty text", all_text),
        ("English coverage", english),
        ("Non-English-only coverage", all_text - english),
        ("2+ languages", sum(len(languages) >= 2 for languages in languages_by_polygon.values())),
        ("5+ languages", sum(len(languages) >= 5 for languages in languages_by_polygon.values())),
        ("10+ languages", sum(len(languages) >= 10 for languages in languages_by_polygon.values())),
    )


# Public collaborator spellings for metric assembly; private names remain
# available through the compatibility facade.
collect_linked_non_empty_text_polygons = _collect_linked_non_empty_text_polygons
count_linked_non_empty_text_polygons = _count_linked_non_empty_text_polygons
merge_linked_non_empty_text_polygons = _merge_linked_non_empty_text_polygons
merge_polygon_languages = _merge_polygon_languages
polygon_languages = _polygon_languages
polygon_languages_file = _polygon_languages_file
text_funnel = _text_funnel
text_metrics_from_scanned = _text_metrics_from_scanned


__all__ = [
    "_collect_linked_non_empty_text_polygons",
    "_count_linked_non_empty_text_polygons",
    "_merge_linked_non_empty_text_polygons",
    "_merge_polygon_languages",
    "_polygon_languages",
    "_polygon_languages_file",
    "_text_funnel",
    "_text_metrics_from_scanned",
]
