"""Combined Wikipedia and Wikivoyage language statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf._geographic.parquet_inputs import sorted_parquets
from osm_polygon_wikidata_only.hf._links.reader import read_document_links
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

from .cache import _file_fingerprint
from .models import CombinedLanguageStats

_CACHE_CONTRACT_VERSION = "combined-languages-v1"
_CACHE_FILE = "combined_languages.json"
_INPUT_SUBDIRS = (
    "polygons",
    "polygon_articles",
    "wikipedia/documents",
    "wikivoyage/documents",
)


def _non_blank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _read_available(path: Path, columns: tuple[str, ...]) -> list[dict[str, object]]:
    """Read the requested columns that exist in a structurally valid Parquet."""
    try:
        available = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid):
        return []
    selected = [column for column in columns if column in available]
    if not selected:
        return []
    return pq.read_table(path, columns=selected).to_pylist()  # type: ignore[no-untyped-call,no-any-return]


def _input_fingerprints(processed_root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            path.relative_to(processed_root).as_posix(),
            _file_fingerprint(path),
        )
        for subdir in _INPUT_SUBDIRS
        for path in sorted_parquets(processed_root / subdir)
    )


def _load_cached(
    cache_path: Path,
    fingerprints: tuple[tuple[str, str], ...],
) -> CombinedLanguageStats | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not _cache_matches(payload, fingerprints):
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    return _parse_cached_stats(stats)


def _cache_matches(
    payload: object,
    fingerprints: tuple[tuple[str, str], ...],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("contract_version") != _CACHE_CONTRACT_VERSION:
        return False
    return payload.get("fingerprints") == [list(item) for item in fingerprints]


def _parse_cached_stats(stats: dict[str, object]) -> CombinedLanguageStats | None:
    try:
        return CombinedLanguageStats(
            document_count=int(cast(Any, stats["document_count"])),
            language_count=int(cast(Any, stats["language_count"])),
            documents_per_language=tuple(
                (str(language), int(count))
                for language, count in cast(Any, stats["documents_per_language"])
            ),
            polygons_per_language=tuple(
                (str(language), int(count))
                for language, count in cast(Any, stats["polygons_per_language"])
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_cached(
    cache_path: Path,
    fingerprints: tuple[tuple[str, str], ...],
    stats: CombinedLanguageStats,
) -> None:
    payload = {
        "contract_version": _CACHE_CONTRACT_VERSION,
        "fingerprints": fingerprints,
        "stats": {
            "document_count": stats.document_count,
            "language_count": stats.language_count,
            "documents_per_language": stats.documents_per_language,
            "polygons_per_language": stats.polygons_per_language,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cache_path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _record_document_row(
    row: dict[str, object],
    *,
    project: str,
    documents: set[tuple[str, str]],
    document_counts: Counter[str],
    text_languages: dict[tuple[str, str], str],
    voyage_qid_languages: dict[str, set[str]] | None,
) -> None:
    values = _document_row_values(row, project)
    if values is None:
        return
    identity, language = values
    _record_document_identity(identity, language, documents, document_counts)
    _record_document_text(
        row,
        identity,
        language,
        text_languages,
        voyage_qid_languages,
    )


def _document_row_values(
    row: dict[str, object],
    project: str,
) -> tuple[tuple[str, str], str] | None:
    document_id = _document_id(row)
    language = str(row.get("language") or "")
    if not document_id or not language:
        return None
    return (project, document_id), language


def _document_id(row: dict[str, object]) -> str:
    value = row.get("document_id")
    if value:
        return str(value)
    return str(row.get("article_id") or "")


def _record_document_identity(
    identity: tuple[str, str],
    language: str,
    documents: set[tuple[str, str]],
    document_counts: Counter[str],
) -> None:
    if identity not in documents:
        documents.add(identity)
        document_counts[language] += 1


def _record_document_text(
    row: dict[str, object],
    identity: tuple[str, str],
    language: str,
    text_languages: dict[tuple[str, str], str],
    voyage_qid_languages: dict[str, set[str]] | None,
) -> None:
    if not _non_blank(row.get("full_text")):
        return
    text_languages[identity] = language
    if voyage_qid_languages is None:
        return
    qid = str(row.get("wikidata") or "")
    if qid:
        voyage_qid_languages[qid].add(language)


def _read_document_project(
    processed_root: Path,
    *,
    project: str,
    subdir: str,
    documents: set[tuple[str, str]],
    document_counts: Counter[str],
    text_languages: dict[tuple[str, str], str],
    voyage_qid_languages: dict[str, set[str]] | None = None,
) -> None:
    columns = ("document_id", "article_id", "wikidata", "language", "full_text")
    for path in sorted_parquets(processed_root / subdir):
        for row in _read_available(path, columns):
            _record_document_row(
                row,
                project=project,
                documents=documents,
                document_counts=document_counts,
                text_languages=text_languages,
                voyage_qid_languages=voyage_qid_languages,
            )


def _polygon_languages_from_links(
    processed_root: Path,
    text_languages: dict[tuple[str, str], str],
) -> dict[str, set[str]]:
    polygons_by_language: dict[str, set[str]] = defaultdict(set)
    for link in read_document_links(processed_root):
        linked_language = text_languages.get((link.project, link.document_id))
        if linked_language and link.polygon_id:
            polygons_by_language[linked_language].add(link.polygon_id)
    return polygons_by_language


def _has_canonical_links(processed_root: Path) -> bool:
    from osm_polygon_wikidata_only.domain.polygon_document_links import (
        polygon_document_link_schema,
    )

    schema = polygon_document_link_schema()
    return any(
        pq.read_schema(path).equals(schema, check_metadata=True)  # type: ignore[no-untyped-call]
        for path in sorted_parquets(processed_root / "polygon_articles")
    )


def _record_fallback_polygon(
    row: dict[str, object],
    voyage_qid_languages: dict[str, set[str]],
    polygons_by_language: dict[str, set[str]],
) -> None:
    polygon_id = str(row.get("polygon_id") or "")
    if not polygon_id:
        return
    for language in voyage_qid_languages.get(str(row.get("wikidata") or ""), ()):
        polygons_by_language[language].add(polygon_id)


def _add_voyage_polygon_fallback(
    processed_root: Path,
    voyage_qid_languages: dict[str, set[str]],
    polygons_by_language: dict[str, set[str]],
) -> None:
    for path in sorted_parquets(processed_root / "polygons"):
        for row in _read_available(path, ("polygon_id", "wikidata")):
            _record_fallback_polygon(row, voyage_qid_languages, polygons_by_language)


def compute_combined_language_stats(
    processed_root: Path,
    *,
    cache_index_dir: Path | None = None,
) -> CombinedLanguageStats:
    """Compute factual cross-project document and polygon language counts."""
    fingerprints = _input_fingerprints(processed_root)
    cache_path = Path(cache_index_dir) / _CACHE_FILE if cache_index_dir is not None else None
    cached = _load_cached_result(cache_path, fingerprints)
    if cached is not None:
        return cached

    result = _compute_uncached_stats(processed_root)
    if cache_path is not None:
        _write_cached(cache_path, fingerprints, result)
    return result


def _load_cached_result(
    cache_path: Path | None,
    fingerprints: tuple[tuple[str, str], ...],
) -> CombinedLanguageStats | None:
    if cache_path is None:
        return None
    return _load_cached(cache_path, fingerprints)


def _compute_uncached_stats(processed_root: Path) -> CombinedLanguageStats:
    """Compute statistics without consulting or updating the cache."""

    documents: set[tuple[str, str]] = set()
    document_counts: Counter[str] = Counter()
    text_languages: dict[tuple[str, str], str] = {}
    voyage_qid_languages: dict[str, set[str]] = defaultdict(set)
    _read_document_project(
        processed_root,
        project="wikipedia",
        subdir="wikipedia/documents",
        documents=documents,
        document_counts=document_counts,
        text_languages=text_languages,
    )
    _read_document_project(
        processed_root,
        project="wikivoyage",
        subdir="wikivoyage/documents",
        documents=documents,
        document_counts=document_counts,
        text_languages=text_languages,
        voyage_qid_languages=voyage_qid_languages,
    )

    polygons_by_language = _polygon_languages_from_links(processed_root, text_languages)
    has_canonical_links = _has_canonical_links(processed_root)
    if not has_canonical_links:
        _add_voyage_polygon_fallback(
            processed_root,
            voyage_qid_languages,
            polygons_by_language,
        )

    polygon_counts = Counter(
        {language: len(polygon_ids) for language, polygon_ids in polygons_by_language.items()}
    )
    result = CombinedLanguageStats(
        document_count=len(documents),
        language_count=len(document_counts),
        documents_per_language=_sorted_counts(document_counts),
        polygons_per_language=_sorted_counts(polygon_counts),
    )
    return result


__all__ = ["compute_combined_language_stats"]
