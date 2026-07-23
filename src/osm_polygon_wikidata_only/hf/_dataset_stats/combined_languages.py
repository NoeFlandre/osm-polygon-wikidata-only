"""Combined Wikipedia and Wikivoyage language statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf._geographic.parquet_inputs import sorted_parquets
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
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != _CACHE_CONTRACT_VERSION
        or payload.get("fingerprints") != [list(item) for item in fingerprints]
    ):
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    try:
        return CombinedLanguageStats(
            document_count=int(stats["document_count"]),
            language_count=int(stats["language_count"]),
            documents_per_language=tuple(
                (str(language), int(count)) for language, count in stats["documents_per_language"]
            ),
            polygons_per_language=tuple(
                (str(language), int(count)) for language, count in stats["polygons_per_language"]
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


def compute_combined_language_stats(
    processed_root: Path,
    *,
    cache_index_dir: Path | None = None,
) -> CombinedLanguageStats:
    """Compute factual cross-project document and polygon language counts."""
    fingerprints = _input_fingerprints(processed_root)
    cache_path = Path(cache_index_dir) / _CACHE_FILE if cache_index_dir is not None else None
    if cache_path is not None:
        cached = _load_cached(cache_path, fingerprints)
        if cached is not None:
            return cached

    documents: set[tuple[str, str]] = set()
    document_counts: Counter[str] = Counter()
    wikipedia_text_languages: dict[str, str] = {}
    for path in sorted_parquets(processed_root / "wikipedia" / "documents"):
        for row in _read_available(path, ("document_id", "article_id", "language", "full_text")):
            document_id = str(row.get("document_id") or row.get("article_id") or "")
            language = str(row.get("language") or "")
            identity = ("wikipedia", document_id)
            if document_id and language and identity not in documents:
                documents.add(identity)
                document_counts[language] += 1
            article_id = str(row.get("article_id") or "")
            if article_id and language and _non_blank(row.get("full_text")):
                wikipedia_text_languages[article_id] = language

    polygons_by_language: dict[str, set[str]] = defaultdict(set)
    for path in sorted_parquets(processed_root / "polygon_articles"):
        for row in _read_available(path, ("polygon_id", "article_id")):
            linked_language = wikipedia_text_languages.get(str(row.get("article_id") or ""))
            polygon_id = str(row.get("polygon_id") or "")
            if linked_language and polygon_id:
                polygons_by_language[linked_language].add(polygon_id)

    voyage_qid_languages: dict[str, set[str]] = defaultdict(set)
    for path in sorted_parquets(processed_root / "wikivoyage" / "documents"):
        for row in _read_available(path, ("document_id", "wikidata", "language", "full_text")):
            document_id = str(row.get("document_id") or "")
            language = str(row.get("language") or "")
            identity = ("wikivoyage", document_id)
            if document_id and language and identity not in documents:
                documents.add(identity)
                document_counts[language] += 1
            qid = str(row.get("wikidata") or "")
            if qid and language and _non_blank(row.get("full_text")):
                voyage_qid_languages[qid].add(language)

    for path in sorted_parquets(processed_root / "polygons"):
        for row in _read_available(path, ("polygon_id", "wikidata")):
            polygon_id = str(row.get("polygon_id") or "")
            for language in voyage_qid_languages.get(str(row.get("wikidata") or ""), ()):
                if polygon_id:
                    polygons_by_language[language].add(polygon_id)

    polygon_counts = Counter(
        {language: len(polygon_ids) for language, polygon_ids in polygons_by_language.items()}
    )
    result = CombinedLanguageStats(
        document_count=len(documents),
        language_count=len(document_counts),
        documents_per_language=_sorted_counts(document_counts),
        polygons_per_language=_sorted_counts(polygon_counts),
    )
    if cache_path is not None:
        _write_cached(cache_path, fingerprints, result)
    return result


__all__ = ["compute_combined_language_stats"]
