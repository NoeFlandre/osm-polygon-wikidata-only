from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats.combined_languages import (
    compute_combined_language_stats,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_combined_languages_count_documents_and_unique_polygons(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [
            {"polygon_id": "p1", "wikidata": "Q1"},
            {"polygon_id": "p2", "wikidata": "Q2"},
            {"polygon_id": "p3", "wikidata": "Q2"},
        ],
    )
    _write(
        processed / "wikipedia" / "documents" / "x.parquet",
        [
            {"document_id": "d1", "article_id": "a1", "language": "en", "full_text": "text"},
            {"document_id": "d2", "article_id": "a2", "language": "fr", "full_text": "texte"},
            {"document_id": "d3", "article_id": "a3", "language": "de", "full_text": "   "},
        ],
    )
    _write(
        processed / "polygon_articles" / "x.parquet",
        [
            {"polygon_id": "p1", "article_id": "a1"},
            {"polygon_id": "p2", "article_id": "a2"},
            {"polygon_id": "p3", "article_id": "a3"},
        ],
    )
    _write(
        processed / "wikivoyage" / "documents" / "x.parquet",
        [
            {"document_id": "v1", "wikidata": "Q1", "language": "en", "full_text": "route"},
            {"document_id": "v2", "wikidata": "Q2", "language": "es", "full_text": "viaje"},
            {"document_id": "v3", "wikidata": "Q2", "language": "fr", "full_text": None},
        ],
    )

    stats = compute_combined_language_stats(processed)

    assert stats.document_count == 6
    assert stats.documents_per_language == (("en", 2), ("fr", 2), ("de", 1), ("es", 1))
    assert stats.polygons_per_language == (("es", 2), ("en", 1), ("fr", 1))
    assert stats.language_count == 4


def test_combined_languages_reuses_unchanged_cached_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "polygons" / "x.parquet",
        [{"polygon_id": "p1", "wikidata": "Q1"}],
    )
    _write(
        processed / "wikivoyage" / "documents" / "x.parquet",
        [{"document_id": "v1", "wikidata": "Q1", "language": "en", "full_text": "route"}],
    )
    cache_dir = tmp_path / "cache"

    first = compute_combined_language_stats(processed, cache_index_dir=cache_dir)

    def unexpected_read(*args: object, **kwargs: object) -> object:
        pytest.fail("unchanged inputs must not reread Parquet tables")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf._dataset_stats.combined_languages.pq.read_table",
        unexpected_read,
    )
    second = compute_combined_language_stats(processed, cache_index_dir=cache_dir)

    assert second == first


def test_combined_languages_invalidates_cache_when_input_changes(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    documents = processed / "wikivoyage" / "documents" / "x.parquet"
    _write(
        documents,
        [{"document_id": "v1", "wikidata": "Q1", "language": "en", "full_text": "route"}],
    )
    cache_dir = tmp_path / "cache"

    first = compute_combined_language_stats(processed, cache_index_dir=cache_dir)
    _write(
        documents,
        [
            {"document_id": "v1", "wikidata": "Q1", "language": "en", "full_text": "route"},
            {"document_id": "v2", "wikidata": "Q2", "language": "fr", "full_text": "voyage"},
        ],
    )
    second = compute_combined_language_stats(processed, cache_index_dir=cache_dir)

    assert first.document_count == 1
    assert second.document_count == 2
    assert second.language_count == 2
