from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
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


def test_combined_languages_deduplicates_successful_text_by_typed_osm_identity(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    polygons = processed / "polygons"
    wikipedia_documents = processed / "wikipedia" / "documents"
    wikivoyage_documents = processed / "wikivoyage" / "documents"
    polygon_articles = processed / "polygon_articles"
    for directory in (
        polygons,
        wikipedia_documents,
        wikivoyage_documents,
        polygon_articles,
    ):
        directory.mkdir(parents=True)

    _write(
        polygons / "north.parquet",
        [
            {
                "polygon_id": "north:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
            },
            {
                "polygon_id": "relation:9",
                "osm_type": "relation",
                "osm_id": 9,
                "wikidata": "Q9",
            },
            {
                "polygon_id": "relation:10",
                "osm_type": "relation",
                "osm_id": 10,
                "wikidata": "Q10",
            },
        ],
    )
    _write(
        polygons / "south.parquet",
        [
            {
                "polygon_id": "south:way:7",
                "osm_type": "way",
                "osm_id": 7,
                "wikidata": "Q7",
            }
        ],
    )
    _write(
        wikipedia_documents / "docs.parquet",
        [
            {
                "document_id": "wiki-7",
                "language": "en",
                "full_text": "Successful article",
                "fetch_status": "ok",
            },
            {
                "document_id": "wiki-10-failed",
                "language": "fr",
                "full_text": "Failed article should not qualify",
                "fetch_status": "http_error",
            },
        ],
    )
    _write(
        wikivoyage_documents / "docs.parquet",
        [
            {
                "document_id": "voy-9",
                "wikidata": "Q9",
                "language": "fr",
                "full_text": "Successful guide",
                "fetch_status": "ok",
            }
        ],
    )

    link_schema = polygon_document_link_schema()

    def link_row(
        polygon_id: str,
        document_id: str,
        project: str,
    ) -> dict[str, object]:
        row = {field.name: None for field in link_schema}
        row.update(
            {
                "polygon_id": polygon_id,
                "document_id": document_id,
                "project": project,
            }
        )
        return row

    pq.write_table(
        pa.Table.from_pylist(
            [
                link_row("north:way:7", "wiki-7", "wikipedia"),
                link_row("south:way:7", "wiki-7", "wikipedia"),
                link_row("relation:10", "wiki-10-failed", "wikipedia"),
                link_row("relation:9", "voy-9", "wikivoyage"),
            ],
            schema=link_schema,
        ),
        polygon_articles / "links.parquet",
    )

    stats = compute_combined_language_stats(processed)

    assert stats.document_count == 3
    assert dict(stats.documents_per_language) == {"en": 1, "fr": 2}
    assert stats.polygons_per_language == (("en", 1), ("fr", 1))


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


def test_combined_languages_recomputes_when_cached_stats_are_malformed(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write(
        processed / "wikivoyage" / "documents" / "x.parquet",
        [{"document_id": "v1", "wikidata": "Q1", "language": "en", "full_text": "route"}],
    )
    cache_dir = tmp_path / "cache"
    first = compute_combined_language_stats(processed, cache_index_dir=cache_dir)
    (cache_file,) = cache_dir.rglob("*.json")
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["stats"]["document_count"] = "not-a-number"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    assert compute_combined_language_stats(processed, cache_index_dir=cache_dir) == first
