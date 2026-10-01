"""Seeded synthetic inputs for the workload benchmarks.

Everything is generated from a fixed seed under a caller-supplied directory, so
the benchmarks never touch real data and every run measures the same input.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import DOCUMENT_COLUMNS, document_schema
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.hf import coverage_map

SEED = 20260926
_BATCH_ROWS = 10_000
_BODY = "body"
_CONTENT_HASH = hashlib.sha256(_BODY.encode()).hexdigest()


def land_geojson() -> Path:
    return Path(coverage_map.__file__).with_name("ne_110m_land.geojson")


def random_points(count: int) -> tuple[list[float], list[float]]:
    """Return ``count`` seeded (longitudes, latitudes) over the whole globe."""
    rng = random.Random(SEED)
    lons = [rng.uniform(-180.0, 180.0) for _ in range(count)]
    lats = [rng.uniform(-85.0, 85.0) for _ in range(count)]
    return lons, lats


def language_batches(rows: int, languages: int, batch_rows: int) -> Iterator[pa.RecordBatch]:
    """Yield ``rows`` seeded rows whose ``language`` is one of ``languages`` codes."""
    rng = random.Random(SEED)
    codes = [f"{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(languages)]
    for start in range(0, rows, batch_rows):
        size = min(batch_rows, rows - start)
        yield pa.record_batch(
            {
                "document_id": pa.array([f"d{start + i}" for i in range(size)]),
                "language": pa.array([codes[rng.randrange(languages)] for _ in range(size)]),
            }
        )


def _polygon_row(index: int, stem: str) -> dict[str, Any]:
    qid = f"Q{index + 1}"
    return {
        "polygon_id": f"{stem}:relation:{index}",
        "region": stem,
        "source_pbf": f"{stem}.osm.pbf",
        "osm_type": "relation",
        "osm_id": index,
        "wikidata": qid,
        "name": "",
        "tags": json.dumps({"wikidata": qid}),
        "tag_keys": json.dumps(["wikidata"]),
        "tag_count": 1,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [0.0, 0.0]}),
        "lat": 0.0,
        "lon": 0.0,
        "bbox": json.dumps([0.0, 0.0, 0.0, 0.0]),
        "geometry": "",
        "area_m2": 0.0,
        "area_km2": 0.0,
        "area_bucket": "0",
        "has_name": False,
        "has_wikidata": True,
        "has_wikipedia": True,
        "wikipedia_language_count": 1,
        "wikipedia_languages": json.dumps(["en"]),
        "wikipedia_article_count": 1,
        "has_english_wikipedia": True,
        "has_french_wikipedia": False,
        "text_available": True,
        "best_language": "en",
        "extraction_version": "benchmark",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def _legacy_link_row(index: int, stem: str) -> dict[str, Any]:
    return {
        "polygon_id": f"{stem}:relation:{index}",
        "article_id": f"Q{index + 1}:en:{index + 1}:1",
        "wikidata": f"Q{index + 1}",
        "language": "en",
        "source_pbf": f"{stem}.osm.pbf",
        "region": stem,
        "osm_type": "relation",
        "osm_id": index,
        "page_id": index + 1,
        "revision_id": 1,
        "is_best_language": True,
    }


def _legacy_document_row(index: int, _stem: str) -> dict[str, Any]:
    return {
        "document_id": f"Q{index + 1}:wikipedia:en:{index + 1}:1",
        "article_id": f"Q{index + 1}:en:{index + 1}:1",
        "wikidata": f"Q{index + 1}",
        "project": "wikipedia",
        "language": "en",
        "site": "enwiki",
        "title": f"T{index}",
        "url": f"https://en.wikipedia.org/wiki/T{index}",
        "page_id": index + 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "full_text": _BODY,
        "full_text_format": "plain_text",
        "article_length_chars": len(_BODY),
        "article_length_words": 1,
        "article_length_tokens_estimate": 1,
        "license": "CC-BY-SA",
        "attribution": "Wikipedia contributors",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": _CONTENT_HASH,
    }


def _write_rows(
    path: Path,
    schema: pa.Schema,
    columns: tuple[str, ...],
    make_row: Callable[[int, str], dict[str, Any]],
    count: int,
    stem: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression="snappy") as writer:
        for start in range(0, count, _BATCH_ROWS):
            stop = min(start + _BATCH_ROWS, count)
            rows = []
            for index in range(start, stop):
                row = make_row(index, stem)
                rows.append({column: row.get(column) for column in columns})
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def write_legacy_link_stem(processed: Path, stem: str, links: int) -> None:
    """Write one migratable legacy stem holding ``links`` polygon/article links."""
    polygon_columns = tuple(polygon_schema().names)
    _write_rows(
        processed / "polygons" / f"{stem}.parquet",
        polygon_schema(),
        polygon_columns,
        _polygon_row,
        links,
        stem,
    )
    _write_rows(
        processed / "polygon_articles" / f"{stem}.parquet",
        polygon_article_schema(),
        tuple(POLYGON_ARTICLE_COLUMNS),
        _legacy_link_row,
        links,
        stem,
    )
    _write_rows(
        processed / "wikipedia" / "documents" / f"{stem}.parquet",
        document_schema(),
        tuple(DOCUMENT_COLUMNS),
        _legacy_document_row,
        links,
        stem,
    )
