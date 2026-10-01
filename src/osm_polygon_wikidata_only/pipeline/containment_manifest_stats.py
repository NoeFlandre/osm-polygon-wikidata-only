"""Processed-manifest statistics recomputed from staged containment tables."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

POLYGON_MANIFEST_COLUMNS = (
    "wikidata",
    "has_wikipedia",
    "text_available",
    "area_bucket",
    "tag_keys",
)
# PyArrow exposes these C++ kernels dynamically; they are missing from its stubs.
_ARROW_COMPUTE: Any = cast(Any, pc)


def polygon_manifest_table_stats(polygons: pa.Table) -> dict[str, Any]:
    """Aggregate manifest statistics from projected polygon columns."""
    wikidata = polygons["wikidata"]
    nonempty_wikidata = _ARROW_COMPUTE.filter(
        wikidata,
        _ARROW_COMPUTE.and_kleene(
            _ARROW_COMPUTE.is_valid(wikidata),
            _ARROW_COMPUTE.not_equal(wikidata, ""),
        ),
    )
    area_bucket_counts = {
        entry["values"]: entry["counts"]
        for entry in _ARROW_COMPUTE.value_counts(polygons["area_bucket"]).to_pylist()
    }
    return {
        "polygon_count": polygons.num_rows,
        "unique_wikidata_count": _ARROW_COMPUTE.count_distinct(nonempty_wikidata).as_py(),
        "rows_with_wikipedia": _ARROW_COMPUTE.sum(
            _ARROW_COMPUTE.fill_null(polygons["has_wikipedia"], False)
        ).as_py()
        or 0,
        "rows_with_full_text": _ARROW_COMPUTE.sum(
            _ARROW_COMPUTE.fill_null(polygons["text_available"], False)
        ).as_py()
        or 0,
        "area_bucket_counts": area_bucket_counts,
        "top_tag_keys": top_tag_keys_from_values(polygons["tag_keys"].to_pylist()),
    }


def polygon_manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate manifest values derived from polygon rows."""
    return {
        "polygon_count": len(rows),
        "unique_wikidata_count": len({row["wikidata"] for row in rows if row["wikidata"]}),
        "rows_with_wikipedia": sum(bool(row["has_wikipedia"]) for row in rows),
        "rows_with_full_text": sum(bool(row["text_available"]) for row in rows),
        "area_bucket_counts": _area_bucket_counts(rows),
        "top_tag_keys": _top_tag_keys(rows),
    }


def _area_bucket_counts(rows: list[dict[str, Any]]) -> dict[Any, int]:
    """Count polygon rows by their existing area bucket."""
    return dict(Counter(row["area_bucket"] for row in rows))


def _top_tag_keys(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count valid serialized tag keys, ignoring malformed rows."""
    return top_tag_keys_from_values(row["tag_keys"] for row in rows)


def top_tag_keys_from_values(values: Iterable[Any]) -> dict[str, int]:
    """Count valid serialized tag keys in row order, ignoring malformed values."""
    tag_keys: Counter[str] = Counter()
    for value, row_count in Counter(values).items():
        try:
            parsed_counts = Counter(json.loads(value))
        except (TypeError, ValueError):
            continue
        for key, count in parsed_counts.items():
            tag_keys[key] += count * row_count
    return dict(tag_keys.most_common(50))


def document_manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate manifest values derived from Wikipedia document rows."""
    languages = sorted({row["language"] for row in rows})
    return {
        "article_count": len(rows),
        "language_count": len(languages),
        "languages": languages,
        "total_full_text_chars": sum(row["article_length_chars"] for row in rows),
    }
