"""Tests for the dataset statistics module.

These tests verify that the stats computed from the processed parquet
files are factual and correct: given known data, the stats must match
exactly. The tests also verify the rendered markdown sections contain
the expected figures.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats import aggregation
from osm_polygon_wikidata_only.hf.dataset_stats import (
    DatasetStats,
    compute_dataset_stats,
    render_stats_section,
)

# --- helpers ------------------------------------------------------------


@pytest.mark.parametrize("canonical_layout", [False, True])
def test_compute_dataset_stats_routes_exact_layout_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_layout: bool,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    canonical_articles = processed / "wikipedia" / "documents"
    if canonical_layout:
        canonical_articles.mkdir(parents=True)
    observed: list[Path] = []

    def record_path(_stats: object, path: Path) -> None:
        observed.append(path)

    monkeypatch.setattr(aggregation, "_accumulate_polygon_files", record_path)
    monkeypatch.setattr(aggregation, "_accumulate_article_files", record_path)
    monkeypatch.setattr(aggregation, "_accumulate_link_files", record_path)

    stats = aggregation.compute_dataset_stats(processed)

    expected_articles = canonical_articles if canonical_layout else processed / "articles"
    assert observed == [
        processed / "polygons",
        expected_articles,
        processed / "polygon_articles",
    ]
    assert stats == aggregation._StatsAccumulator().to_stats()


def test_file_scans_accumulate_sizes_and_counts_from_existing_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon_dir = tmp_path / "polygons"
    article_dir = tmp_path / "articles"
    links_dir = tmp_path / "polygon_articles"
    for directory in (polygon_dir, article_dir, links_dir):
        directory.mkdir()
        (directory / "part.parquet").write_bytes(b"abc")

    monkeypatch.setattr(aggregation, "safe_table", lambda *_args: None)
    monkeypatch.setattr(aggregation, "_count_link_rows", lambda _paths: 4)
    stats = aggregation._StatsAccumulator(dataset_size_bytes=7, link_count=3)

    aggregation._accumulate_polygon_files(stats, polygon_dir)
    aggregation._accumulate_article_files(stats, article_dir)
    aggregation._accumulate_link_files(stats, links_dir)

    assert stats.dataset_size_bytes == 16
    assert stats.link_count == 7


def test_polygon_table_accumulates_into_existing_totals() -> None:
    table = pa.table(
        {
            "wikidata": ["Q1", "Q2"],
            "region": ["a", "b"],
            "has_wikipedia": [True, True],
            "text_available": [True, False],
            "has_english_wikipedia": [True, False],
            "wikipedia_language_count": [10, 1],
            "wikipedia_languages": ["[]", "[]"],
        }
    )
    stats = aggregation._StatsAccumulator(
        polygon_count=5,
        polygons_with_wikipedia=5,
        polygons_with_text=5,
        polygons_with_english=5,
        polygons_with_no_english_other_lang=5,
        polygons_with_2plus_langs=5,
        polygons_with_5plus_langs=5,
        polygons_with_10plus_langs=5,
    )

    aggregation._accumulate_polygon_table(stats, table)

    assert stats.polygon_count == 7
    assert stats.polygons_with_wikipedia == 7
    assert stats.polygons_with_text == 6
    assert stats.polygons_with_english == 6
    assert stats.polygons_with_no_english_other_lang == 6
    assert stats.polygons_with_2plus_langs == 6
    assert stats.polygons_with_5plus_langs == 6
    assert stats.polygons_with_10plus_langs == 6


def test_article_table_accumulates_into_existing_totals() -> None:
    table = pa.table(
        {
            "language": ["fr", "en"],
            "article_length_words": [2, 3],
            "article_length_tokens_estimate": [4, 5],
        }
    )
    stats = aggregation._StatsAccumulator(
        article_count=5,
        total_words=7,
        total_tokens_estimate=11,
        articles_per_language=Counter({"fr": 3}),
    )

    aggregation._accumulate_article_table(stats, table)

    assert stats.article_count == 7
    assert stats.total_words == 12
    assert stats.total_tokens_estimate == 20
    assert stats.articles_per_language == {"fr": 4, "en": 1}


def test_python_reductions_preserve_boundaries_and_strict_pairing() -> None:
    assert aggregation._count_language_buckets([2, None, 5, 10, 10]) == (4, 3, 2)
    assert aggregation._sum_numeric([1.9, None, 2.2, 0.9]) == 3
    with pytest.raises(ValueError):
        aggregation._count_non_english([True], [])


def test_mixed_boolean_columns_use_python_non_english_fallback() -> None:
    wikipedia = pa.chunked_array([[True, False]])
    english = pa.chunked_array([[0, 1]])

    assert aggregation._count_non_english_columns(wikipedia, english) == 1


@pytest.mark.parametrize(
    "data_type",
    [pa.string(), pa.large_string(), pa.binary(), pa.large_binary()],
)
def test_serialized_arrow_types_are_recognized(data_type: pa.DataType) -> None:
    assert aggregation._is_serialized_string(data_type)


def test_value_string_counting_routes_by_arrow_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def count_python(_values: list[object]) -> Counter[str]:
        observed.append("python")
        return Counter({"python": 1})

    def count_serialized(_values: pa.ChunkedArray) -> Counter[str]:
        observed.append("serialized")
        return Counter({"serialized": 1})

    monkeypatch.setattr(aggregation, "_count_python_value_strings", count_python)
    monkeypatch.setattr(aggregation, "_count_serialized_value_strings", count_serialized)

    assert aggregation._count_value_strings(pa.chunked_array([["en"]])) == {"serialized": 1}
    assert aggregation._count_value_strings(pa.chunked_array([[1]])) == {"python": 1}
    assert observed == ["serialized", "python"]


def test_arrow_value_counts_rejects_misaligned_kernel_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MisalignedFrequencies:
        def field(self, name: str) -> pa.Array:
            return pa.array(["en", "fr"] if name == "values" else [1])

    monkeypatch.setattr(
        aggregation,
        "_compute_array",
        lambda *_args: MisalignedFrequencies(),
    )

    with pytest.raises(ValueError):
        aggregation._arrow_value_counts(pa.chunked_array([["en"]]))


def test_scalar_int_maps_null_to_zero() -> None:
    assert aggregation._scalar_int(pa.scalar(None, type=pa.int64())) == 0


def test_parse_language_list_does_not_decode_canonical_empty_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_decoded(_value: object) -> object:
        raise AssertionError("canonical empty language lists must not be decoded")

    monkeypatch.setattr(aggregation.json, "loads", fail_if_decoded)

    assert aggregation._parse_language_list("[]") == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('["en", "fr"]', ["en", "fr"]),
        ('{"language": "en"}', []),
        ("not-json", []),
    ],
)
def test_decoded_language_list_handles_json_shapes_and_errors(
    value: str, expected: list[object]
) -> None:
    assert aggregation._decoded_language_list(value) == expected


@pytest.mark.parametrize("value", [None, 0, {}, object()])
def test_parse_language_list_rejects_non_string_values(value: object) -> None:
    assert aggregation._parse_language_list(value) == []


def test_polygon_language_values_are_decoded_once_per_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = '["en","en","fr"]'
    malformed = "not-json"
    table = pa.table(
        {
            "wikidata": ["Q1", "Q2", "Q3", "Q4"],
            "region": ["a", "a", "a", "a"],
            "has_wikipedia": [True, True, False, False],
            "text_available": [True, True, False, False],
            "has_english_wikipedia": [True, True, False, False],
            "wikipedia_language_count": [3, 3, 0, 0],
            "wikipedia_languages": [repeated, repeated, malformed, "[]"],
        }
    )
    decoded: list[object] = []
    original_loads = aggregation.json.loads

    def counting_loads(value: object) -> object:
        decoded.append(value)
        return original_loads(value)

    monkeypatch.setattr(aggregation.json, "loads", counting_loads)
    stats = aggregation._StatsAccumulator()

    aggregation._accumulate_polygon_table(stats, table)
    aggregation._accumulate_polygon_table(stats, table.slice(0, 2))

    assert decoded == [repeated, malformed]
    assert stats.polygons_per_language == {"en": 8, "fr": 4}


def test_polygon_language_cache_cap_preserves_uncached_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregation, "_LANGUAGE_CACHE_MAX_ENTRIES", 1)
    decoded: list[object] = []
    original_loads = aggregation.json.loads

    def counting_loads(value: object) -> object:
        decoded.append(value)
        return original_loads(value)

    monkeypatch.setattr(aggregation.json, "loads", counting_loads)
    stats = aggregation._StatsAccumulator()
    table = pa.table({"wikipedia_languages": ['["en"]', '["fr"]']})

    for _ in range(2):
        stats.polygons_per_language.update(
            aggregation._count_polygon_language_column(
                table.column("wikipedia_languages"),
                stats.language_lists,
            )
        )

    assert stats.language_lists == {'["en"]': ("en",)}
    assert decoded == ['["en"]', '["fr"]', '["fr"]']
    assert stats.polygons_per_language == {"en": 2, "fr": 2}


def test_native_polygon_reductions_preserve_chunked_null_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "wikidata": pa.chunked_array([["Q1", None], ["", "Q2", "Q2"]]),
            "region": pa.chunked_array([["a", "a"], [None, "b", ""]]),
            "has_wikipedia": pa.chunked_array([[True, True], [None, False, True]]),
            "text_available": pa.chunked_array([[True, None], [False, True, False]]),
            "has_english_wikipedia": pa.chunked_array([[True, None], [False, None, False]]),
            "wikipedia_language_count": pa.chunked_array([[1, 2], [None, 5, 10]]),
            "wikipedia_languages": pa.chunked_array(
                [['["en"]', '["fr"]'], ["[]", '["de"]', '["es"]']]
            ),
        }
    )

    def fail_legacy_helper(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("native Arrow columns used a Python scalar helper")

    monkeypatch.setattr(aggregation, "_count_truthy", fail_legacy_helper)
    monkeypatch.setattr(aggregation, "_count_non_english", fail_legacy_helper)
    monkeypatch.setattr(aggregation, "_count_language_buckets", fail_legacy_helper)
    stats = aggregation._StatsAccumulator()

    aggregation._accumulate_polygon_table(stats, table)

    assert stats.polygon_count == 5
    assert stats.unique_wikidata == {"Q1", "Q2"}
    assert stats.distinct_regions == {"a", "b"}
    assert stats.polygons_with_wikipedia == 3
    assert stats.polygons_with_text == 2
    assert stats.polygons_with_english == 1
    assert stats.polygons_with_no_english_other_lang == 2
    assert stats.polygons_with_2plus_langs == 3
    assert stats.polygons_with_5plus_langs == 2
    assert stats.polygons_with_10plus_langs == 1


def test_polygon_identity_columns_are_uniqued_before_python_set_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "wikidata": ["Q1", "Q1", "", "Q2"],
            "region": ["a", "a", "b", "b"],
            "has_wikipedia": [False] * 4,
            "text_available": [False] * 4,
            "has_english_wikipedia": [False] * 4,
            "wikipedia_language_count": [0] * 4,
            "wikipedia_languages": ["[]"] * 4,
        }
    )
    observed: list[tuple[object, ...]] = []
    original_update = aggregation._update_unique_values

    def tracking_update(values: list[object], target: set[str]) -> None:
        observed.append(tuple(values))
        original_update(values, target)

    monkeypatch.setattr(aggregation, "_update_unique_values", tracking_update)
    stats = aggregation._StatsAccumulator()

    aggregation._accumulate_polygon_table(stats, table)

    assert observed == [("Q1", "", "Q2"), ("a", "b")]
    assert stats.unique_wikidata == {"Q1", "Q2"}
    assert stats.distinct_regions == {"a", "b"}


def test_native_article_reductions_preserve_totals_and_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "language": pa.chunked_array([["fr", "en", None], ["", "en", "fr"]]),
            "article_length_words": pa.chunked_array([[10, None, 5], [1, 4, 6]]),
            "article_length_tokens_estimate": pa.chunked_array([[2, 3, None], [1, 1, 2]]),
        }
    )

    def fail_legacy_helper(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("native Arrow columns used a Python scalar helper")

    monkeypatch.setattr(aggregation, "_sum_numeric", fail_legacy_helper)
    stats = aggregation._StatsAccumulator()

    aggregation._accumulate_article_table(stats, table)

    assert stats.article_count == 6
    assert stats.total_words == 26
    assert stats.total_tokens_estimate == 9
    assert list(stats.to_stats().articles_per_language.items()) == [("fr", 2), ("en", 2)]


def test_non_native_arrow_columns_preserve_python_fallback_semantics() -> None:
    polygon_table = pa.table(
        {
            "wikidata": [1, 1, None],
            "region": [7, 8, 8],
            "has_wikipedia": [1, 0, None],
            "text_available": ["yes", "", None],
            "has_english_wikipedia": [0, 1, None],
            "wikipedia_language_count": [2.0, 5.0, None],
            "wikipedia_languages": [["en"], ["fr"], None],
        }
    )
    article_table = pa.table(
        {
            "language": [1, 1, 0, None],
            "article_length_words": [1.9, None, 2.2, 0.9],
            "article_length_tokens_estimate": [None, 3.8, 4.2, 0.0],
        }
    )
    stats = aggregation._StatsAccumulator()

    aggregation._accumulate_polygon_table(stats, polygon_table)
    aggregation._accumulate_article_table(stats, article_table)

    assert stats.unique_wikidata == {"1"}
    assert stats.distinct_regions == {"7", "8"}
    assert stats.polygons_with_wikipedia == 1
    assert stats.polygons_with_text == 1
    assert stats.polygons_with_english == 1
    assert stats.polygons_with_no_english_other_lang == 1
    assert stats.polygons_with_2plus_langs == 2
    assert stats.polygons_with_5plus_langs == 1
    assert stats.polygons_with_10plus_langs == 0
    assert stats.polygons_per_language == {}
    assert stats.articles_per_language == {"1": 2}
    assert stats.total_words == 3
    assert stats.total_tokens_estimate == 7


def test_python_polygon_language_fallback_preserves_supported_inputs() -> None:
    assert aggregation._count_polygon_languages(
        [
            '["en","en"]',
            b'["fr"]',
            bytearray(b'["de"]'),
            ["ignored"],
            "not-json",
            None,
        ]
    ) == {"en": 2, "fr": 1, "de": 1}


def test_link_stats_use_bounded_parallel_metadata_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    links_dir = tmp_path / "polygon_articles"
    links_dir.mkdir()
    for index in range(3):
        pq.write_table(pa.table({"value": [index]}), links_dir / f"part-{index}.parquet")

    original_executor = aggregation.ThreadPoolExecutor
    worker_counts: list[int] = []
    thread_name_prefixes: list[str] = []

    def tracking_executor(*args: object, **kwargs: object):
        worker_counts.append(int(kwargs["max_workers"]))
        thread_name_prefixes.append(str(kwargs["thread_name_prefix"]))
        return original_executor(*args, **kwargs)

    monkeypatch.setattr(aggregation, "ThreadPoolExecutor", tracking_executor)

    stats = aggregation._StatsAccumulator()
    aggregation._accumulate_link_files(stats, links_dir)

    assert stats.link_count == 3
    assert worker_counts == [3]
    assert thread_name_prefixes == ["dataset-stats-metadata"]


def test_count_link_rows_returns_zero_for_no_files() -> None:
    assert aggregation._count_link_rows([]) == 0


def _write_polygons_parquet(path: Path, rows: list[dict]) -> Path:
    """Write a polygons parquet with the columns the stats module reads."""
    columns = [
        "wikidata",
        "region",
        "has_wikipedia",
        "text_available",
        "has_english_wikipedia",
        "wikipedia_language_count",
        "wikipedia_languages",
    ]
    data: dict[str, list] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            data[c].append(row.get(c))
    table = pa.table(data)
    pq.write_table(table, path)
    return path


def _write_articles_parquet(path: Path, rows: list[dict]) -> Path:
    """Write an articles parquet with the columns the stats module reads."""
    columns = ["language", "article_length_words", "article_length_tokens_estimate"]
    data: dict[str, list] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            data[c].append(row.get(c))
    table = pa.table(data)
    pq.write_table(table, path)
    return path


def _write_links_parquet(path: Path, rows: list[dict]) -> Path:
    """Write a polygon-articles links parquet."""
    columns = ["polygon_id", "article_id", "language"]
    data: dict[str, list] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            data[c].append(row.get(c))
    table = pa.table(data)
    pq.write_table(table, path)
    return path


def _setup_processed_dir(tmp_path: Path) -> Path:
    """Create the standard processed sub-directories."""
    processed = tmp_path / "processed"
    (processed / "polygons").mkdir(parents=True)
    (processed / "articles").mkdir(parents=True)
    (processed / "polygon_articles").mkdir(parents=True)
    return processed


# --- compute_dataset_stats: headline counts -----------------------------


def test_compute_stats_polygon_count(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_polygons_parquet(
        processed / "polygons" / "a-latest.parquet",
        [
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 2,
                "wikipedia_languages": '["en","fr"]',
            },
            {
                "wikidata": "Q2",
                "region": "a",
                "has_wikipedia": True,
                "text_available": False,
                "has_english_wikipedia": False,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["de"]',
            },
            {
                "wikidata": "Q3",
                "region": "a",
                "has_wikipedia": False,
                "text_available": False,
                "has_english_wikipedia": False,
                "wikipedia_language_count": 0,
                "wikipedia_languages": "[]",
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.polygon_count == 3


def test_compute_stats_unique_wikidata_count(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_polygons_parquet(
        processed / "polygons" / "a-latest.parquet",
        [
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
            {
                "wikidata": "Q2",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
            {
                "wikidata": "",
                "region": "a",
                "has_wikipedia": False,
                "text_available": False,
                "has_english_wikipedia": False,
                "wikipedia_language_count": 0,
                "wikipedia_languages": "[]",
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.unique_wikidata_count == 2  # Q1 and Q2 (empty ignored)


def test_compute_stats_article_and_link_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_articles_parquet(
        processed / "articles" / "a-latest.parquet",
        [
            {"language": "en", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "fr", "article_length_words": 200, "article_length_tokens_estimate": 50},
            {"language": "de", "article_length_words": 150, "article_length_tokens_estimate": 37},
        ],
    )
    _write_links_parquet(
        processed / "polygon_articles" / "a-latest.parquet",
        [
            {"polygon_id": "p1", "article_id": "a1", "language": "en"},
            {"polygon_id": "p1", "article_id": "a2", "language": "fr"},
            {"polygon_id": "p2", "article_id": "a1", "language": "en"},
            {"polygon_id": "p2", "article_id": "a3", "language": "de"},
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.article_count == 3
    assert stats.link_count == 4


def test_compute_stats_total_words_and_tokens(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_articles_parquet(
        processed / "articles" / "a-latest.parquet",
        [
            {"language": "en", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "fr", "article_length_words": 200, "article_length_tokens_estimate": 50},
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.total_words == 300
    assert stats.total_tokens_estimate == 75


def test_compute_stats_region_count(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_polygons_parquet(
        processed / "polygons" / "a-latest.parquet",
        [
            {
                "wikidata": "Q1",
                "region": "monaco",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
        ],
    )
    _write_polygons_parquet(
        processed / "polygons" / "b-latest.parquet",
        [
            {
                "wikidata": "Q2",
                "region": "albania",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.region_count == 2


def test_compute_stats_dataset_size_bytes(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    polygons_path = processed / "polygons" / "a-latest.parquet"
    _write_polygons_parquet(
        polygons_path,
        [
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.dataset_size_bytes > 0
    assert stats.dataset_size_bytes == polygons_path.stat().st_size


# --- compute_dataset_stats: Wikipedia coverage funnel -------------------


def test_compute_stats_funnel_counts(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    # 4 polygons:
    #   p1: has_wiki=True, text=True, en=True, langs=2
    #   p2: has_wiki=True, text=True, en=True, langs=1
    #   p3: has_wiki=True, text=False, en=False, langs=1  (no text, not English, but has wiki)
    #   p4: has_wiki=False, text=False, en=False, langs=0
    _write_polygons_parquet(
        processed / "polygons" / "a-latest.parquet",
        [
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 2,
                "wikipedia_languages": '["en","fr"]',
            },
            {
                "wikidata": "Q2",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
            {
                "wikidata": "Q3",
                "region": "a",
                "has_wikipedia": True,
                "text_available": False,
                "has_english_wikipedia": False,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["de"]',
            },
            {
                "wikidata": "Q4",
                "region": "a",
                "has_wikipedia": False,
                "text_available": False,
                "has_english_wikipedia": False,
                "wikipedia_language_count": 0,
                "wikipedia_languages": "[]",
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.polygons_with_wikipedia == 3
    assert stats.polygons_with_text == 2
    assert stats.polygons_with_english == 2
    assert stats.polygons_with_no_english_other_lang == 1  # p3
    assert stats.polygons_with_2plus_langs == 1
    assert stats.polygons_with_5plus_langs == 0
    assert stats.polygons_with_10plus_langs == 0


def test_compute_stats_funnel_handles_empty(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    stats = compute_dataset_stats(processed)
    assert stats.polygon_count == 0
    assert stats.polygons_with_wikipedia == 0
    assert stats.polygons_with_text == 0
    assert stats.polygons_with_english == 0


# --- compute_dataset_stats: language distribution -----------------------


def test_compute_stats_articles_per_language(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_articles_parquet(
        processed / "articles" / "a-latest.parquet",
        [
            {"language": "en", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "en", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "en", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "fr", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "fr", "article_length_words": 100, "article_length_tokens_estimate": 25},
            {"language": "de", "article_length_words": 100, "article_length_tokens_estimate": 25},
        ],
    )
    stats = compute_dataset_stats(processed)
    assert stats.articles_per_language == {"en": 3, "fr": 2, "de": 1}
    assert stats.language_count == 3


def test_compute_stats_polygons_per_language(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_polygons_parquet(
        processed / "polygons" / "a-latest.parquet",
        [
            {
                "wikidata": "Q1",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 2,
                "wikipedia_languages": '["en","fr"]',
            },
            {
                "wikidata": "Q2",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 2,
                "wikipedia_languages": '["en","de"]',
            },
            {
                "wikidata": "Q3",
                "region": "a",
                "has_wikipedia": True,
                "text_available": True,
                "has_english_wikipedia": True,
                "wikipedia_language_count": 1,
                "wikipedia_languages": '["en"]',
            },
        ],
    )
    stats = compute_dataset_stats(processed)
    # en appears in 3 polygons, fr in 1, de in 1.
    assert stats.polygons_per_language == {"en": 3, "fr": 1, "de": 1}


# --- render_stats_section: headline table -------------------------------


def test_render_stats_section_includes_headline_metrics() -> None:
    stats = DatasetStats(
        polygon_count=5929,
        unique_wikidata_count=1209,
        article_count=9310,
        link_count=15000,
        language_count=214,
        region_count=3,
        total_words=12_345_678,
        total_tokens_estimate=3_086_419,
        dataset_size_bytes=5_242_880,
        polygons_with_wikipedia=5800,
        polygons_with_text=5750,
        polygons_with_english=4200,
        polygons_with_no_english_other_lang=1600,
        polygons_with_2plus_langs=3500,
        polygons_with_5plus_langs=1200,
        polygons_with_10plus_langs=400,
        articles_per_language={"en": 3200, "fr": 1500, "de": 1000},
        polygons_per_language={"en": 2500, "fr": 1200, "de": 800},
    )
    md = render_stats_section(stats)
    assert "5,929" in md  # polygon count
    assert "1,209" in md  # unique wikidata
    assert "9,310" in md  # articles
    assert "15,000" in md  # links
    assert "214" in md  # languages
    assert "12,345,678" in md  # words


def test_render_stats_section_dataset_size_human_readable() -> None:
    stats = DatasetStats(
        polygon_count=1,
        unique_wikidata_count=1,
        article_count=1,
        link_count=0,
        language_count=1,
        region_count=1,
        total_words=0,
        total_tokens_estimate=0,
        dataset_size_bytes=5_242_880,  # 5 MB
        polygons_with_wikipedia=0,
        polygons_with_text=0,
        polygons_with_english=0,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=0,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={},
        polygons_per_language={},
    )
    md = render_stats_section(stats)
    assert "5.0 MB" in md


# --- render_stats_section: funnel --------------------------------------


def test_render_stats_section_includes_funnel() -> None:
    stats = DatasetStats(
        polygon_count=100,
        unique_wikidata_count=50,
        article_count=200,
        link_count=300,
        language_count=10,
        region_count=2,
        total_words=1000,
        total_tokens_estimate=250,
        dataset_size_bytes=1024,
        polygons_with_wikipedia=80,
        polygons_with_text=70,
        polygons_with_english=60,
        polygons_with_no_english_other_lang=20,
        polygons_with_2plus_langs=30,
        polygons_with_5plus_langs=10,
        polygons_with_10plus_langs=2,
        articles_per_language={"en": 100, "fr": 50},
        polygons_per_language={"en": 80, "fr": 40},
    )
    md = render_stats_section(stats)
    assert "funnel" in md.lower()
    # Percentages
    assert "80.0%" in md  # with wiki: 80/100
    assert "70.0%" in md  # with text: 70/100
    assert "60.0%" in md  # with en: 60/100
    assert "20.0%" in md  # no en other lang: 20/100


# --- render_stats_section: language distribution -----------------------


def test_render_stats_section_includes_top_languages() -> None:
    articles = {f"lang{i:02d}": 100 - i for i in range(25)}
    polygons = {f"lang{i:02d}": 50 - i for i in range(25)}
    stats = DatasetStats(
        polygon_count=1000,
        unique_wikidata_count=500,
        article_count=2000,
        link_count=3000,
        language_count=25,
        region_count=5,
        total_words=50000,
        total_tokens_estimate=12500,
        dataset_size_bytes=10240,
        polygons_with_wikipedia=900,
        polygons_with_text=850,
        polygons_with_english=700,
        polygons_with_no_english_other_lang=200,
        polygons_with_2plus_langs=500,
        polygons_with_5plus_langs=200,
        polygons_with_10plus_langs=50,
        articles_per_language=articles,
        polygons_per_language=polygons,
    )
    md = render_stats_section(stats)
    # Top 20 should be shown, not all 25.
    assert "lang00" in md
    assert "lang19" in md
    assert "lang20" not in md  # not in top 20


def test_render_stats_section_concentration_percentages() -> None:
    articles = {"en": 50, "fr": 20, "de": 15, "es": 10, "it": 5}  # 100 total
    polygons = {k: 10 for k in articles}
    stats = DatasetStats(
        polygon_count=100,
        unique_wikidata_count=50,
        article_count=100,
        link_count=100,
        language_count=5,
        region_count=1,
        total_words=1000,
        total_tokens_estimate=250,
        dataset_size_bytes=1024,
        polygons_with_wikipedia=100,
        polygons_with_text=100,
        polygons_with_english=100,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=100,
        polygons_with_5plus_langs=100,
        polygons_with_10plus_langs=100,
        articles_per_language=articles,
        polygons_per_language=polygons,
    )
    md = render_stats_section(stats)
    assert "50.0%" in md  # top 1: en has 50/100
    # Top 5 = 100%, so we should see 100.0% somewhere.


def test_render_stats_section_handles_empty_language_data() -> None:
    stats = DatasetStats(
        polygon_count=0,
        unique_wikidata_count=0,
        article_count=0,
        link_count=0,
        language_count=0,
        region_count=0,
        total_words=0,
        total_tokens_estimate=0,
        dataset_size_bytes=0,
        polygons_with_wikipedia=0,
        polygons_with_text=0,
        polygons_with_english=0,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=0,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={},
        polygons_per_language={},
    )
    md = render_stats_section(stats)
    # Should still produce valid markdown without crashing.
    assert "## " in md  # at least one section header


# --- integration: stats are factual against the data -------------------


def test_stats_match_manual_count_from_parquet(tmp_path: Path) -> None:
    """Cross-check: compute_stats matches a manual count over the data."""
    processed = _setup_processed_dir(tmp_path)
    polygons = [
        {
            "wikidata": "Q1",
            "region": "monaco",
            "has_wikipedia": True,
            "text_available": True,
            "has_english_wikipedia": True,
            "wikipedia_language_count": 3,
            "wikipedia_languages": '["en","fr","de"]',
        },
        {
            "wikidata": "Q1",
            "region": "monaco",
            "has_wikipedia": True,
            "text_available": True,
            "has_english_wikipedia": True,
            "wikipedia_language_count": 3,
            "wikipedia_languages": '["en","fr","de"]',
        },
        {
            "wikidata": "Q2",
            "region": "albania",
            "has_wikipedia": True,
            "text_available": True,
            "has_english_wikipedia": False,
            "wikipedia_language_count": 2,
            "wikipedia_languages": '["de","es"]',
        },
    ]
    articles = [
        {"language": "en", "article_length_words": 50, "article_length_tokens_estimate": 12},
        {"language": "fr", "article_length_words": 60, "article_length_tokens_estimate": 15},
        {"language": "de", "article_length_words": 70, "article_length_tokens_estimate": 17},
        {"language": "de", "article_length_words": 80, "article_length_tokens_estimate": 20},
    ]
    links = [
        {"polygon_id": "p1", "article_id": "a1", "language": "en"},
        {"polygon_id": "p1", "article_id": "a2", "language": "fr"},
        {"polygon_id": "p2", "article_id": "a3", "language": "de"},
        {"polygon_id": "p3", "article_id": "a3", "language": "de"},
        {"polygon_id": "p3", "article_id": "a4", "language": "de"},
    ]
    _write_polygons_parquet(processed / "polygons" / "x.parquet", polygons)
    _write_articles_parquet(processed / "articles" / "x.parquet", articles)
    _write_links_parquet(processed / "polygon_articles" / "x.parquet", links)

    stats = compute_dataset_stats(processed)

    # Manual cross-checks.
    assert stats.polygon_count == 3
    assert stats.unique_wikidata_count == 2  # Q1, Q2
    assert stats.article_count == 4
    assert stats.link_count == 5
    assert stats.total_words == 50 + 60 + 70 + 80
    assert stats.total_tokens_estimate == 12 + 15 + 17 + 20
    assert stats.polygons_with_wikipedia == 3
    assert stats.polygons_with_text == 3
    assert stats.polygons_with_english == 2  # Q1 polygons
    assert stats.polygons_with_no_english_other_lang == 1  # Q2 polygon
    assert stats.polygons_with_2plus_langs == 3
    assert stats.polygons_with_5plus_langs == 0
    assert stats.polygons_with_10plus_langs == 0
    assert stats.region_count == 2
    assert stats.language_count == 3  # en, fr, de
    assert stats.articles_per_language == {"en": 1, "fr": 1, "de": 2}
    # en appears in polygons 1+2 (Q1 x2), fr in 1+2, de in 1+2+3 (Q1 x2 + Q2), es in 3.
    assert stats.polygons_per_language == {"en": 2, "fr": 2, "de": 3, "es": 1}


# --- integration: dataset card includes stats section ------------------


def test_dataset_card_includes_stats_section_when_provided() -> None:
    from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card

    stats_section = "## Dataset snapshot\n\n| Metric | Value |\n| --- | --- |\n"
    markdown = render_dataset_card(
        repo_id="org/name",
        stats={"polygon_count": 1, "article_count": 2, "unique_wikidata_count": 1},
        polygon_columns=["polygon_id"],
        polygon_descriptions={"polygon_id": "id"},
        article_columns=["article_id"],
        article_descriptions={"article_id": "id"},
        link_columns=["polygon_id"],
        link_descriptions={"polygon_id": "id"},
        stats_section=stats_section,
    )
    assert stats_section in markdown
