"""Tests for the dataset statistics module.

These tests verify that the stats computed from the processed parquet
files are factual and correct: given known data, the stats must match
exactly. The tests also verify the rendered markdown sections contain
the expected figures.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.dataset_stats import (
    DatasetStats,
    compute_dataset_stats,
    render_stats_section,
)

# --- helpers ------------------------------------------------------------


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
