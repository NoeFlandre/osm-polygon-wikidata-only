"""Split coverage tests (part 3)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


def test_second_refresh_reuses_cache_zero_parquet_reads(tmp_path: Path) -> None:
    """The per-file cache makes the second refresh a no-op for stable files.

    We assert that no Parquet table is read during the second call by
    spying on :func:`safe_table` calls. A reuse must, by definition, not
    touch any PyArrow table IO.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])

    cache_dir = tmp_path / "cache"

    # Cold refresh: many safe_table calls.
    real_safe_table = augmod.safe_table
    call_log: list[Path] = []

    def spy_safe_table(path, columns):
        call_log.append(Path(path))
        return real_safe_table(path, columns)

    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    cold_calls = len(call_log)
    assert cold_calls > 0
    assert first.fully_augmented_count == 1

    # Warm refresh: no new safe_table calls; the cache satisfies every
    # lookup.
    call_log.clear()
    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    warm_calls = len(call_log)
    assert warm_calls == 0
    # Same numbers across the two refreshes.
    assert second == first


def test_one_changed_file_rescans_only_that_file(tmp_path: Path) -> None:
    """A fingerprint change in one Parquet forces a rescan of that
    file (and its fingerprint change), and only that file's
    :func:`safe_table` is invoked.
    """
    from osm_polygon_wikidata_only.hf._dataset_stats import augmentation as augmod

    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    docs_path = _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_documents(
        processed / "wikivoyage" / "documents" / "monaco-latest.parquet",
        [],
    )
    _write_sections(
        processed / "wikivoyage" / "sections" / "monaco-latest.parquet",
        [],
    )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])

    cache_dir = tmp_path / "cache"
    # Cold then warm refresh.
    compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    compute_augmentation_stats(processed, cache_index_dir=cache_dir)

    # Force a fingerprint change on docs_path only.
    import time

    time.sleep(0.01)
    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "Hello world",
                "article_length_chars": 11,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
            {
                "document_id": "d2",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "And again",
                "article_length_chars": 9,
                "article_length_words": 2,
                "article_length_tokens_estimate": 3,
            },
        ],
    )
    assert docs_path.stat().st_size > 0  # touched.

    real_safe_table = augmod.safe_table
    call_log: list[Path] = []

    def spy_safe_table(path, columns):
        call_log.append(Path(path))
        return real_safe_table(path, columns)

    augmod.safe_table = spy_safe_table  # type: ignore[assignment]
    try:
        third = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    finally:
        augmod.safe_table = real_safe_table  # type: ignore[assignment]
    # Only the changed file's table is read.
    assert len(call_log) == 1
    assert call_log[0] == docs_path
    assert third.wikipedia_documents.rows == 2


def test_deleted_files_removed_from_aggregates(tmp_path: Path) -> None:
    """A sidecar removed from disk disappears from the next refresh's
    aggregates; the cache key for the missing file is dropped.
    """
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_documents(
        processed / "wikipedia" / "documents" / "monaco-latest.parquet",
        [
            {
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "full_text": "x",
                "article_length_chars": 1,
                "article_length_words": 1,
                "article_length_tokens_estimate": 1,
            }
        ],
    )
    _write_sections(
        processed / "wikipedia" / "sections" / "monaco-latest.parquet",
        [
            {
                "section_id": "s1",
                "document_id": "d1",
                "wikidata": "Q1",
                "project": "wikipedia",
                "language": "en",
                "text": "x",
                "text_length_chars": 1,
                "text_length_words": 1,
                "text_length_tokens_estimate": 1,
            }
        ],
    )
    _write_documents(processed / "wikivoyage" / "documents" / "monaco-latest.parquet", [])
    _write_sections(processed / "wikivoyage" / "sections" / "monaco-latest.parquet", [])
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", [])
    docs_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    cache_dir = tmp_path / "cache"

    first = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert first.wikipedia_documents.rows == 1
    assert first.fully_augmented_count == 1

    docs_path.unlink()
    second = compute_augmentation_stats(processed, cache_index_dir=cache_dir)
    assert second.wikipedia_documents.rows == 0
    assert second.wikipedia_documents.region_count == 0
    # Coverage drops from fully to partial after the deletion.
    assert second.fully_augmented_count == 0
    assert second.partial_augmented_count == 1


def test_publication_uses_external_cache_dir(tmp_path: Path) -> None:
    """The publication layer points compute_augmentation_stats at
    data_root.cache (an external directory). Sanity-check that path.
    """
    data_root_path = tmp_path / "data"
    cache_path = data_root_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    # Touching the cache index file proves compute_augmentation_stats
    # writes under data_root.cache.
    from osm_polygon_wikidata_only.config.paths import DataRoot
    from osm_polygon_wikidata_only.hf.publication import write_readme_snapshot

    data_root = DataRoot(data_root_path)
    data_root.ensure()
    # No core or augmentation artifacts; write_readme_snapshot still
    # writes the cache index.
    write_readme_snapshot(
        data_root,
        "test/repo",
        tmp_path / "out.md",
    )
    assert (cache_path / "stats_cache" / "index.json").exists()
