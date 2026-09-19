"""Split coverage tests (part 3)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


def test_render_stats_section_headline_includes_augmentation_totals() -> None:
    """The Dataset snapshot headline shows the augmentation totals."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(rows=433201, total_words=164952567),
        wikipedia_sections=ProjectTextStats(rows=2318909, total_words=230802671),
        wikivoyage_documents=ProjectTextStats(rows=3876, total_words=4896213),
        wikivoyage_sections=ProjectTextStats(rows=79889),
        wikidata_facts=WikidataFactStats(rows=1018033),
        core_parquet_bytes=3140525690,
        augmentation_parquet_bytes=3009776614,
        total_parquet_bytes=6150302304,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    # Headline augmentation totals appear (no change to legacy rows).
    for label in (
        "Wikipedia documents",
        "Wikipedia sections",
        "Wikivoyage documents",
        "Wikivoyage sections",
        "Wikidata facts",
        "Wikipedia + Wikivoyage document words",
        "Total Parquet size",
    ):
        assert label in md, f"headline missing {label!r}"

def test_render_stats_headline_drops_redundant_wikipedia_articles_row() -> None:
    """When augmentation stats are present the legacy ``Wikipedia
    articles`` headline row is dropped because it counts the same
    canonical Wikipedia document rows as ``Wikipedia documents`` and
    is therefore redundant.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Wikipedia articles |" not in md

def test_render_stats_headline_drops_ambiguous_total_words_row() -> None:
    """When augmentation stats are present the legacy ``Total words``
    headline row is dropped: it is ambiguous once the augmentation
    word totals (Wikipedia + Wikivoyage documents) are shown.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Total words |" not in md

def test_render_stats_headline_renames_document_corpus_words() -> None:
    """The combined word total is renamed to the explicit
    ``Wikipedia + Wikivoyage document words`` label.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "| Document corpus words |" not in md
    wiki = aug.wikipedia_documents.total_words
    voy = aug.wikivoyage_documents.total_words
    assert f"| Wikipedia + Wikivoyage document words | {_fmt_int(wiki + voy)} |" in md, (
        f"combined word value wrong; got snippet:\n{md[:600]!r}"
    )

def test_render_stats_headline_section_words_excluded_from_corpus_total() -> None:
    """The combined document-word total must equal Wikipedia document
    words plus Wikivoyage document words, and must NOT include either
    project's section words (sections duplicate document text).
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    wiki = aug.wikipedia_documents.total_words
    voy = aug.wikivoyage_documents.total_words
    sec = aug.wikipedia_sections.total_words + aug.wikivoyage_sections.total_words
    combined = wiki + voy
    assert sec > 0, "fixture must include non-zero section words to prove exclusion"
    assert f"| Wikipedia + Wikivoyage document words | {_fmt_int(combined)} |" in md

def test_render_stats_headline_includes_exclusion_sentence() -> None:
    """A one-line explanation immediately below the headline table
    states that the total sums full Wikipedia and Wikivoyage documents
    and excludes section rows because sections duplicate document text.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    snippet = "sums the full Wikipedia and Wikivoyage documents and excludes section rows"
    assert snippet in md, f"explanatory sentence missing; snippet:\n{md[:600]!r}"

def test_render_stats_headline_counts_from_supplied_snapshot() -> None:
    """Every displayed count in the augmentation-aware headline comes
    from the supplied/computed snapshot, not from hardcoded values.
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    # Wikipedia documents count from the snapshot
    assert f"| Wikipedia documents | {_fmt_int(aug.wikipedia_documents.rows)} |" in md
    # Wikivoyage documents count from the snapshot
    assert f"| Wikivoyage documents | {_fmt_int(aug.wikivoyage_documents.rows)} |" in md
    assert "Fully augmented regions" not in md
    assert "Augmentation tables size" not in md

def test_render_stats_headline_deterministic() -> None:
    """Rendering the same stats + augmentation snapshot twice yields
    byte-identical output (no timestamp, UUID, or clock dependence).
    """
    stats = _empty_dataset_stats()
    aug = _sample_augmentation_stats()
    first = render_stats_section(stats, augmentation_stats=aug)
    second = render_stats_section(stats, augmentation_stats=aug)
    assert first == second

def test_render_stats_language_section_uses_wikipedia_documents_terminology() -> None:
    """The public-facing language-distribution terminology describes
    the canonical rows as "Wikipedia documents" rather than "articles".

    This only applies to the augmentation-aware render, which is the
    surface that surfaces canonical Wikipedia document counts.
    """
    stats = DatasetStats(
        polygon_count=2,
        unique_wikidata_count=1,
        article_count=3,
        link_count=3,
        language_count=2,
        region_count=1,
        total_words=200,
        total_tokens_estimate=50,
        dataset_size_bytes=4096,
        polygons_with_wikipedia=2,
        polygons_with_text=2,
        polygons_with_english=2,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=2,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 2, "fr": 1},
        polygons_per_language={"en": 2, "fr": 1},
    )
    aug = _sample_augmentation_stats()
    md = render_stats_section(stats, augmentation_stats=aug)
    # The explanatory notion "of all articles" is replaced.
    assert "of all articles" not in md
    assert "of all Wikipedia documents" in md

def test_render_stats_combined_language_polygons_describe_identity_success_semantics() -> None:
    stats = _empty_dataset_stats()
    aug = replace(
        _sample_augmentation_stats(),
        combined_languages=CombinedLanguageStats(
            document_count=3,
            language_count=2,
            documents_per_language=(("en", 2), ("fr", 1)),
            polygons_per_language=(("en", 1), ("fr", 1)),
        ),
    )

    md = render_stats_section(stats, augmentation_stats=aug)

    assert "unique `(osm_type, osm_id)` polygon identities" in md
    assert "`fetch_status=ok`" in md
    assert "trimmed non-empty `full_text`" in md

def test_stats_rendering_and_cache_helpers_cover_boundary_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Small helpers retain their documented edge behavior."""
    assert _article_tail_counts({"none": 0, "few": 4, "some": 9, "many": 10}) == {
        "articles_lt1": 1,
        "articles_lt5": 2,
        "articles_lt10": 3,
    }
    assert [_fmt_size(value) for value in (0, 1024, 1024**2, 1024**3, 1024**4, 1024**5)] == [
        "0.0 B",
        "1.0 KB",
        "1.0 MB",
        "1.0 GB",
        "1.0 TB",
        "1.0 PB",
    ]

    clean = _sample_augmentation_stats()
    unreadable = replace(clean, unreadable_file_count=2)
    assert _render_unreadable_note(clean) == ""
    assert "2 unreadable sidecar file(s)" in _render_unreadable_note(unreadable)

    summary = PerFileSummary(
        relative_path="wikipedia/documents/example.parquet",
        fingerprint="fingerprint",
        file_size_bytes=17,
        kind="documents",
    )
    failed = replace(summary, scan_failed=True)
    assert _augmentation_bytes({"wikipedia/documents": [summary, failed]}) == 34
    assert _has_readable_summary([summary, failed]) is True
    assert _has_readable_summary([failed]) is False

    parquet_path = tmp_path / "example.parquet"
    parquet_path.write_bytes(b"cached")
    fingerprint = augmentation_module._file_fingerprint(parquet_path)
    cached_summary = replace(summary, fingerprint=fingerprint)
    cached = augmentation_module._summary_to_json(cached_summary)
    assert _load_or_scan_summary(tmp_path, parquet_path, cached) == cached_summary

    scanned = replace(summary, fingerprint="rescanned")
    monkeypatch.setattr(augmentation_module, "_scan_one_file", lambda *_args: scanned)
    assert _load_or_scan_summary(tmp_path, parquet_path, None) is scanned

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

def test_render_stats_section_legacy_headline_label_unchanged() -> None:
    """Without augmentation, the headline's last label MUST stay
    ``Dataset size on disk`` -- the exact wording used before the
    augmentation extension was introduced.
    """
    stats = DatasetStats(
        polygon_count=2,
        unique_wikidata_count=1,
        article_count=3,
        link_count=3,
        language_count=2,
        region_count=1,
        total_words=200,
        total_tokens_estimate=50,
        dataset_size_bytes=4096,
        polygons_with_wikipedia=2,
        polygons_with_text=2,
        polygons_with_english=2,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=2,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 2, "fr": 1},
        polygons_per_language={"en": 2, "fr": 1},
    )
    md = render_stats_section(stats)
    # Locate the legacy last headline row by its unique label.
    assert "| Dataset size on disk | 4.0 KB |" in md, (
        f"legacy headline last row drifted; snippet: {md[:500]!r}"
    )
    # The legacy-only render must not include the augmentation
    # headline row "Wikipedia documents | <n> |" (a value-bearing
    # table row). The language-distribution section legitimately
    # uses the "Wikipedia documents" column header, so we anchor on
    # the headline table's value row shape.
    assert "| Core tables size |" not in md
    import re as _re

    headline_document_row = _re.compile(r"\| Wikipedia documents \| \d")
    assert not headline_document_row.search(md)
    assert "Total sidecar words" not in md
    # The new headline label (augmentation-aware only).
    assert "| Wikipedia + Wikivoyage document words |" not in md
    # The public-facing language section uses canonical terminology.
    assert "| Language | Wikipedia documents | % of total | Polygons |" in md
