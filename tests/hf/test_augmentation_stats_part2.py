"""Split coverage tests (part 2)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.hf.augmentation_stats_support import *


def test_wikidata_facts_qualifiers_and_references_detected(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": json.dumps({"P580": "2000-01-01"}),
                "references": json.dumps([{"snaks": {"P248": ["Q5"]}}]),
            },
            {
                "fact_id": "f2",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "null",
                "references": "[]",
            },
            {
                "fact_id": "f3",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "",
                "references": "   ",
            },
            {
                "fact_id": "f4",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q235",
                "value_label_en": "Monaco",
                "value_labels": '{"en": "Monaco"}',
                "value_text": "Q235",
                "qualifiers": "{}",
                "references": "{}",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.with_qualifiers == 1
    assert stats.wikidata_facts.with_references == 1

def test_wikidata_facts_top_properties_deterministic(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    rows: list[dict] = []
    facts = [("P17", "country"), ("P31", "instance of"), ("P131", "located in")]
    for property_id, label in facts:
        for _ in range(2):
            rows.append(
                {
                    "fact_id": f"{property_id}-{label}",
                    "wikidata": "Q1",
                    "property_id": property_id,
                    "property_label_en": label,
                    "property_labels": json.dumps({"en": label}),
                    "value_type": "wikibase-entityid",
                    "value_entity_id": "Q2",
                    "value_label_en": "x",
                    "value_labels": '{"en": "x"}',
                    "value_text": "Q2",
                    "qualifiers": "{}",
                    "references": "[]",
                }
            )
    _write_facts(processed / "wikidata" / "facts" / "monaco-latest.parquet", rows)
    stats = _stats(processed, tmp_path)
    top = stats.wikidata_facts.top_properties
    assert [prop for prop, _, _ in top] == ["P131", "P17", "P31"]

def test_wikidata_facts_top_properties_falls_back_to_property_id(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P9999",
                "property_label_en": "",
                "property_labels": "{}",
                "value_type": "string",
                "value_entity_id": "",
                "value_label_en": "",
                "value_labels": "{}",
                "value_text": "n/a",
                "qualifiers": "{}",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.top_properties[0][0] == "P9999"

def test_wikidata_malformed_qualifiers_count_as_unavailable(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_facts(
        processed / "wikidata" / "facts" / "monaco-latest.parquet",
        [
            {
                "fact_id": "f1",
                "wikidata": "Q1",
                "property_id": "P17",
                "property_label_en": "country",
                "property_labels": '{"en": "country"}',
                "value_type": "wikibase-entityid",
                "value_entity_id": "Q2",
                "value_label_en": "x",
                "value_labels": '{"en": "x"}',
                "value_text": "Q2",
                "qualifiers": "{this is not valid json",
                "references": "[]",
            },
        ],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikidata_facts.with_qualifiers == 0
    assert stats.wikidata_facts.unavailable_qualifiers == 1

def test_storage_bytes_separate_core_augmentation_total(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    polygons_path = _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    wiki_doc_path = _write_documents(
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
    stats = _stats(processed, tmp_path)
    expected_core = polygons_path.stat().st_size
    expected_aug = wiki_doc_path.stat().st_size
    assert stats.core_parquet_bytes == expected_core
    assert stats.augmentation_parquet_bytes == expected_aug
    assert stats.total_parquet_bytes == expected_core + expected_aug

def test_compute_augmentation_stats_handles_missing_directories(tmp_path: Path) -> None:
    """Missing sidecar sub-directories surface "No data exists yet".

    The runtime path produces a present-but-zero ProjectTextStats for
    the four document/section kinds and an empty WikidataFactStats.
    """
    import shutil

    processed = _setup_processed_dir(tmp_path)
    shutil.rmtree(processed / "wikipedia")
    shutil.rmtree(processed / "wikivoyage")
    shutil.rmtree(processed / "wikidata")
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 1
    assert stats.not_augmented_count == 1
    assert stats.fully_augmented_count == 0
    # All four text aggregations and the facts aggregation are present
    # but zeroed.
    assert stats.wikidata_facts.rows == 0
    assert stats.wikipedia_documents.rows == 0
    assert stats.wikipedia_documents.region_count == 0
    assert stats.wikivoyage_sections.rows == 0

def test_compute_augmentation_stats_handles_empty_sidecar_dirs(tmp_path: Path) -> None:
    """Sidecar dirs that exist but contain no parquet must not crash."""
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.wikipedia_documents.rows == 0
    assert stats.wikidata_facts.rows == 0

def test_compute_augmentation_stats_skips_unreadable_sidecar(
    tmp_path: Path,
    caplog: logging.LogCaptureFixture,
) -> None:
    """An unreadable Parquet file is counted as unreadable and skipped.

    The bytes still count toward ``augmentation_parquet_bytes`` so the
    storage accounting invariant
    ``core + augmentation == total`` keeps holding.
    """
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    # Plant a corrupt parquet by writing garbage.
    bad_path = processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    bad_path.write_bytes(b"not a parquet at all")
    caplog.set_level(logging.WARNING)
    stats = _stats(processed, tmp_path)
    # We don't read the table a second time; the unreadable count is
    # collected during the primary scan.
    assert stats.unreadable_file_count == 1
    assert any("Skipping" in r.getMessage() for r in caplog.records)
    # Storage bytes still include the corrupt file.
    assert stats.augmentation_parquet_bytes == bad_path.stat().st_size
    # But the rows are not aggregated.
    assert stats.wikipedia_documents.rows == 0

def test_compute_augmentation_stats_records_one_region_per_core_stem(tmp_path: Path) -> None:
    processed = _setup_processed_dir(tmp_path)
    _write_parquet(
        processed / "polygons" / "monaco-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q1"}],
    )
    _write_parquet(
        processed / "polygons" / "albania-latest.parquet",
        ["wikidata"],
        [{"wikidata": "Q2"}],
    )
    stats = _stats(processed, tmp_path)
    assert stats.core_region_count == 2
    assert stats.not_augmented_count == 2

def test_render_stats_section_without_augmentation_stats_unchanged() -> None:
    """render_stats_section(stats) (no augmentation kwarg) must produce
    the exact previous output."""
    stats = _empty_dataset_stats()
    md = render_stats_section(stats)
    assert "## Dataset snapshot" in md
    assert "## Wikipedia coverage funnel" in md
    assert "## Language distribution" in md
    assert "## Wikipedia text corpus" not in md
    assert "## Wikidata facts" not in md
    assert "## Augmentation coverage" not in md

def test_render_stats_section_with_augmentation_adds_new_sections() -> None:
    """render_stats_section(stats, augmentation_stats=...) MUST append
    the new sections in the documented order."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "## Wikipedia coverage funnel" not in md
    assert "## Augmentation coverage" not in md
    assert "## Storage accounting" in md
    assert "## Wikipedia text corpus" in md
    assert "## Wikivoyage text corpus" in md
    assert "## Wikidata facts" in md
    assert (
        md.index("## Storage accounting")
        < md.index("## Wikipedia text corpus")
        < md.index("## Wikivoyage text corpus")
        < md.index("## Wikidata facts")
    )

def test_render_stats_section_legacy_three_sections_byte_identical() -> None:
    """The legacy three sections must remain byte-identical when
    augmentation is provided.

    Equality is asserted directly on the rendered string, not via a
    substring prefix; only the headline table grows when augmentation
    is supplied, so the legacy-only output plus the first part of the
    with-augmentation output share exactly the same byte sequence up
    to the augmentation-specific rows of the headline table.
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
    no_aug = render_stats_section(stats)
    with_aug = render_stats_section(
        stats,
        augmentation_stats=AugmentationStats(
            core_region_count=1,
            fully_augmented_count=1,
            partial_augmented_count=0,
            not_augmented_count=0,
            orphan_sidecar_stems=[],
            wikipedia_documents=ProjectTextStats(),
            wikipedia_sections=ProjectTextStats(),
            wikivoyage_documents=ProjectTextStats(),
            wikivoyage_sections=ProjectTextStats(),
            wikidata_facts=WikidataFactStats(),
            core_parquet_bytes=10,
            augmentation_parquet_bytes=20,
            total_parquet_bytes=30,
            unreadable_file_count=0,
        ),
    )
    # The legacy last headline row is "Dataset size on disk | 4.0 KB
    # |". The augmentation-aware render renames the label to "Core
    # tables size". The numeric suffix ("| 4.0 KB |") is unchanged.
    legacy_label_row = "| Dataset size on disk |"
    aug_label_row = "| Polygon and link tables size |"
    legacy_offset = no_aug.index(legacy_label_row) + len(legacy_label_row)
    aug_offset = with_aug.index(aug_label_row) + len(aug_label_row)
    # The legacy render keeps the redundant "Wikipedia articles" and
    # "Total words" rows; the augmentation render drops them. Both
    # share the same leading rows (Polygons, Unique Wikidata entities)
    # and the same language-distribution section content; only the
    # headline's middle rows diverge by design.
    assert "| Wikipedia articles |" in no_aug
    assert "| Wikipedia articles |" not in with_aug
    assert "| Total words |" in no_aug
    assert "| Total words |" not in with_aug
    # The numeric tail (" 4.0 KB |") is identical in both versions.
    legacy_suffix = no_aug[legacy_offset : legacy_offset + len(" 4.0 KB |")]
    aug_suffix = with_aug[aug_offset : aug_offset + len(" 4.0 KB |")]
    assert legacy_suffix == aug_suffix == " 4.0 KB |"
    # The legacy label must NOT appear in the augmentation render.
    assert legacy_label_row not in with_aug

def test_render_stats_section_storage_bytes_labels() -> None:
    """The rendered sections must label storage bytes with the new wording
    pinned by the task."""
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=0,
        partial_augmented_count=1,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=4096,
        augmentation_parquet_bytes=2048,
        total_parquet_bytes=6144,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    assert "Wikipedia, Wikivoyage, and Wikidata tables size" in md
    assert "Total Parquet size" in md
    # The public polygon/link size label is used in the storage block
    # table (not the headline row). It must appear there.
    assert "Polygon and link tables size |" in md

def test_render_stats_section_legacy_storage_bytes_label() -> None:
    """Without augmentation, the storage-bytes wording is NOT rendered.

    The legacy sections do not contain any storage accounting at all.
    """
    stats = _empty_dataset_stats()
    md = render_stats_section(stats)
    assert "Core tables size" not in md
    assert "Augmentation tables size" not in md
    assert "Total Parquet size" not in md

def test_render_stats_section_distinguishes_missing_vs_present_empty() -> None:
    """A sub-directory that does not exist OR exists but holds zero
    Parquet files renders "No data exists yet."

    A sub-directory with at least one readable, valid zero-row Parquet
    sidecar renders "This sidecar is present but empty."
    """
    import tempfile

    stats = _empty_dataset_stats()
    with tempfile.TemporaryDirectory() as tmp:
        missing_path = Path(tmp) / "missing"
        empty_path = Path(tmp) / "empty"
        aug_missing = compute_augmentation_stats(
            _setup_missing_processed(missing_path),
            cache_index_dir=missing_path / "cache",
        )
        aug_empty = compute_augmentation_stats(
            _setup_processed_dir(empty_path),
            cache_index_dir=empty_path / "cache",
        )
    md_missing = render_stats_section(stats, augmentation_stats=aug_missing)
    md_empty = render_stats_section(stats, augmentation_stats=aug_empty)

    # Both fixtures have NO readable Parquet anywhere inside the
    # augmentation sub-directories, so both are "No data exists yet."
    assert md_missing.count("No data exists yet.") >= 5
    assert md_empty.count("No data exists yet.") >= 5
    assert md_empty.count("This sidecar is present but empty.") == 0
    # Headline rows still rendered for the empty case (zero rows are
    # a valid metric).
    assert "| Wikipedia documents | 0 |" in md_empty
    assert "| Wikipedia sections | 0 |" in md_empty

def test_render_stats_section_present_zero_row_sidecar_distinct(tmp_path: Path) -> None:
    """A readable, valid zero-row parquet sidecar renders "present but empty".

    The presence of a real (zero-row) parquet flips the renderer from
    "No data exists yet." to "This sidecar is present but empty."
    while leaving headline rows unchanged.
    """
    processed_dir = _setup_processed_dir_with_zero_row_parquets(tmp_path)
    aug = compute_augmentation_stats(
        processed_dir,
        cache_index_dir=tmp_path / "cache",
    )
    md = render_stats_section(_empty_dataset_stats(), augmentation_stats=aug)
    assert md.count("This sidecar is present but empty.") >= 5
    assert md.count("No data exists yet.") == 0

def test_render_stats_section_sections_table_row_layout() -> None:
    """The Sections metrics row closes its markdown row with a final `|`.

    Also verify the exact byte layout for the four section-only rows
    to catch any malformed markdown row regressions.
    """
    stats = _empty_dataset_stats()
    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(
            subdir_present=True,
            rows=4,
            unique_section_ids=4,
            unique_documents=2,
            region_count=2,
            avg_sections_per_doc=2.0,
            non_empty=3,
            empty_or_null=1,
            non_empty_rate=0.75,
            total_words=8,
        ),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    # The 'Avg sections per represented document' row must close
    # with a final '|'.
    avg_line = "| Avg sections per represented document | 2.00 |"
    assert avg_line in md, f"Malformed markdown row, missing final '|': {avg_line!r}"
    # The non-empty section rate row must close with a final '|'.
    rate_line = "| Non-empty section rate | 75.0% |"
    assert rate_line in md, f"Malformed markdown row, missing final '|': {rate_line!r}"
    # Unique sections and Documents represented are distinct rows.
    assert "| Unique sections | 4 |" in md
    assert "| Documents represented | 2 |" in md
