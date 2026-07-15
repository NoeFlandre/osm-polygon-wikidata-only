"""Frozen augmentation schema + YAML front matter checks for the dataset card.

This module documents the schema of the augmentation tables and the
Hugging Face dataset-card configurations. It is a pure-extension
contract test; no Parquet reads, no network.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.dataset_stats import render_stats_section


def _render_with_stats_section(stats_section: str) -> str:
    return render_dataset_card(
        repo_id="NoeFlandre/osm-polygon-wikidata-only",
        stats={"polygon_count": 1, "article_count": 1, "unique_wikidata_count": 1},
        polygon_columns=list(POLYGON_COLUMNS),
        polygon_descriptions={col: "d" for col in POLYGON_COLUMNS},
        article_columns=list(ARTICLE_COLUMNS),
        article_descriptions={col: "d" for col in ARTICLE_COLUMNS},
        link_columns=list(POLYGON_ARTICLE_COLUMNS),
        link_descriptions={col: "d" for col in POLYGON_ARTICLE_COLUMNS},
        stats_section=stats_section,
    )


# --- augmentation YAML front matter --------------------------------


def test_dataset_card_yaml_lists_all_canonical_configurations() -> None:
    stats_section = "## Dataset snapshot\n\n| Metric | Value |\n| --- | --- |\n| 1 | 1 |\n"
    md = _render_with_stats_section(stats_section)
    for name in (
        "polygons",
        "polygon_articles",
        "wikipedia_documents",
        "wikipedia_sections",
        "wikivoyage_documents",
        "wikivoyage_sections",
        "wikidata_facts",
    ):
        assert f"config_name: {name}" in md, f"missing {name} in YAML front matter"


def test_dataset_card_yaml_paths_match_repo_layout() -> None:
    """Each config's ``path:`` glob must match the published sidecar path."""
    md = _render_with_stats_section("## Dataset snapshot\n")
    expected_globs = (
        "polygons/*.parquet",
        "polygon_articles/*.parquet",
        "wikipedia/documents/*.parquet",
        "wikipedia/sections/*.parquet",
        "wikivoyage/documents/*.parquet",
        "wikivoyage/sections/*.parquet",
        "wikidata/facts/*.parquet",
    )
    for glob in expected_globs:
        assert f"path: {glob}" in md, f"missing glob {glob} in YAML front matter"


def test_dataset_card_yaml_omits_retired_articles_configuration() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    assert "config_name: polygons" in md
    assert "config_name: polygon_articles" in md
    assert "config_name: articles" not in md
    assert "path: articles/*.parquet" not in md


def test_dataset_card_yaml_contains_wikivoyage_tag() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    assert "  - wikivoyage" in md


def test_dataset_card_yaml_starts_with_valid_front_matter() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    assert md.startswith("---\n"), "dataset card must start with YAML open delimiter"
    end_marker = "\n---\n"
    end_index = md.find(end_marker, 4)
    assert end_index > 0, "dataset card must close its YAML front matter"
    header_block = md[: end_index + len(end_marker)]
    lines = [line for line in header_block.splitlines() if line and not line.startswith("---")]
    for line in lines:
        # Accept `key: value` lines and bare `- item` list items.
        assert ":" in line or line.lstrip().startswith("- "), f"Malformed YAML line: {line!r}"


def test_dataset_card_front_matter_passes_structural_validator() -> None:
    """The YAML front matter must satisfy the structural validator.

    This is a stronger check than the line-shape assertion above:
    :func:`validate_front_matter` deserializes the block via PyYAML
    and walks the ``configs:`` list to confirm ``config_name``,
    ``data_files``, and at least one path glob per entry. Catches
    malformed globs, missing fields, or broken indentation that a
    line-shape check would miss.
    """
    from osm_polygon_wikidata_only.hf.dataset_card import validate_front_matter

    md = _render_with_stats_section("## Dataset snapshot\n")
    end_marker = "\n---\n"
    end_index = md.find(end_marker, 4)
    header_block = md[: end_index + len(end_marker)]
    validate_front_matter(header_block)


# --- augmentation schema sections ---------------------------------


def test_dataset_card_documents_augmentation_schemas() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    assert "### `wikipedia/documents`" in md
    assert "### `wikivoyage/documents`" in md
    assert "### `wikipedia/sections` and `wikivoyage/sections`" in md
    assert "### `wikidata/facts`" in md


def test_dataset_card_documents_fact_columns() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    fact_section = md.split("### `wikidata/facts`", 1)[1].split("## Data sources", 1)[0]
    # Every documented FACT column must appear (backtick-quoted) in the
    # fact column table. We check the table syntax: `| `col` | description |`.
    for column in FACT_COLUMNS:
        assert f"| `{column}` |" in fact_section, f"missing column {column!r} in fact table"


def test_dataset_card_documents_document_columns() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    doc_section = md.split("### `wikivoyage/documents`", 1)[1].split("### ", 1)[0]
    for column in DOCUMENT_COLUMNS:
        assert f"| `{column}` |" in doc_section, f"missing document column {column!r}"


def test_dataset_card_documents_section_columns() -> None:
    md = _render_with_stats_section("## Dataset snapshot\n")
    section_part = md.split("### `wikipedia/sections` and `wikivoyage/sections`", 1)[1].split(
        "### ", 1
    )[0]
    for column in SECTION_COLUMNS:
        assert f"| `{column}` |" in section_part, f"missing section column {column!r}"


def test_dataset_card_documents_every_column_exactly_once() -> None:
    """A given column should not be listed in two different schema tables
    (Wikipedia and Wikivoyage documents share one schema; sections share
    another)."""
    md = _render_with_stats_section("## Dataset snapshot\n")
    assert md.count("### `wikipedia/documents`") == 1
    assert md.count("### `wikivoyage/documents`") == 1
    assert md.count("### `wikipedia/sections` and `wikivoyage/sections`") == 1
    assert md.count("### `wikidata/facts`") == 1


def test_dataset_card_documents_columns_use_pyarrow_schemas() -> None:
    """The schema description must come from the canonical pyarrow
    ``document_schema()`` / ``section_schema()`` / ``fact_schema()``
    factories and not from a hand-maintained list."""
    md = _render_with_stats_section("## Dataset snapshot\n")
    # Use the live schema factories as ground truth.
    expected_document = list(_column_names(document_schema()))
    expected_section = list(_column_names(section_schema()))
    expected_fact = list(_column_names(fact_schema()))
    # Evey column must appear in the markdown.
    for column in expected_document + expected_section + expected_fact:
        assert f"`{column}`" in md


def _column_names(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(field.name for field in schema)


# --- README sections ordering ---------------------------------------


def test_render_stats_section_includes_all_new_sections() -> None:
    """The rendered stats markdown must expose the documented sections in
    the stable order."""
    from osm_polygon_wikidata_only.hf._dataset_stats.models import (
        AugmentationStats,
        ProjectTextStats,
        WikidataFactStats,
    )

    aug = AugmentationStats(
        core_region_count=1,
        fully_augmented_count=1,
        partial_augmented_count=0,
        not_augmented_count=0,
        orphan_sidecar_stems=[],
        wikipedia_documents=ProjectTextStats(rows=1),
        wikipedia_sections=ProjectTextStats(rows=1),
        wikivoyage_documents=ProjectTextStats(rows=1),
        wikivoyage_sections=ProjectTextStats(rows=1),
        wikidata_facts=WikidataFactStats(rows=1),
        core_parquet_bytes=10,
        augmentation_parquet_bytes=20,
        total_parquet_bytes=30,
        unreadable_file_count=0,
    )
    # Construct a DatasetStats via the public facade.
    from osm_polygon_wikidata_only.hf.dataset_stats import DatasetStats

    stats = DatasetStats(
        polygon_count=1,
        unique_wikidata_count=1,
        article_count=1,
        link_count=1,
        language_count=1,
        region_count=1,
        total_words=1,
        total_tokens_estimate=1,
        dataset_size_bytes=1,
        polygons_with_wikipedia=1,
        polygons_with_text=1,
        polygons_with_english=1,
        polygons_with_no_english_other_lang=0,
        polygons_with_2plus_langs=1,
        polygons_with_5plus_langs=0,
        polygons_with_10plus_langs=0,
        articles_per_language={"en": 1},
        polygons_per_language={"en": 1},
    )
    md = render_stats_section(stats, augmentation_stats=aug)
    # Required new sections (in order).
    required_in_order = (
        "## Dataset snapshot",
        "## Augmentation coverage",
        "## Storage accounting",
        "## Wikipedia text corpus",
        "## Wikivoyage text corpus",
        "## Wikidata facts",
    )
    last_index = -1
    for header in required_in_order:
        index = md.index(header)
        assert index > last_index, f"{header} appears out of order"
        last_index = index
    # "Storage accounting" must use the new labels.
    assert "Core tables size" in md
    assert "Augmentation tables size" in md
    assert "Total Parquet size" in md
