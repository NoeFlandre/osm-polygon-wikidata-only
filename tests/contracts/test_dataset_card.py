"""Frozen output of ``hf.dataset_card.render_dataset_card``.

The dataset card is a publication contract: every section, the YAML
front matter, the schema tables, the example loader snippet, and the
license / attribution blocks are stable. This test uses only
in-memory schema column lists, so it does not require any Parquet
files.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card

REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"


def _render() -> str:
    return render_dataset_card(
        repo_id=REPO_ID,
        stats={"polygon_count": 100, "article_count": 50, "unique_wikidata_count": 40},
        polygon_columns=list(POLYGON_COLUMNS),
        polygon_descriptions=POLYGON_DESCRIPTIONS,
        article_columns=list(ARTICLE_COLUMNS),
        article_descriptions=ARTICLE_DESCRIPTIONS,
        link_columns=list(POLYGON_ARTICLE_COLUMNS),
        link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
        maintainer="Noé Flandre",
    )


def test_card_has_yaml_front_matter() -> None:
    md = _render()
    assert md.startswith("---\n")
    assert "license: odbl" in md
    assert f"polygon_count: {100}" in md


def test_card_embeds_coverage_map_reference() -> None:
    md = _render()
    assert "![Coverage Map](assets/coverage_map.png)" in md


def test_card_starts_with_the_dataset_hero() -> None:
    md = _render()
    hero = "![NoeFlandre/osm-polygon-wikidata-only dataset overview](assets/dataset_hero.png)"
    assert hero in md
    assert md.index(hero) < md.index("OSM polygons tagged with")


def test_card_places_blog_post_at_the_top() -> None:
    md = _render()
    hero = "![NoeFlandre/osm-polygon-wikidata-only dataset overview](assets/dataset_hero.png)"
    story = "https://noeflandre.com/posts/describe-place-on-earth-part1-wikidata"
    intro = "OSM polygons tagged with"
    assert "Blog post: [How to describe a place on Earth with Wikidata]" in md
    assert "Project story:" not in md
    assert story in md
    assert md.index(hero) < md.index(story) < md.index(intro)


def test_card_places_trackio_snapshot_after_intro_before_coverage() -> None:
    md = _render()
    hero = "![NoeFlandre/osm-polygon-wikidata-only dataset overview](assets/dataset_hero.png)"
    intro = "OSM polygons tagged with"
    trackio = "## Trackio snapshot"
    coverage = "## Coverage"
    assert md.index(hero) < md.index(intro) < md.index(trackio) < md.index(coverage)


def test_card_embeds_geographic_assets() -> None:
    md = _render()
    assert "assets/geographic_text_density.png" in md
    assert "assets/geographic_wikipedia_text_coverage.png" not in md
    assert "assets/geographic_polygon_count.png" not in md


def test_card_documents_canonical_core_tables() -> None:
    md = _render()
    for table in ("polygons", "wikipedia/documents", "polygon_articles"):
        assert f"### `{table}`" in md


def test_card_includes_license_attribution() -> None:
    md = _render()
    assert "ODbL" in md
    assert "CC0" in md
    assert "CC BY-SA" in md


def test_card_links_to_source_repository() -> None:
    md = _render()
    assert "[GitHub repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only)" in md


def test_card_identifies_the_v1_github_contract() -> None:
    md = _render()
    assert "This Hugging Face dataset is **V1**" in md
    assert (
        "[V1.0.0 GitHub code](https://github.com/NoeFlandre/"
        "osm-polygon-wikidata-only/tree/v1.0.0)" in md
    )


def test_card_ends_with_dataset_citation_link() -> None:
    md = _render()
    assert md.rstrip().endswith(
        "Download the dataset citation metadata from [`CITATION.cff`](CITATION.cff)."
    )
    assert md.index("## Citation") > md.index("## How to load")


def test_card_includes_loader_snippet() -> None:
    md = _render()
    assert "load_dataset" in md
    assert REPO_ID in md


def test_card_includes_dataset_snapshot_when_stats_section_provided() -> None:
    md = render_dataset_card(
        repo_id=REPO_ID,
        stats={"polygon_count": 1, "article_count": 1, "unique_wikidata_count": 1},
        polygon_columns=list(POLYGON_COLUMNS),
        polygon_descriptions=POLYGON_DESCRIPTIONS,
        article_columns=list(ARTICLE_COLUMNS),
        article_descriptions=ARTICLE_DESCRIPTIONS,
        link_columns=list(POLYGON_ARTICLE_COLUMNS),
        link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
        stats_section="## Custom\n\nCustom body.\n",
    )
    assert "## Custom" in md
    assert "Custom body." in md


def test_card_lists_supported_augmentation_paths() -> None:
    md = _render()
    for path in (
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ):
        assert path in md
