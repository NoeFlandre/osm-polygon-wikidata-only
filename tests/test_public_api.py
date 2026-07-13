"""Compatibility tests for the documented Python surface."""

from __future__ import annotations


def test_wikipedia_facade_preserves_public_types() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        FetchResult as FocusedFetchResult,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        WikipediaArticle as FocusedArticle,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        FetchResult,
        WikipediaArticle,
    )

    assert FetchResult is FocusedFetchResult
    assert WikipediaArticle is FocusedArticle


def test_wikipedia_facade_preserves_public_parser() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia.parsing import (
        parse_wikipedia_response as focused_parser,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import (
        parse_wikipedia_response,
    )

    assert parse_wikipedia_response is focused_parser


def test_wikidata_facade_preserves_public_types() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata.models import (
        WikidataEntity as FocusedEntity,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataEntity

    assert WikidataEntity is FocusedEntity


def test_wikidata_facade_preserves_public_parsing_helpers() -> None:
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        is_valid_qid as focused_is_valid_qid,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        language_from_site as focused_language_from_site,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        parse_wikidata_entity as focused_parser,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata_client import (
        is_valid_qid,
        language_from_site,
        parse_wikidata_entity,
    )

    assert is_valid_qid is focused_is_valid_qid
    assert language_from_site is focused_language_from_site
    assert parse_wikidata_entity is focused_parser


def test_processor_facade_preserves_public_error() -> None:
    from osm_polygon_wikidata_only.pipeline.completeness import (
        IncompleteEnrichmentError as FocusedError,
    )
    from osm_polygon_wikidata_only.pipeline.processor import IncompleteEnrichmentError

    assert IncompleteEnrichmentError is FocusedError


def test_processor_facade_re_exports_extractor_symbols() -> None:
    """``pipeline.processor`` must re-export ``extract_pbf`` and
    ``ExtractedPbf`` from the focused extractor module."""
    from osm_polygon_wikidata_only.pipeline import extractor as extractor_mod
    from osm_polygon_wikidata_only.pipeline import processor as processor_mod

    assert processor_mod.extract_pbf is extractor_mod.extract_pbf
    assert processor_mod.ExtractedPbf is extractor_mod.ExtractedPbf


def test_rows_facade_re_exports_row_construction() -> None:
    """``pipeline.rows`` keeps backwards-compatible re-exports of the
    three row-construction helpers by identity."""
    from osm_polygon_wikidata_only.pipeline import row_construction as focused
    from osm_polygon_wikidata_only.pipeline import rows as legacy

    assert legacy.enrich_polygon is focused.enrich_polygon
    assert legacy.build_articles_and_links is focused.build_articles_and_links
    assert legacy.article_row is focused.article_row


def test_enrichment_phase_owns_unique_qids_helper() -> None:
    """The ``unique_qids`` helper exposes a deterministic QID tuple."""
    from dataclasses import dataclass

    from osm_polygon_wikidata_only.pipeline.enrichment_phase import unique_qids

    @dataclass(slots=True)
    class _Stub:
        wikidata: str

    assert unique_qids([_Stub(wikidata="Q3"), _Stub(wikidata="Q1"), _Stub(wikidata="Q1")]) == (
        "Q1",
        "Q3",
    )


def test_cli_facade_preserves_parser() -> None:
    from osm_polygon_wikidata_only.cli.commands import build_parser
    from osm_polygon_wikidata_only.cli.parser import build_parser as focused_build_parser

    assert build_parser is focused_build_parser


def test_geographic_facade_preserves_public_types() -> None:
    """The geographic facade must re-export the four documented types
    from their focused modules unchanged."""
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade
    from osm_polygon_wikidata_only.hf._geographic.models import (
        CoverageCell,
        CoverageMapError,
        PolygonCountCell,
        RenderResult,
    )

    assert facade.CoverageCell is CoverageCell
    assert facade.PolygonCountCell is PolygonCountCell
    assert facade.RenderResult is RenderResult
    assert facade.CoverageMapError is CoverageMapError


def test_geographic_facade_preserves_aggregation_helpers() -> None:
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade
    from osm_polygon_wikidata_only.hf._geographic.aggregation import (
        aggregate_geographic_polygon_count as focused_count_agg,
    )
    from osm_polygon_wikidata_only.hf._geographic.aggregation import (
        aggregate_geographic_text_coverage as focused_text_agg,
    )

    assert facade.aggregate_geographic_text_coverage is focused_text_agg
    assert facade.aggregate_geographic_polygon_count is focused_count_agg


def test_geographic_facade_preserves_rendering_helpers() -> None:
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade
    from osm_polygon_wikidata_only.hf._geographic.coverage import (
        generate_geographic_text_coverage as focused_text_generate,
    )
    from osm_polygon_wikidata_only.hf._geographic.coverage import (
        render_geographic_text_coverage as focused_text_render,
    )
    from osm_polygon_wikidata_only.hf._geographic.polygon_count import (
        generate_geographic_polygon_count as focused_count_generate,
    )
    from osm_polygon_wikidata_only.hf._geographic.polygon_count import (
        render_geographic_polygon_count as focused_count_render,
    )

    assert facade.render_geographic_text_coverage is focused_text_render
    assert facade.generate_geographic_text_coverage is focused_text_generate
    assert facade.render_geographic_polygon_count is focused_count_render
    assert facade.generate_geographic_polygon_count is focused_count_generate


def test_geographic_facade_preserves_assign_h3_cell_and_defaults() -> None:
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade
    from osm_polygon_wikidata_only.hf._geographic.h3_geometry import (
        DEFAULT_H3_RESOLUTION as focused_default_resolution,
    )
    from osm_polygon_wikidata_only.hf._geographic.h3_geometry import (
        DEFAULT_MIN_POLYGONS_PER_CELL as focused_default_min_polygons,
    )
    from osm_polygon_wikidata_only.hf._geographic.h3_geometry import (
        assign_h3_cell as focused_assign,
    )

    assert facade.assign_h3_cell is focused_assign
    assert facade.DEFAULT_H3_RESOLUTION is focused_default_resolution
    assert facade.DEFAULT_MIN_POLYGONS_PER_CELL is focused_default_min_polygons


def test_geographic_facade_preserves_asset_path_constants() -> None:
    """The stable asset paths and backwards-compatible aliases are stable
    by value and live on the facade."""
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade

    assert facade.REMOTE_TEXT_COVERAGE_ASSET_PATH == "assets/geographic_wikipedia_text_coverage.png"
    assert facade.LOCAL_TEXT_COVERAGE_ASSET_PATH == facade.REMOTE_TEXT_COVERAGE_ASSET_PATH
    assert facade.REMOTE_POLYGON_COUNT_ASSET_PATH == "assets/geographic_polygon_count.png"
    assert facade.LOCAL_POLYGON_COUNT_ASSET_PATH == facade.REMOTE_POLYGON_COUNT_ASSET_PATH
    assert facade.LOCAL_ASSET_PATH == facade.LOCAL_TEXT_COVERAGE_ASSET_PATH
    assert facade.REMOTE_ASSET_PATH == facade.REMOTE_TEXT_COVERAGE_ASSET_PATH


def test_dataset_stats_facade_preserves_public_symbols() -> None:
    """The dataset_stats facade must re-export the three documented
    public symbols unchanged."""
    from osm_polygon_wikidata_only.hf import dataset_stats as facade
    from osm_polygon_wikidata_only.hf._dataset_stats.aggregation import (
        compute_dataset_stats as focused_compute,
    )
    from osm_polygon_wikidata_only.hf._dataset_stats.models import (
        DatasetStats as focused_stats,
    )
    from osm_polygon_wikidata_only.hf._dataset_stats.rendering import (
        render_stats_section as focused_render,
    )

    assert facade.DatasetStats is focused_stats
    assert facade.compute_dataset_stats is focused_compute
    assert facade.render_stats_section is focused_render
