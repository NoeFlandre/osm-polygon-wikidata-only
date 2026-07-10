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


def test_cli_facade_preserves_parser() -> None:
    from osm_polygon_wikidata_only.cli.commands import build_parser
    from osm_polygon_wikidata_only.cli.parser import build_parser as focused_build_parser

    assert build_parser is focused_build_parser
