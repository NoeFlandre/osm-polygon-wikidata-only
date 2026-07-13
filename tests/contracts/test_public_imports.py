"""Frozen import surface derived from ``docs/api.md``.

These tests assert that every documented public import is importable
through its facade module and remains identity-equal to the focused
implementation. They extend ``tests/test_public_api.py`` with broader
coverage of the public surface.
"""

from __future__ import annotations


def test_config_paths_facade_identity() -> None:
    from osm_polygon_wikidata_only.config import paths as facade
    from osm_polygon_wikidata_only.config.paths import (
        DataRoot as FocusedDataRoot,
    )
    from osm_polygon_wikidata_only.config.paths import (
        resolve_data_root as focused_resolve_data_root,
    )

    assert facade.DataRoot is FocusedDataRoot
    assert facade.resolve_data_root is focused_resolve_data_root


def test_config_settings_facade_identity() -> None:
    from osm_polygon_wikidata_only.config import settings as facade
    from osm_polygon_wikidata_only.config.settings import Settings as FocusedSettings

    assert facade.Settings is FocusedSettings
    assert facade.Settings.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    assert isinstance(facade.DEFAULT_REPO_ID, str)
    assert isinstance(facade.DEFAULT_USER_AGENT, str)


def test_enrichment_wikidata_client_public_names_identity() -> None:
    from osm_polygon_wikidata_only.enrichment import wikidata_client as facade
    from osm_polygon_wikidata_only.enrichment.wikidata.models import (
        BatchWikidataClient as FocusedBatch,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.models import (
        Sitelinks as FocusedSitelinks,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.models import (
        WikidataClient as FocusedClient,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.models import (
        WikidataEntity as FocusedEntity,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        is_valid_qid as focused_is_valid_qid,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        language_from_site as focused_language_from_site,
    )
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
        parse_wikidata_entity as focused_parser,
    )

    # Identity assertions for every name a refactor must preserve.
    assert facade.WikidataEntity is FocusedEntity
    assert facade.WikidataClient is FocusedClient
    assert facade.BatchWikidataClient is FocusedBatch
    assert facade.Sitelinks is FocusedSitelinks
    assert facade.is_valid_qid is focused_is_valid_qid
    assert facade.language_from_site is focused_language_from_site
    assert facade.parse_wikidata_entity is focused_parser

    # The remaining facade names must at least be importable.
    for name in (
        "CachedWikidataClient",
        "HttpWikidataClient",
        "InMemoryWikidataClient",
        "WikidataError",
    ):
        assert hasattr(facade, name), f"missing documented name on wikidata_client: {name}"


def test_enrichment_wikipedia_client_public_names_identity() -> None:
    from osm_polygon_wikidata_only.enrichment import wikipedia_client as facade
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        BatchWikipediaClient as FocusedBatch,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        FetchResult as FocusedFetch,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        WikipediaArticle as FocusedArticle,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
        WikipediaClient as FocusedClient,
    )
    from osm_polygon_wikidata_only.enrichment.wikipedia.parsing import (
        parse_wikipedia_response as focused_parser,
    )

    assert facade.FetchResult is FocusedFetch
    assert facade.WikipediaArticle is FocusedArticle
    assert facade.WikipediaClient is FocusedClient
    assert facade.BatchWikipediaClient is FocusedBatch
    assert facade.parse_wikipedia_response is focused_parser

    for name in (
        "CachedWikipediaClient",
        "HttpWikipediaClient",
        "InMemoryWikipediaClient",
    ):
        assert hasattr(facade, name), f"missing documented name on wikipedia_client: {name}"


def test_pipeline_processor_facade_identity() -> None:
    from osm_polygon_wikidata_only.pipeline import processor
    from osm_polygon_wikidata_only.pipeline.completeness import (
        IncompleteEnrichmentError as FocusedError,
    )
    from osm_polygon_wikidata_only.pipeline.processor import ExtractedPbf as FocusedExtracted
    from osm_polygon_wikidata_only.pipeline.processor import PbfStem as FocusedStem
    from osm_polygon_wikidata_only.pipeline.processor import (
        ProcessResult as FocusedResult,
    )

    assert processor.PbfStem is FocusedStem
    assert processor.ProcessResult is FocusedResult
    assert processor.ExtractedPbf is FocusedExtracted
    assert processor.IncompleteEnrichmentError is FocusedError
    assert callable(processor.process_pbf)
    assert callable(processor.process_extracted_pbf)
    assert callable(processor.extract_pbf)


def test_pipeline_orchestrator_facade_identity() -> None:
    from osm_polygon_wikidata_only.pipeline import orchestrator

    assert callable(orchestrator.orchestrate)
    assert callable(orchestrator.collect_pbfs)
    assert callable(orchestrator.already_processed)


def test_hf_dataset_card_facade_identity() -> None:
    from osm_polygon_wikidata_only.hf import dataset_card as facade
    from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card as focused

    assert facade.render_dataset_card is focused


def test_hf_uploader_facade_identity() -> None:
    from osm_polygon_wikidata_only.hf import uploader as facade
    from osm_polygon_wikidata_only.hf.uploader import HfHub as FocusedHfHub
    from osm_polygon_wikidata_only.hf.uploader import StubHfHub as FocusedStub
    from osm_polygon_wikidata_only.hf.uploader import UploadError as FocusedError
    from osm_polygon_wikidata_only.hf.uploader import (
        default_commit_message as focused_commit,
    )
    from osm_polygon_wikidata_only.hf.uploader import (
        resolve_hf_token as focused_resolve,
    )
    from osm_polygon_wikidata_only.hf.uploader import upload_card as focused_upload_card
    from osm_polygon_wikidata_only.hf.uploader import upload_files as focused_upload_files
    from osm_polygon_wikidata_only.hf.uploader import (
        upload_manifest as focused_upload_manifest,
    )
    from osm_polygon_wikidata_only.hf.uploader import (
        upload_parquet as focused_upload_parquet,
    )
    from osm_polygon_wikidata_only.hf.uploader import verify_hf_token as focused_verify
    from osm_polygon_wikidata_only.hf.uploader import (
        verify_repo_authorization as focused_verify_repo,
    )

    assert facade.HfHub is FocusedHfHub
    assert facade.StubHfHub is FocusedStub
    assert facade.UploadError is FocusedError
    assert facade.default_commit_message is focused_commit
    assert facade.resolve_hf_token is focused_resolve
    assert facade.upload_card is focused_upload_card
    assert facade.upload_files is focused_upload_files
    assert facade.upload_manifest is focused_upload_manifest
    assert facade.upload_parquet is focused_upload_parquet
    assert facade.verify_hf_token is focused_verify
    assert facade.verify_repo_authorization is focused_verify_repo


def test_hf_coverage_map_facade_callables() -> None:
    from osm_polygon_wikidata_only.hf import coverage_map

    assert callable(coverage_map.generate_coverage_map)
    assert callable(coverage_map.ensure_world_land)
    assert callable(coverage_map.load_centroids_from_parquet)


def test_hf_geographic_text_coverage_facade_identity() -> None:
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade

    expected = {
        "CoverageCell",
        "CoverageMapError",
        "DEFAULT_H3_RESOLUTION",
        "DEFAULT_MIN_POLYGONS_PER_CELL",
        "LOCAL_ASSET_PATH",
        "LOCAL_POLYGON_COUNT_ASSET_PATH",
        "LOCAL_TEXT_COVERAGE_ASSET_PATH",
        "PolygonCountCell",
        "REMOTE_ASSET_PATH",
        "REMOTE_POLYGON_COUNT_ASSET_PATH",
        "REMOTE_TEXT_COVERAGE_ASSET_PATH",
        "RenderResult",
        "assign_h3_cell",
        "generate_geographic_polygon_count",
        "generate_geographic_text_coverage",
    }
    missing = expected - set(dir(facade))
    assert not missing, f"missing documented names on geographic_text_coverage: {missing}"
    # Spot-check one constant value for identity preservation.
    assert facade.LOCAL_TEXT_COVERAGE_ASSET_PATH == "assets/geographic_wikipedia_text_coverage.png"
    assert facade.REMOTE_TEXT_COVERAGE_ASSET_PATH == "assets/geographic_wikipedia_text_coverage.png"


# ---------------------------------------------------------------------------
# Frozen ``__all__`` equality contracts.
#
# Each facade's ``__all__`` must equal the Phase 1 frozen public list -- the
# exact set of names the module publicly advertises. This is a stricter
# check than the subset assertions above: a refactor must not silently add
# names to (or remove names from) a facade's ``__all__``.
# ---------------------------------------------------------------------------


def test_config_paths_public_names_match_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.config import paths as facade

    frozen = {"DataRoot", "resolve_data_root"}
    assert set(facade.__dict__) | set(getattr(facade, "__all__", ())) >= frozen
    # ``config.paths`` does not define ``__all__``; the Phase 1 contract is
    # a documented-name subset, so the frozen list is enforced against the
    # public attributes instead.
    for name in frozen:
        assert hasattr(facade, name), f"missing Phase 1 name on config.paths: {name}"


def test_config_settings_public_names_match_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.config import settings as facade

    frozen = {"Settings", "DEFAULT_REPO_ID", "DEFAULT_USER_AGENT"}
    for name in frozen:
        assert hasattr(facade, name), f"missing Phase 1 name on config.settings: {name}"


def test_enrichment_wikidata_client_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.enrichment import wikidata_client as facade

    frozen = {
        "BatchWikidataClient",
        "CachedWikidataClient",
        "HttpWikidataClient",
        "InMemoryWikidataClient",
        "Sitelinks",
        "WikidataClient",
        "WikidataEntity",
        "WikidataError",
        "is_valid_qid",
        "language_from_site",
        "parse_wikidata_entity",
    }
    assert set(facade.__all__) == frozen


def test_enrichment_wikipedia_client_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.enrichment import wikipedia_client as facade

    frozen = {
        "BatchWikipediaClient",
        "CachedWikipediaClient",
        "FetchResult",
        "HttpWikipediaClient",
        "InMemoryWikipediaClient",
        "WikipediaArticle",
        "WikipediaClient",
        "parse_wikipedia_response",
    }
    assert set(facade.__all__) == frozen


def test_pipeline_processor_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.pipeline import processor as facade

    frozen = {
        "ExtractedPbf",
        "IncompleteEnrichmentError",
        "PbfStem",
        "ProcessResult",
        "extract_pbf",
        "process_extracted_pbf",
        "process_pbf",
    }
    assert set(facade.__all__) == frozen


def test_pipeline_orchestrator_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.pipeline import orchestrator as facade

    frozen = {"already_processed", "collect_pbfs", "orchestrate"}
    assert set(facade.__all__) == frozen


def test_hf_dataset_card_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.hf import dataset_card as facade

    frozen = {"render_dataset_card"}
    assert set(facade.__all__) == frozen


def test_hf_uploader_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.hf import uploader as facade

    frozen = {
        "HfHub",
        "StubHfHub",
        "UploadError",
        "default_commit_message",
        "resolve_hf_token",
        "upload_card",
        "upload_files",
        "upload_manifest",
        "upload_parquet",
        "verify_hf_token",
        "verify_repo_authorization",
    }
    assert set(facade.__all__) == frozen


def test_hf_coverage_map_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.hf import coverage_map as facade

    frozen = {
        "WORLD_LAND_FILENAME",
        "ensure_world_land",
        "generate_coverage_map",
        "load_centroids_from_parquet",
    }
    assert set(facade.__all__) == frozen


def test_hf_geographic_text_coverage_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.hf import geographic_text_coverage as facade

    frozen = {
        "DEFAULT_H3_RESOLUTION",
        "DEFAULT_MIN_POLYGONS_PER_CELL",
        "LOCAL_ASSET_PATH",
        "LOCAL_POLYGON_COUNT_ASSET_PATH",
        "LOCAL_TEXT_COVERAGE_ASSET_PATH",
        "REMOTE_ASSET_PATH",
        "REMOTE_POLYGON_COUNT_ASSET_PATH",
        "REMOTE_TEXT_COVERAGE_ASSET_PATH",
        "CoverageCell",
        "CoverageMapError",
        "PolygonCountCell",
        "RenderResult",
        "aggregate_geographic_polygon_count",
        "aggregate_geographic_text_coverage",
        "assign_h3_cell",
        "generate_geographic_polygon_count",
        "generate_geographic_text_coverage",
        "render_geographic_polygon_count",
        "render_geographic_text_coverage",
    }
    assert set(facade.__all__) == frozen


def test_cli_parser_all_equals_phase_1_frozen_list() -> None:
    from osm_polygon_wikidata_only.cli import parser as facade

    frozen = {"build_parser", "build_settings", "parse_languages"}
    assert set(facade.__all__) == frozen
