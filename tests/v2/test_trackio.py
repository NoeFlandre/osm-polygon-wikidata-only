from __future__ import annotations

from osm_polygon_wikidata_only.hf.v2_trackio_snapshot import snapshot_from_v2_stats
from osm_polygon_wikidata_only.v2.card import V2CardStats
from osm_polygon_wikidata_only.v2.config import V2_TRACKIO_RUN_NAME


def test_v2_trackio_snapshot_uses_data_derived_metrics_only() -> None:
    stats = V2CardStats(
        regions=2,
        polygons=11,
        unique_wikidata_entities=9,
        wikipedia_documents=7,
        wikipedia_sections=14,
        wikivoyage_documents=3,
        wikivoyage_sections=6,
        wikidata_facts=20,
        polygon_document_links=12,
        wikipedia_tag_only_polygons=2,
        document_words=101,
        languages=4,
        new_polygons_vs_v1=2,
        new_wikipedia_documents_vs_v1=3,
        text_coverage_funnel=(("All polygons", 11),),
        top_wikipedia_languages=(("en", 7),),
        polygon_link_storage_bytes=1_000_000_000,
        total_parquet_storage_bytes=2_000_000_000,
    )

    snapshot = snapshot_from_v2_stats(stats)

    assert snapshot.metrics() == {
        "scale/polygons": 11,
        "scale/unique_wikidata_entities": 9,
        "corpus/documents": 10,
        "corpus/document_words": 101,
        "coverage/languages": 4,
        "coverage/geographic_regions": 2,
    }
    assert snapshot.wikipedia_documents == 7
    assert snapshot.wikivoyage_documents == 3
    assert snapshot.wikipedia_polygon_document_links == 12
    assert V2_TRACKIO_RUN_NAME == "final-dataset-snapshot-v2"


def test_v2_card_stats_count_documents_outside_the_top_languages() -> None:
    stats = V2CardStats(
        regions=1,
        polygons=1,
        unique_wikidata_entities=1,
        wikipedia_documents=10,
        wikipedia_sections=0,
        wikivoyage_documents=0,
        wikivoyage_sections=0,
        wikidata_facts=0,
        polygon_document_links=0,
        wikipedia_tag_only_polygons=0,
        document_words=0,
        languages=3,
        new_polygons_vs_v1=None,
        new_wikipedia_documents_vs_v1=None,
        text_coverage_funnel=(),
        top_wikipedia_languages=(("en", 7), ("fr", 1)),
        polygon_link_storage_bytes=0,
        total_parquet_storage_bytes=0,
    )

    assert stats.other_wikipedia_languages == 2
