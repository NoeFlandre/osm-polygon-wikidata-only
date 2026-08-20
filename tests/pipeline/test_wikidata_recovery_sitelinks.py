"""Contracts for selecting eligible recovery sitelinks."""

from __future__ import annotations

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.repair import _eligible_sitelinks


def test_eligible_sitelinks_filters_languages_and_applies_nonnegative_cap() -> None:
    entity = WikidataEntity(
        qid="Q1",
        sitelinks={
            "dewiki": "German",
            "enwiki": "English",
            "frwiki": "French",
        },
    )

    assert _eligible_sitelinks(entity, Settings(languages=("fr",))) == [("frwiki", "French")]
    assert _eligible_sitelinks(entity, Settings(max_articles_per_qid=2)) == [
        ("dewiki", "German"),
        ("enwiki", "English"),
    ]
    assert _eligible_sitelinks(entity, Settings(max_articles_per_qid=-1)) == []
