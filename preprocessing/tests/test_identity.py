from osm_polygon_wikidata_only_preprocessing.deduplication.identity import (
    canonical_article_id,
)


def test_canonical_article_id_uses_site_page_and_revision():
    assert canonical_article_id("enwiki", 30151915, 1321767249) == "enwiki:30151915:1321767249"
