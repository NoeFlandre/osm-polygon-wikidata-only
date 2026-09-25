import duckdb
from osm_polygon_wikidata_only_preprocessing.deduplication.identity import (
    CANONICAL_ARTICLE_ID_SQL,
)


def test_canonical_article_id_sql_uses_site_page_and_revision():
    source = "SELECT 'enwiki' AS site, 30151915 AS page_id, 1321767249 AS revision_id"
    query = f"SELECT {CANONICAL_ARTICLE_ID_SQL} FROM ({source})"  # noqa: S608 - constant SQL
    row = duckdb.sql(query).fetchone()
    assert row == ("enwiki:30151915:1321767249",)
