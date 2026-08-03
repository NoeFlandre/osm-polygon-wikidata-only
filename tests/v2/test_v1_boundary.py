from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.domain.polygon_document_links import LINK_CONTRACT_VERSION
from osm_polygon_wikidata_only.domain.schema import POLYGON_COLUMNS
from osm_polygon_wikidata_only.hf.repo_layout import REMOTE_LINKS_DIR


def test_v1_contract_remains_unchanged() -> None:
    assert DEFAULT_REPO_ID == "NoeFlandre/osm-polygon-wikidata-only"
    assert LINK_CONTRACT_VERSION == "polygon-document-links-v1"
    assert REMOTE_LINKS_DIR == "polygon_articles"
    assert "wikidata" in POLYGON_COLUMNS
