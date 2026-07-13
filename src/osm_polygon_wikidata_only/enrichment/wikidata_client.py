"""Wikidata enrichment compatibility facade.

Re-exports the public Wikidata client surface from the focused
:mod:`enrichment.wikidata` subpackage so callers that import
``osm_polygon_wikidata_only.enrichment.wikidata_client`` keep working
unchanged. The implementation lives in
:mod:`enrichment.wikidata.transport` and
:mod:`enrichment.wikidata.cache`; this module owns no code.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.enrichment.wikidata.cache import (
    CachedWikidataClient,
)
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    Sitelinks,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    is_valid_qid,
    language_from_site,
    parse_wikidata_entity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.transport import (
    HttpWikidataClient,
    InMemoryWikidataClient,
    WikidataError,
)

__all__ = [
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
]
