"""Wikipedia enrichment compatibility facade.

Re-exports the public Wikipedia client surface from the focused
:mod:`enrichment.wikipedia` subpackage so callers that import
``osm_polygon_wikidata_only.enrichment.wikipedia_client`` keep
working unchanged. The implementation lives in
:mod:`enrichment.wikipedia.transport` and
:mod:`enrichment.wikipedia.cache`; this module owns no code.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.enrichment.wikipedia.cache import (
    CachedWikipediaClient,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.models import (
    BatchWikipediaClient,
    FetchResult,
    WikipediaArticle,
    WikipediaClient,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.parsing import (
    parse_wikipedia_response,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import (
    HttpWikipediaClient,
    InMemoryWikipediaClient,
)

__all__ = [
    "BatchWikipediaClient",
    "CachedWikipediaClient",
    "FetchResult",
    "HttpWikipediaClient",
    "InMemoryWikipediaClient",
    "WikipediaArticle",
    "WikipediaClient",
    "parse_wikipedia_response",
]
