"""Focused Wikipedia enrichment implementation.

The stable compatibility facade lives at
:mod:`osm_polygon_wikidata_only.enrichment.wikipedia_client`. This
package owns the implementation, split into:

* :mod:`enrichment.wikipedia.models` -- typed contracts and value objects.
* :mod:`enrichment.wikipedia.parsing` -- pure parsing helpers.
* :mod:`enrichment.wikipedia.transport` -- HTTP and in-memory clients.
* :mod:`enrichment.wikipedia.cache` -- cached client + serialization.
"""

from .models import BatchWikipediaClient, FetchResult, WikipediaArticle, WikipediaClient

__all__ = ["BatchWikipediaClient", "FetchResult", "WikipediaArticle", "WikipediaClient"]
