"""Focused Wikidata enrichment implementation.

The stable compatibility facade lives at
:mod:`osm_polygon_wikidata_only.enrichment.wikidata_client`. This
package owns the implementation, split into:

* :mod:`enrichment.wikidata.models` -- typed contracts and value objects.
* :mod:`enrichment.wikidata.parsing` -- pure parsing helpers.
* :mod:`enrichment.wikidata.transport` -- HTTP and in-memory clients.
* :mod:`enrichment.wikidata.cache` -- cached client + serialization.
"""

from .models import BatchWikidataClient, Sitelinks, WikidataClient, WikidataEntity

__all__ = ["BatchWikidataClient", "Sitelinks", "WikidataClient", "WikidataEntity"]
