"""Focused Wikipedia enrichment implementation.

Use :mod:`osm_polygon_wikidata_only.enrichment.wikipedia_client` for the
stable compatibility surface.
"""

from .models import BatchWikipediaClient, FetchResult, WikipediaArticle, WikipediaClient

__all__ = ["BatchWikipediaClient", "FetchResult", "WikipediaArticle", "WikipediaClient"]
