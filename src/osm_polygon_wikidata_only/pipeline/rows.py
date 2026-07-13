"""Backwards-compatible re-exports of :mod:`pipeline.row_construction`.

The historical module lived at ``pipeline.rows``. The decomposition
moved the actual implementations to :mod:`pipeline.row_construction`;
this module re-exports the three public helpers by identity so older
imports (tests and code) keep working without change.

The text-cleaning helpers (e.g. :func:`count_words`) that existing
tests monkeypatch in this module are also re-exported, so the
focused implementation can swap underneath without breaking the
historical patches.
"""

from osm_polygon_wikidata_only.enrichment import text_cleaning
from osm_polygon_wikidata_only.pipeline.row_construction import (
    article_row,
    build_articles_and_links,
    enrich_polygon,
)

# Re-export the text-cleaning helpers by identity so existing
# tests can monkeypatch them through this module's namespace.
count_words = text_cleaning.count_words
estimate_tokens = text_cleaning.estimate_tokens

__all__ = [
    "article_row",
    "build_articles_and_links",
    "count_words",
    "enrich_polygon",
    "estimate_tokens",
]
