"""Per-PBF statistics used by the manifest and CLI logs.

Kept separate from the processor so tests can exercise it in isolation
without spinning up the rest of the pipeline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from osm_polygon_wikidata_only.domain.models import (
    Article,
    ManifestStats,
    Polygon,
    PolygonArticleLink,
)


def accumulate_stats(
    polygons: Iterable[Polygon],
    articles: Iterable[Article],
    links: Iterable[PolygonArticleLink],
) -> ManifestStats:
    """Compute aggregate stats from one-pass iterables.

    This compatibility helper delegates to :class:`StreamingStats`, so it
    retains the historical result while avoiding a second in-memory copy of
    large polygon, article, or link collections.
    """
    stats = StreamingStats()
    for polygon in polygons:
        stats.add_polygon(polygon)
    for article in articles:
        stats.add_article(article)
    for link in links:
        stats.add_link(link)
    return stats.finalize()


class StreamingStats:
    """Single-pass accumulator that avoids materializing rows.

    Use this when polygons/articles/links are produced by a large
    generator.
    """

    def __init__(self) -> None:
        self._polygon_count = 0
        self._unique_qids: set[str] = set()
        self._article_count = 0
        self._languages: set[str] = set()
        self._rows_with_wikipedia = 0
        self._text_polygons: set[str] = set()
        self._total_chars = 0
        self._area_buckets: Counter[str] = Counter()
        self._tag_keys: Counter[str] = Counter()
        self._links_count = 0

    def add_polygon(self, p: Polygon) -> None:
        self._polygon_count += 1
        if p.wikidata:
            self._unique_qids.add(p.wikidata)
        if p.has_wikipedia:
            self._rows_with_wikipedia += 1
        if p.text_available:
            self._text_polygons.add(p.polygon_id)
        self._area_buckets[p.area_bucket] += 1
        import json

        try:
            keys = json.loads(p.tag_keys)
        except (ValueError, TypeError):
            keys = []
        self._tag_keys.update(keys)

    def add_article(self, a: Article) -> None:
        self._article_count += 1
        self._languages.add(a.language)
        self._total_chars += a.article_length_chars

    def add_link(self, link: PolygonArticleLink) -> None:
        self._links_count += 1

    def finalize(self) -> ManifestStats:
        return ManifestStats(
            polygon_count=self._polygon_count,
            unique_wikidata_count=len(self._unique_qids),
            article_count=self._article_count,
            language_count=len(self._languages),
            languages=sorted(self._languages),
            rows_with_wikipedia=self._rows_with_wikipedia,
            rows_with_full_text=len(self._text_polygons),
            total_full_text_chars=self._total_chars,
            area_bucket_counts=dict(self._area_buckets),
            top_tag_keys=dict(self._tag_keys.most_common(50)),
        )


__all__ = ["StreamingStats", "accumulate_stats"]
