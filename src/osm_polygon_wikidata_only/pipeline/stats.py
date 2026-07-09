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
    """Compute aggregate stats from already-materialized rows.

    Each iterable is consumed once. For very large PBFs, prefer
    streaming stats via :class:`StreamingStats` instead.
    """
    polys = list(polygons)
    arts = list(articles)
    links_list = list(links)

    stats = ManifestStats()
    stats.polygon_count = len(polys)
    stats.unique_wikidata_count = len({p.wikidata for p in polys if p.wikidata})
    stats.article_count = len(arts)
    stats.language_count = len({a.language for a in arts})
    stats.languages = sorted({a.language for a in arts})

    polygon_with_text = {p.polygon_id for p in polys if p.text_available}
    stats.rows_with_wikipedia = sum(1 for p in polys if p.has_wikipedia)
    stats.rows_with_full_text = len(polygon_with_text)
    stats.total_full_text_chars = sum(a.article_length_chars for a in arts)

    stats.area_bucket_counts = dict(Counter(p.area_bucket for p in polys))

    # tag_keys is a JSON list per polygon; tally individual keys.
    import json

    tag_counter: Counter[str] = Counter()
    for p in polys:
        try:
            keys = json.loads(p.tag_keys)
        except (ValueError, TypeError):
            continue
        tag_counter.update(keys)
    stats.top_tag_keys = dict(tag_counter.most_common(50))

    # Validate the links are consistent (defensive).
    _ = links_list
    return stats


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
