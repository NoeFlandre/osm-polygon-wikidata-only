"""Tests for the streaming manifest-statistics helpers."""

from __future__ import annotations

from osm_polygon_wikidata_only.pipeline.stats import StreamingStats, accumulate_stats


def test_accumulate_stats_consumes_links_without_materializing_them() -> None:
    class LinkStream:
        def __init__(self, count: int) -> None:
            self._remaining = count

        def __iter__(self) -> LinkStream:
            return self

        def __next__(self) -> object:
            if self._remaining == 0:
                raise StopIteration
            self._remaining -= 1
            return object()

        def __len__(self) -> int:
            raise AssertionError("link iterables must not be materialized")

    stats = accumulate_stats((), (), LinkStream(3))

    assert stats.polygon_count == 0
    assert stats.article_count == 0


def test_accumulate_stats_matches_streaming_stats() -> None:
    from types import SimpleNamespace

    polygon = SimpleNamespace(
        polygon_id="region:way:1",
        wikidata="Q1",
        has_wikipedia=True,
        text_available=True,
        area_bucket="1k_m2-10k_m2",
        tag_keys='["name", "wikidata"]',
    )
    article = SimpleNamespace(language="en", article_length_chars=12)

    expected = StreamingStats()
    expected.add_polygon(polygon)
    expected.add_article(article)
    expected.add_link(object())

    actual = accumulate_stats((polygon,), (article,), (object(),))

    assert actual.to_dict() == expected.finalize().to_dict()
