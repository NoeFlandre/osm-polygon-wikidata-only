"""Tests for the streaming manifest-statistics helpers."""

from __future__ import annotations

from osm_polygon_wikidata_only.pipeline.stats import accumulate_stats


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

    links = LinkStream(3)
    stats = accumulate_stats((), (), links)

    assert stats.polygon_count == 0
    assert stats.article_count == 0
    assert links._remaining == 0


def test_accumulate_stats_preserves_manifest_values() -> None:
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

    actual = accumulate_stats((polygon,), (article,), (object(),))

    assert actual.to_dict() == {
        "polygon_count": 1,
        "unique_wikidata_count": 1,
        "article_count": 1,
        "language_count": 1,
        "languages": ["en"],
        "rows_with_wikipedia": 1,
        "rows_with_full_text": 1,
        "total_full_text_chars": 12,
        "area_bucket_counts": {"1k_m2-10k_m2": 1},
        "top_tag_keys": {"name": 1, "wikidata": 1},
    }
