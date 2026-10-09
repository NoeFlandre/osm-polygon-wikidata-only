from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult, WikipediaArticle
from osm_polygon_wikidata_only.v2 import direct_enrichment, maps, reuse_reconcile, runner
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    DirectLookupOptions,
    DirectWikipediaStatus,
)
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef


def test_remove_speculative_link_preserves_other_sources_and_drops_empty_links() -> None:
    key = ("polygon", "document", "wikipedia")
    links = {
        key: {
            "document_id": "document",
            "project": "wikipedia",
            "wikidata": None,
            "link_sources": json.dumps(["manual", "osm_wikipedia_tag"]),
        }
    }
    direct_document_ids: set[str] = set()

    reuse_reconcile._remove_speculative_link(key, links[key], links, direct_document_ids)

    assert json.loads(cast(str, links[key]["link_sources"])) == ["manual"]
    assert direct_document_ids == {"document"}

    links[key]["link_sources"] = json.dumps(["osm_wikipedia_tag"])
    reuse_reconcile._remove_speculative_link(key, links[key], links, direct_document_ids)
    assert key not in links


def test_remove_speculative_link_leaves_non_speculative_links_unchanged() -> None:
    key = ("polygon", "document", "wikipedia")
    row = {"link_sources": json.dumps(["manual"])}
    links = {key: row}

    reuse_reconcile._remove_speculative_link(key, row, links, set())

    assert links == {key: row}


def test_find_reconciliation_candidates_uses_cached_or_recovered_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = WikipediaTagRef("en", "Title", "wikipedia", "en:Title")
    cached = {"document_id": "cached"}
    key = reuse_reconcile._title_key(ref.language, ref.title)

    assert reuse_reconcile._find_reconciliation_candidates(
        "polygon",
        {},
        ref,
        matches={key: [cached]},
        current_by_title={},
        index=object(),
        wikipedia_client=None,
        cache=None,
        fetch_full_text=True,
    ) == [cached]
    assert reuse_reconcile._find_reconciliation_candidates(
        "polygon",
        {},
        ref,
        matches={},
        current_by_title={("en", "title"): [cached]},
        index=object(),
        wikipedia_client=None,
        cache=None,
        fetch_full_text=True,
    ) == [cached]
    assert (
        reuse_reconcile._find_reconciliation_candidates(
            "polygon",
            {},
            ref,
            matches={},
            current_by_title={},
            index=object(),
            wikipedia_client=None,
            cache=None,
            fetch_full_text=True,
        )
        == ()
    )

    recovered = {"document_id": "recovered"}
    calls: list[dict[str, object]] = []

    def enrich(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(documents=[recovered])

    monkeypatch.setattr(reuse_reconcile, "enrich_wikipedia_refs", enrich)
    assert reuse_reconcile._find_reconciliation_candidates(
        "polygon",
        {"region": "region"},
        ref,
        matches={},
        current_by_title={},
        index="index",
        wikipedia_client="client",
        cache="cache",
        fetch_full_text=False,
    ) == [recovered]
    assert calls[0]["options"] == DirectLookupOptions(fetch_full_text=False, wait_for_index=True)


def test_has_wikipedia_refs_ignores_malformed_and_empty_values() -> None:
    assert not runner._has_wikipedia_refs(
        SimpleNamespace(polygons=[{"wikipedia_tag_refs": "not-json"}, {"wikipedia_tag_refs": "{}"}])
    )
    assert runner._has_wikipedia_refs(
        SimpleNamespace(polygons=[{"wikipedia_tag_refs": '[{"language":"en"}]'}])
    )


def test_resolve_land_context_handles_explicit_sibling_cache_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = tmp_path / "processed_v2"
    explicit = tmp_path / "land.geojson"
    cache = tmp_path / "cache"
    sibling = cache / maps.WORLD_LAND_FILENAME
    assert maps._resolve_land_context(processed, explicit, None) == (explicit, None)

    monkeypatch.setattr(maps, "_existing_sibling_land", lambda _processed: sibling)
    assert maps._resolve_land_context(processed, None, None) == (sibling, cache)
    monkeypatch.setattr(maps, "_existing_sibling_land", lambda _processed: None)
    assert maps._resolve_land_context(processed, None, None) == (None, None)

    expected = cache / "world.geojson"
    monkeypatch.setattr(maps, "ensure_world_land", lambda _cache: expected)
    assert maps._resolve_land_context(processed, None, cache) == (expected, cache)

    def unavailable(_cache: Path) -> Path:
        raise OSError("offline")

    monkeypatch.setattr(maps, "ensure_world_land", unavailable)
    assert maps._resolve_land_context(processed, None, cache) == (None, cache)


def test_apply_pending_outcome_records_error_and_non_ok_article() -> None:
    ref = WikipediaTagRef("en", "Title", "wikipedia", "en:Title")
    deferred = RuntimeError("deferred")
    statuses: dict[int, DirectWikipediaStatus] = {}
    deferred_errors: dict[int, Exception] = {}

    direct_enrichment._apply_pending_outcome(
        "polygon",
        0,
        (DirectWikipediaStatus(ref, "deferred_error"), None, deferred),
        ref,
        {},
        direct_enrichment._DirectRows(statuses=statuses),
        deferred_errors,
    )
    assert statuses[0].status == "deferred_error"
    assert deferred_errors == {0: deferred}

    article = WikipediaArticle(
        language="en",
        site="enwiki",
        title="Title",
        page_id=1,
        revision_id=2,
        revision_timestamp="2026-01-01T00:00:00Z",
        url="https://en.wikipedia.org/wiki/Title",
        lead_text="lead",
        extract="extract",
        full_text="text",
        full_text_format="plain_text",
        thumbnail_url="",
        thumbnail_width=None,
        thumbnail_height=None,
        categories=[],
        license="CC BY-SA",
        attribution="",
        source_api="mediawiki_action_api",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    documents: dict[str, dict[str, object]] = {}
    links: dict[str, dict[str, object]] = {}
    direct_enrichment._apply_pending_outcome(
        "polygon",
        1,
        (
            DirectWikipediaStatus(ref, "partial"),
            FetchResult("partial", article, "incomplete"),
            None,
        ),
        ref,
        {"source_pbf": "region.osm.pbf"},
        direct_enrichment._DirectRows(documents=documents, links=links, statuses=statuses),
        deferred_errors,
    )
    [(document_id, document)] = list(documents.items())
    assert document["fetch_status"] == "partial"
    assert document["fetch_error"] == "incomplete"
    assert document_id in links
