"""Contracts for Wikivoyage document page selection and conversion."""

from __future__ import annotations

from typing import Any

from osm_polygon_wikidata_only.augmentation.mediawiki import _voyage_page


def test_voyage_page_returns_none_for_missing_or_empty_revisions() -> None:
    assert _voyage_page({}) is None
    assert _voyage_page({"query": {"pages": [{"missing": True}]}}) is None
    assert _voyage_page({"query": {"pages": [{"revisions": []}]}}) is None


def test_voyage_page_returns_first_page_and_revision() -> None:
    page: dict[str, Any] = {
        "pageid": 42,
        "title": "Example",
        "revisions": [{"revid": 7}],
    }

    assert _voyage_page({"query": {"pages": [page]}}) == (page, page["revisions"][0])


def test_wikivoyage_document_builds_from_cached_payload() -> None:
    from osm_polygon_wikidata_only.augmentation.mediawiki import _voyage_document

    page = {
        "pageid": 42,
        "title": "Example",
        "fullurl": "https://en.wikivoyage.org/wiki/Example",
        "extract": "A short travel guide.",
        "revisions": [{"revid": 7, "timestamp": "2026-01-01T00:00:00Z"}],
    }
    document = _voyage_document(
        qid="Q42",
        language="en",
        site="enwiki",
        title="Example",
        page=page,
        revision=page["revisions"][0],
        retrieved="2026-01-02T00:00:00Z",
    )

    assert document.document_id == "Q42:wikivoyage:en:42:7"
    assert document.full_text == "A short travel guide."
    assert document.source_api == "mediawiki_action_api"


def test_wikivoyage_document_uses_expected_endpoint_and_cache_key(monkeypatch) -> None:
    from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient

    client = AugmentationWikimediaClient.__new__(AugmentationWikimediaClient)
    calls: list[tuple[str, str]] = []

    def get_json(url: str, *, key: str) -> dict[str, Any]:
        calls.append((url, key))
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 42,
                        "title": "Example",
                        "fullurl": "https://en.wikivoyage.org/wiki/Example",
                        "extract": "Guide",
                        "revisions": [{"revid": 7, "timestamp": "2026-01-01T00:00:00Z"}],
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "get_json", get_json)
    document = client.wikivoyage_document("Q42", "en", "enwiki", "Example title")

    assert document is not None
    assert calls == [
        (
            calls[0][0],
            "wikivoyage/en/Example%20title.json",
        )
    ]
    assert "https://en.wikivoyage.org/w/api.php?" in calls[0][0]
    assert document.document_id == "Q42:wikivoyage:en:42:7"
