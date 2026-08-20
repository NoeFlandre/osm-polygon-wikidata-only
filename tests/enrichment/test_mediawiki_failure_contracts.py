"""Failure and cache-boundary contracts for the augmentation transport."""

from __future__ import annotations

from typing import Any

import pytest

import osm_polygon_wikidata_only.augmentation.mediawiki as mediawiki
from osm_polygon_wikidata_only.config.settings import Settings


class _CacheEntry:
    def __init__(self, parsed_result: object, status: str = "ok") -> None:
        self.status = status
        self.parsed_result = parsed_result


class _Cache:
    def __init__(self, entry: _CacheEntry | None = None) -> None:
        self.entry = entry
        self.deleted: list[str] = []
        self.stored: list[tuple[str, object]] = []

    def get(self, key: str) -> _CacheEntry | None:
        return self.entry

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.entry = None

    def set(self, key: str, payload: object, **kwargs: object) -> None:
        del kwargs
        self.stored.append((key, payload))


def _client(cache: _Cache, *, attempts: int = 1) -> mediawiki.AugmentationWikimediaClient:
    client = mediawiki.AugmentationWikimediaClient.__new__(mediawiki.AugmentationWikimediaClient)
    client._settings = Settings(request_max_retries=attempts, request_base_delay_s=0)
    client._cache = cache
    client._session = object()
    client._scheduler = object()
    return client


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        ("maxlag", mediawiki._TransientMediaWikiApiError),
        ("ratelimited", mediawiki._TransientMediaWikiApiError),
        ("readonly", mediawiki._TransientMediaWikiApiError),
        ("readonlytext", mediawiki._TransientMediaWikiApiError),
        ("internal_api_error_DB", mediawiki._TransientMediaWikiApiError),
        ("nosuchrevid", mediawiki.MediaWikiApiError),
        ("permissiondenied", mediawiki.MediaWikiApiError),
    ],
)
def test_api_error_codes_preserve_retry_classification(
    code: str, expected_type: type[mediawiki.MediaWikiApiError]
) -> None:
    with pytest.raises(expected_type) as caught:
        mediawiki._raise_for_api_error({"error": {"code": code, "info": "details"}})

    assert caught.value.code == code
    assert caught.value.info == "details"


def test_api_payload_without_error_is_accepted() -> None:
    mediawiki._raise_for_api_error({"query": {"pages": {}}})


def test_entity_list_normalization_keeps_only_identified_mappings() -> None:
    assert mediawiki._normalize_entity_list("not-a-list") == {}
    assert mediawiki._normalize_entity_list(
        [{"id": "Q1", "labels": {}}, {"labels": {}}, "invalid"]
    ) == {"Q1": {"id": "Q1", "labels": {}}}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"parse": {"text": "plain"}}, "plain"),
        ({"parse": {"text": {"*": "expanded"}}}, "expanded"),
        ({"parse": {"text": 42}}, "42"),
        ({"parse": []}, ""),
    ],
)
def test_parsed_html_text_normalizes_mediawiki_text_shapes(
    payload: dict[str, object], expected: str
) -> None:
    assert mediawiki._parsed_html_text(payload) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"query": {"pages": [{"pageid": 1}]}}, {"pageid": 1}),
        ({"query": {"pages": [{"missing": True}]}}, None),
        ({"query": {"pages": []}}, None),
        ({}, None),
    ],
)
def test_first_voyage_page_handles_missing_and_empty_queries(
    payload: dict[str, object], expected: dict[str, object] | None
) -> None:
    assert mediawiki._first_voyage_page(payload) == expected


def test_permanent_api_error_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _Cache()
    client = _client(cache)
    monkeypatch.setattr(
        mediawiki,
        "read_wikimedia_json",
        lambda *_args, **_kwargs: {"error": {"code": "nosuchrevid", "info": "deleted"}},
    )

    with pytest.raises(mediawiki.MediaWikiApiError, match="nosuchrevid"):
        client.get_json("https://en.wikipedia.org/w/api.php", key="missing")

    assert cache.stored == []


def test_transient_api_error_retries_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _Cache()
    client = _client(cache, attempts=2)
    calls = 0

    def always_lagging(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"error": {"code": "maxlag", "info": "try later"}}

    monkeypatch.setattr(mediawiki, "read_wikimedia_json", always_lagging)

    with pytest.raises(mediawiki._TransientMediaWikiApiError, match="maxlag"):
        client.get_json("https://en.wikipedia.org/w/api.php", key="lagging")

    assert calls == 2
    assert cache.stored == []


def test_cached_api_error_is_deleted_before_refetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _Cache(_CacheEntry({"error": {"code": "nosuchrevid", "info": "old"}}))
    client = _client(cache)
    fetched = {"query": {"pages": {"1": {"pageid": 1}}}}
    calls: list[str] = []

    def refetch(*_args: object, **_kwargs: object) -> dict[str, Any]:
        calls.append("read")
        return fetched

    monkeypatch.setattr(mediawiki, "read_wikimedia_json", refetch)

    assert client.get_json("https://en.wikipedia.org/w/api.php", key="cached-error") == fetched
    assert cache.deleted == ["cached-error"]
    assert calls == ["read"]
    assert cache.stored == [("cached-error", fetched)]


def test_cached_success_short_circuits_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"query": {"pages": {"1": {"pageid": 1}}}}
    cache = _Cache(_CacheEntry(payload))
    client = _client(cache)

    def transport(*_args: object, **_kwargs: object) -> object:
        pytest.fail("transport called")

    monkeypatch.setattr(mediawiki, "read_wikimedia_json", transport)

    assert client.get_json("not-a-url", key="cached") == payload
