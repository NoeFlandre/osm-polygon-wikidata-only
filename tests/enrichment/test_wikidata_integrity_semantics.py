from __future__ import annotations

import urllib.error
from email.message import Message
from typing import Any

import pytest

from osm_polygon_wikidata_only.enrichment.wikidata.cache import CachedWikidataClient
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity
from osm_polygon_wikidata_only.enrichment.wikidata.transport import (
    HttpWikidataClient,
    InMemoryWikidataClient,
    WikidataError,
)


class _StubSession:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.read_count = 0

    def read(
        self,
        request: Any,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]:
        del request, min_interval_anonymous_s, min_interval_authenticated_s
        self.read_count += 1
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _Scheduler:
    def report_host_throttled(self, host: str, delay: float) -> None:
        del host, delay


def _settings(**overrides: Any) -> Any:
    values = {
        "user_agent": "test",
        "request_max_retries": 1,
        "request_base_delay_s": 0.0,
        "request_timeout_s": 1.0,
        "wikidata_min_interval_s": 0.0,
        "wikimedia_authenticated_min_interval_s": 0.0,
        "rate_limit_retry_after_default_s": 0.0,
    }
    values.update(overrides)
    return type("Settings", (), values)()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://www.wikidata.org/w/api.php", code, "error", Message(), None
    )


def _client(responses: list[Any], **settings: Any) -> tuple[HttpWikidataClient, _StubSession]:
    session = _StubSession(responses)
    client = HttpWikidataClient(
        _settings(**settings),
        scheduler=_Scheduler(),  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
    )
    return client, session


class _Cache:
    def __init__(self, entries: dict[str, Any] | None = None) -> None:
        self.entries = entries or {}
        self.writes: list[tuple[str, Any, dict[str, Any]]] = []

    def get(self, key: str) -> Any:
        return self.entries.get(key)

    def set(self, key: str, payload: Any, **kwargs: Any) -> None:
        self.writes.append((key, payload, kwargs))


def _entry(
    status: str,
    payload: Any,
    *,
    reason: str | None = None,
) -> Any:
    return type(
        "CacheEntry",
        (),
        {
            "status": status,
            "parsed_result": payload,
            "request_url": "",
            "response_metadata": {"reason": reason} if reason else {},
        },
    )()


def _entity(qid: str) -> WikidataEntity:
    return WikidataEntity(qid=qid, sitelinks={"enwiki": f"Title {qid}"})


def test_failed_fifty_qid_transport_batch_propagates_and_caches_nothing() -> None:
    client, _ = _client([_http_error(503)])
    cache = _Cache()
    cached = CachedWikidataClient(client, cache)  # type: ignore[arg-type]

    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 503"):
        cached.get_entities([f"Q{index}" for index in range(1, 51)])

    assert cache.writes == []


def test_exhausted_finite_transient_retry_propagates() -> None:
    error = urllib.error.URLError(TimeoutError("offline"))
    client, session = _client([error, error], request_max_retries=2)

    with pytest.raises(urllib.error.URLError):
        client.get_entities(["Q1"])

    assert session.read_count == 2


def test_permanent_http_failure_propagates_without_retry() -> None:
    client, session = _client([_http_error(400)], request_max_retries=None)

    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 400"):
        client.get_entities(["Q1"])

    assert session.read_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        b'{"entities":[]}',
        b'{"entities":{"Q1":{"id":"Q1","sitelinks":{}}}}',
    ],
)
def test_malformed_or_incomplete_batch_response_fails_closed(payload: bytes) -> None:
    client, _ = _client([(payload, "identity")])

    with pytest.raises(WikidataError):
        client.get_entities(["Q1", "Q2"])


def test_explicit_missing_is_the_only_negative_cached_outcome() -> None:
    client, _ = _client([(b'{"entities":{"Q1":{"id":"Q1","missing":""}}}', "identity")])
    cache = _Cache()

    assert CachedWikidataClient(client, cache).get_entity("Q1") is None  # type: ignore[arg-type]

    assert len(cache.writes) == 1
    key, payload, metadata = cache.writes[0]
    assert key == "wikidata/Q1.json"
    assert payload is None
    assert metadata["status"] == "not_found"
    assert metadata["response_metadata"] == {"reason": "wikidata_entity_missing"}


def test_mixed_success_and_missing_batch_preserves_order() -> None:
    response = (
        b'{"entities":{'
        b'"Q1":{"id":"Q1","sitelinks":{"enwiki":{"title":"One"}}},'
        b'"Q2":{"id":"Q2","missing":""}'
        b"}}"
    )
    client, _ = _client([(response, "identity")])

    results = client.get_entities(["Q2", "Q1", "Q2", "invalid"])

    assert [result.qid if result else None for result in results] == [None, "Q1", None, None]


def test_legacy_ambiguous_negative_is_a_miss_and_overwritten_after_success() -> None:
    cache = _Cache(
        {
            "wikidata/Q1.json": _entry(
                "error",
                None,
                reason="wikidata_not_found",
            )
        }
    )
    inner = InMemoryWikidataClient({"Q1": _entity("Q1")})

    result = CachedWikidataClient(inner, cache).get_entity("Q1")  # type: ignore[arg-type]

    assert result == _entity("Q1")
    assert len(cache.writes) == 1
    assert cache.writes[0][2]["status"] == "ok"


def test_authoritative_negative_cache_entry_is_reused() -> None:
    cache = _Cache({"wikidata/Q1.json": _entry("not_found", None)})

    class _FailingInner(InMemoryWikidataClient):
        def get_entity(self, qid: str) -> WikidataEntity | None:
            raise AssertionError(f"unexpected fetch: {qid}")

    result = CachedWikidataClient(_FailingInner({}), cache).get_entity("Q1")  # type: ignore[arg-type]

    assert result is None
    assert cache.writes == []


def test_existing_positive_cache_entry_is_reused_unchanged() -> None:
    payload = {
        "qid": "Q1",
        "sitelinks": {"enwiki": "One"},
        "labels": {},
        "descriptions": {},
        "aliases": {},
    }
    cache = _Cache({"wikidata/Q1.json": _entry("ok", payload)})

    result = CachedWikidataClient(InMemoryWikidataClient({}), cache).get_entity("Q1")  # type: ignore[arg-type]

    assert result == WikidataEntity(qid="Q1", sitelinks={"enwiki": "One"})
    assert cache.writes == []


def test_unbounded_transient_retry_still_reaches_success() -> None:
    errors = [urllib.error.URLError(TimeoutError("offline")) for _ in range(3)]
    response = b'{"entities":{"Q1":{"id":"Q1","sitelinks":{}}}}'
    client, session = _client([*errors, (response, "identity")], request_max_retries=None)

    result = client.get_entity("Q1")

    assert result == WikidataEntity(qid="Q1")
    assert session.read_count == 4
