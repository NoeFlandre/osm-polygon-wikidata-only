"""Constructible fakes shared by the Wikidata and Wikipedia client tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from osm_polygon_wikidata_only.io.cache import JsonCache

_T = TypeVar("_T")


class _StubSession:
    def __init__(self, responses: list[Any]) -> None:
        self.reads: list[tuple[Any, float, float]] = []
        self._responses = list(responses)

    def read(
        self,
        request: Any,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]:
        self.reads.append((request, min_interval_anonymous_s, min_interval_authenticated_s))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _RecordingScheduler:
    def __init__(self) -> None:
        self.throttle_calls: list[tuple[str, float]] = []
        self.max_in_flight = 3

    def pace_host(self, host: str, *, min_interval_s: float = 0.0) -> None:
        del host, min_interval_s

    def report_success(self) -> None:
        return None

    def report_host_throttled(self, host: str, delay_s: float) -> None:
        self.throttle_calls.append((host, delay_s))

    def run(self, operation: Callable[[], _T]) -> _T:
        return operation()


def _make_settings(**overrides: Any) -> Any:
    base = {
        "user_agent": "ua",
        "request_max_retries": 1,
        "request_base_delay_s": 0.0,
        "request_timeout_s": 60.0,
        "wikidata_min_interval_s": 1.0,
        "wikipedia_min_interval_s": 1.0,
        "wikimedia_authenticated_min_interval_s": 0.5,
        "rate_limit_retry_after_default_s": 60.0,
    }
    base.update(overrides)
    return type("Settings", (), base)()


@dataclass(frozen=True)
class _CacheEntry:
    status: str
    parsed_result: Any
    request_url: str | None


def _cache_entry(status: str, parsed_result: Any, request_url: str | None) -> _CacheEntry:
    return _CacheEntry(status, parsed_result, request_url)


class _MemoryCache(JsonCache):
    """Small protocol-conforming cache for client characterization tests."""

    def __init__(self, entries: dict[str, _CacheEntry] | None = None) -> None:
        self.entries = entries or {}
        self.writes: list[tuple[str, Any, dict[str, Any]]] = []

    def get(self, key: str) -> _CacheEntry | None:
        return self.entries.get(key)

    def set(
        self,
        key: str,
        payload: Any,
        *,
        request_url: str = "",
        response_metadata: dict[str, Any] | None = None,
        status: str = "ok",
        ttl_s: int | None = None,
    ) -> object:
        self.writes.append(
            (
                key,
                payload,
                {
                    "request_url": request_url,
                    "response_metadata": response_metadata,
                    "status": status,
                    "ttl_s": ttl_s,
                },
            )
        )
        return self
