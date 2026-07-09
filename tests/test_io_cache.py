"""Tests for io.cache."""

from __future__ import annotations

import time
from pathlib import Path

from osm_polygon_wikidata_only.io.cache import JsonFileCache


def test_set_then_get_round_trips(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("foo", {"a": 1}, request_url="http://x", ttl_s=60)
    entry = cache.get("foo")
    assert entry is not None
    assert entry.parsed_result == {"a": 1}
    assert entry.status == "ok"
    assert entry.request_url == "http://x"


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    assert cache.get("nope") is None


def test_stale_entry_is_miss(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path, default_ttl_s=10)
    cache.set("foo", {"a": 1}, ttl_s=1)
    time.sleep(1.1)
    assert cache.get("foo") is None


def test_keys_with_slashes_are_sanitized(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("wikipedia/en/123_456", {"x": 1})
    # The file ends up flat in the cache dir.
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "__" in files[0].name


def test_set_then_clear_removes_all(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(list(tmp_path.glob("*.json"))) == 2
    cache.clear()
    assert len(list(tmp_path.glob("*.json"))) == 0


def test_status_field_persists(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("foo", None, status="error", ttl_s=60)
    entry = cache.get("foo")
    assert entry is not None
    assert entry.status == "error"
    assert entry.parsed_result is None
