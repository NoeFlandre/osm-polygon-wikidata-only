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


def test_contract_version_mismatch_is_a_cache_miss(tmp_path: Path) -> None:
    JsonFileCache(tmp_path, contract_version="v1").set("foo", {"a": 1})
    assert JsonFileCache(tmp_path, contract_version="v2").get("foo") is None


def test_long_cache_key_uses_a_filesystem_safe_filename(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    key = "entities/" + "-".join(f"Q{number}" for number in range(100))

    cache.set(key, {"ok": True})

    entry = cache.get(key)
    assert entry is not None
    assert entry.parsed_result == {"ok": True}
    assert max(len(path.name.encode()) for path in tmp_path.iterdir()) <= 255
