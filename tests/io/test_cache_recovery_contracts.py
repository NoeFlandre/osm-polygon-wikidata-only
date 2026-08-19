"""Corruption, atomicity, and cleanup contracts for the JSON cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.io.cache import JsonFileCache


def _cache_path(cache: JsonFileCache, key: str) -> Path:
    return cache._path_for(key)


@pytest.mark.parametrize(
    "raw",
    [
        {"meta": []},
        {"meta": {"expires_at": "not-a-number"}},
        {"meta": {"response_metadata": []}},
    ],
)
def test_invalid_cache_metadata_is_removed_and_treated_as_miss(
    tmp_path: Path, raw: dict[str, object]
) -> None:
    cache = JsonFileCache(tmp_path)
    path = _cache_path(cache, "bad")
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert cache.get("bad") is None
    assert not path.exists()


def test_contract_mismatch_preserves_entry_for_an_older_reader(tmp_path: Path) -> None:
    current = JsonFileCache(tmp_path, contract_version="v2")
    current.set("entry", {"value": 1})

    old_reader = JsonFileCache(tmp_path, contract_version="v1")

    assert old_reader.get("entry") is None
    assert _cache_path(current, "entry").exists()


def test_cache_delete_is_idempotent(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)

    cache.delete("missing")
    cache.set("entry", {"value": 1})
    cache.delete("entry")
    cache.delete("entry")

    assert cache.get("entry") is None


def test_set_copies_response_metadata_before_callers_mutate_it(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    metadata = {"attempt": 1}

    cache.set("entry", {"value": 1}, response_metadata=metadata)
    metadata["attempt"] = 2

    entry = cache.get("entry")
    assert entry is not None
    assert entry.response_metadata == {"attempt": 1}


def test_atomic_cache_write_failure_preserves_previous_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("entry", {"value": "old"})
    path = _cache_path(cache, "entry")
    before = path.read_bytes()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated cache write failure")

    monkeypatch.setattr("osm_polygon_wikidata_only.io.cache.atomic_write_text", fail)
    with pytest.raises(OSError, match="simulated cache write failure"):
        cache.set("entry", {"value": "new"})

    assert path.read_bytes() == before
    assert cache.get("entry").parsed_result == {"value": "old"}  # type: ignore[union-attr]


def test_clear_removes_only_cache_json_files(tmp_path: Path) -> None:
    cache = JsonFileCache(tmp_path)
    cache.set("entry", {"value": 1})
    keep = tmp_path / "operator-note.txt"
    keep.write_text("keep", encoding="utf-8")

    cache.clear()

    assert not _cache_path(cache, "entry").exists()
    assert keep.exists()
