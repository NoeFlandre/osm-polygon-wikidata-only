"""Tests for the small utilities."""

from __future__ import annotations

import json

import pytest

from osm_polygon_wikidata_only.utils import rate_limit
from osm_polygon_wikidata_only.utils.json import dumps, dumps_compact_list, loads
from osm_polygon_wikidata_only.utils.time import parse_iso_to_z, utc_now_iso


def test_dumps_is_deterministic() -> None:
    a = dumps({"b": 2, "a": 1})
    b = dumps({"a": 1, "b": 2})
    assert a == b
    assert a == json.dumps(
        {"a": 1, "b": 2}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def test_dumps_preserves_unicode() -> None:
    assert "é" in dumps({"name": "Café"})


def test_dumps_uses_compact_separators() -> None:
    assert dumps({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_loads_round_trips() -> None:
    assert loads('{"a":1}') == {"a": 1}


def test_dumps_compact_list_sorts_and_dedups() -> None:
    out = dumps_compact_list(["b", "a", "a", ""])
    assert loads(out) == ["a", "b"]


def test_utc_now_iso_has_z_suffix() -> None:
    ts = utc_now_iso()
    assert ts.endswith("Z")
    # 20 chars: YYYY-MM-DDTHH:MM:SSZ
    assert len(ts) == 20


def test_parse_iso_to_z_normalizes_z_suffix() -> None:
    assert parse_iso_to_z("2026-01-02T03:04:05Z") == "2026-01-02T03:04:05Z"


def test_parse_iso_to_z_normalizes_offset() -> None:
    assert parse_iso_to_z("2026-01-02T03:04:05+00:00") == "2026-01-02T03:04:05Z"


def test_parse_iso_to_z_returns_input_on_garbage() -> None:
    assert parse_iso_to_z("not a date") == "not a date"


def test_defer_host_moves_next_request_after_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: 10.0)
    rate_limit.defer_host("en.wikipedia.org", 30.0)
    assert rate_limit.next_wait_seconds("en.wikipedia.org") == 30.0
