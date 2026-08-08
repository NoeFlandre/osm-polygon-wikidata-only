"""Tests for io.manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.domain.models import ManifestStats
from osm_polygon_wikidata_only.io.manifest import (
    iter_entries,
    load_manifest,
    make_entry,
    manifest_path,
    save_manifest,
    upsert_entry,
)


def _entry(source_pbf: str = "monaco-latest.osm.pbf") -> dict:
    return make_entry(
        source_pbf=source_pbf,
        region="monaco",
        polygons_path="polygons/monaco-latest.parquet",
        articles_path="articles/monaco-latest.parquet",
        polygon_articles_path="polygon_articles/monaco-latest.parquet",
        stats=ManifestStats(polygon_count=10),
        extraction_version="0.1.0",
        processed_at="2026-01-01T00:00:00Z",
    )


def test_load_manifest_returns_empty_for_missing(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "nope.json") == {}


def test_load_manifest_rejects_corrupt_utf8_with_actionable_error(tmp_path: Path) -> None:
    """Corrupt manifest bytes must identify the processed manifest format."""
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="Malformed processed manifest"):
        load_manifest(path)


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    save_manifest(p, {"a": _entry("a-latest.osm.pbf")})
    out = load_manifest(p)
    assert "a" in out
    assert out["a"]["polygon_count"] == 10


def test_save_manifest_sorts_keys(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    save_manifest(
        p,
        {
            "z": _entry("z-latest.osm.pbf"),
            "a": _entry("a-latest.osm.pbf"),
        },
    )
    text = p.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"z"')


def test_save_manifest_uses_atomic_writer(tmp_path: Path, monkeypatch) -> None:
    """A manifest update must go through the shared atomic writer."""
    p = tmp_path / "manifest.json"
    calls: list[Path] = []

    def record(path: Path, text: str, *, encoding: str = "utf-8") -> None:
        calls.append(path)
        p.write_text(text, encoding=encoding)

    monkeypatch.setattr("osm_polygon_wikidata_only.io.manifest.atomic_write_text", record)
    save_manifest(p, {"a": _entry("a-latest.osm.pbf")})

    assert calls == [p]
    assert load_manifest(p)["a"]["polygon_count"] == 10


def test_save_manifest_preserves_previous_bytes_when_atomic_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed replacement must not destroy the last valid manifest."""
    p = tmp_path / "manifest.json"
    save_manifest(p, {"old": _entry("old-latest.osm.pbf")})
    before = p.read_bytes()

    def fail(*args, **kwargs) -> None:
        raise OSError("simulated interrupted manifest write")

    monkeypatch.setattr("osm_polygon_wikidata_only.io.manifest.atomic_write_text", fail)
    with pytest.raises(OSError, match="simulated interrupted manifest write"):
        save_manifest(p, {"new": _entry("new-latest.osm.pbf")})

    assert p.read_bytes() == before
    assert load_manifest(p)["old"]["polygon_count"] == 10


def test_upsert_inserts_and_updates(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    upsert_entry(
        p,
        source_pbf="monaco-latest.osm.pbf",
        region="monaco",
        polygons_path="polygons/monaco-latest.parquet",
        articles_path="articles/monaco-latest.parquet",
        polygon_articles_path="polygon_articles/monaco-latest.parquet",
        stats=ManifestStats(polygon_count=10),
        extraction_version="0.1.0",
    )
    # Update with new stats.
    upsert_entry(
        p,
        source_pbf="monaco-latest.osm.pbf",
        region="monaco",
        polygons_path="polygons/monaco-latest.parquet",
        articles_path="articles/monaco-latest.parquet",
        polygon_articles_path="polygon_articles/monaco-latest.parquet",
        stats=ManifestStats(polygon_count=20),
        extraction_version="0.1.0",
    )
    entries = load_manifest(p)
    assert len(entries) == 1
    assert entries["monaco-latest.osm.pbf"]["polygon_count"] == 20


def test_iter_entries_is_sorted(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    save_manifest(
        p,
        {
            "c": _entry("c-latest.osm.pbf"),
            "a": _entry("a-latest.osm.pbf"),
            "b": _entry("b-latest.osm.pbf"),
        },
    )
    keys = [k for k, _ in iter_entries(load_manifest(p))]
    assert keys == ["a", "b", "c"]


def test_manifest_path_helper(tmp_path: Path) -> None:
    p = manifest_path(tmp_path)
    assert p.name == "processed_pbfs.json"
