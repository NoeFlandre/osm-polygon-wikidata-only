"""Boundary tests for PBF path validation and streaming error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

import osm_polygon_wikidata_only.io.pbf_reader as pbf_reader


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("monaco-latest.osm.pbf", "monaco"),
        ("north-america-latest.osm.pbf", "north-america"),
    ],
)
def test_region_name_accepts_geofabrik_stems(filename: str, expected: str) -> None:
    assert pbf_reader.region_from_filename(filename) == expected


@pytest.mark.parametrize("filename", ["monaco.osm.pbf", "monaco-latest.pbf", "monaco.osm"])
def test_region_name_rejects_non_geofabrik_filenames(filename: str) -> None:
    with pytest.raises(ValueError, match="does not match the Geofabrik pattern"):
        pbf_reader.region_from_filename(filename)


def test_reader_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(pbf_reader.PBFReadError, match="does not exist"):
        pbf_reader.PBFReader(tmp_path / "missing-latest.osm.pbf")


def test_reader_rejects_directory_path(tmp_path: Path) -> None:
    path = tmp_path / "directory-latest.osm.pbf"
    path.mkdir()

    with pytest.raises(pbf_reader.PBFReadError, match="is not a file"):
        pbf_reader.PBFReader(path)


def test_iter_polygon_candidates_wraps_osmium_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "monaco-latest.osm.pbf"
    path.touch()

    class FailingHandler:
        def __init__(self, callback: object, *, include_wikipedia_tagged: bool) -> None:
            del callback, include_wikipedia_tagged

        def apply_file(self, filename: str, *, locations: bool) -> None:
            assert filename == str(path)
            assert locations is True
            raise OSError("corrupt PBF")

    monkeypatch.setattr(pbf_reader, "_PolygonHandler", FailingHandler)

    with pytest.raises(pbf_reader.PBFReadError, match=r"Failed to read PBF.*corrupt PBF"):
        pbf_reader.PBFReader(path).iter_polygon_candidates(lambda candidate: None)


@pytest.mark.parametrize("error", [RuntimeError("parser error"), ValueError("bad geometry")])
def test_iter_polygon_candidates_wraps_parser_value_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    path = tmp_path / "monaco-latest.osm.pbf"
    path.touch()

    class FailingHandler:
        def __init__(self, callback: object, *, include_wikipedia_tagged: bool) -> None:
            del callback, include_wikipedia_tagged

        def apply_file(self, filename: str, *, locations: bool) -> None:
            del filename, locations
            raise error

    monkeypatch.setattr(pbf_reader, "_PolygonHandler", FailingHandler)

    with pytest.raises(pbf_reader.PBFReadError, match="Failed to read PBF"):
        pbf_reader.PBFReader(path).iter_polygon_candidates(lambda candidate: None)


def test_iter_polygon_candidates_delivers_callbacks_without_collecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "monaco-latest.osm.pbf"
    path.touch()
    candidate = ("way", 42, {"wikidata": "Q42"}, "{}")

    class StubHandler:
        def __init__(self, callback, *, include_wikipedia_tagged: bool) -> None:
            self.callback = callback
            self.include_wikipedia_tagged = include_wikipedia_tagged

        def apply_file(self, filename: str, *, locations: bool) -> None:
            assert filename == str(path)
            assert locations is True
            self.callback(candidate)

    monkeypatch.setattr(pbf_reader, "_PolygonHandler", StubHandler)
    received: list[pbf_reader.PolygonCandidate] = []

    reader = pbf_reader.PBFReader(path, include_wikipedia_tagged=True)
    reader.iter_polygon_candidates(received.append)

    assert received == [candidate]


def test_collect_polygon_candidates_uses_streaming_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "monaco-latest.osm.pbf"
    path.touch()
    candidates = [
        ("way", 1, {"wikidata": "Q1"}, "{}"),
        ("relation", 2, {"wikidata": "Q2"}, "{}"),
    ]

    class StubHandler:
        def __init__(self, callback, *, include_wikipedia_tagged: bool) -> None:
            del include_wikipedia_tagged
            self.callback = callback

        def apply_file(self, filename: str, *, locations: bool) -> None:
            del filename, locations
            for candidate in candidates:
                self.callback(candidate)

    monkeypatch.setattr(pbf_reader, "_PolygonHandler", StubHandler)

    assert pbf_reader.PBFReader(path).collect_polygon_candidates() == candidates


def test_reader_region_name_is_derived_from_path_basename(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "alps-latest.osm.pbf"
    path.touch()

    assert pbf_reader.PBFReader(path).region_name == "alps"
