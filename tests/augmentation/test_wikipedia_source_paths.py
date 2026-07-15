"""Tests for the consolidated local Wikipedia source-path abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.augmentation.steps import (
    WikipediaSourcePaths,
    read_source_path,
    wikipedia_source_paths,
)
from osm_polygon_wikidata_only.config.paths import DataRoot

STEM = "monaco-latest"


def _make_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy", encoding="utf-8")


# ---------------------------------------------------------------------------
# Factory + deterministic paths
# ---------------------------------------------------------------------------


def test_returns_deterministic_canonical_and_legacy_paths(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)

    result = wikipedia_source_paths(data_root, STEM)

    assert result.canonical == data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    assert result.legacy == data_root.processed_articles / f"{STEM}.parquet"


def test_factory_does_not_mutate_filesystem(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)

    wikipedia_source_paths(data_root, STEM)

    assert not (data_root.processed_articles / f"{STEM}.parquet").exists()
    assert not (data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet").exists()
    assert not data_root.processed_articles.exists()
    assert not (data_root.processed / "wikipedia" / "documents").exists()


# ---------------------------------------------------------------------------
# read_source_path (load_core_inputs policy)
# ---------------------------------------------------------------------------


def test_read_source_selects_legacy_when_only_legacy_present(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    _make_file(legacy)

    assert read_source_path(data_root, STEM) == legacy


def test_read_source_selects_canonical_when_only_canonical_present(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    _make_file(canonical)

    assert read_source_path(data_root, STEM) == canonical


def test_read_source_prefers_legacy_when_both_present(tmp_path: Path) -> None:
    """During migration both coexist; legacy is the source of truth."""
    data_root = DataRoot(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    _make_file(legacy)
    _make_file(canonical)

    assert read_source_path(data_root, STEM) == legacy


def test_read_source_falls_back_to_canonical_when_neither_present(tmp_path: Path) -> None:
    """Absence of both: read_source falls back to canonical (non-existent)."""
    data_root = DataRoot(tmp_path)

    assert read_source_path(data_root, STEM) == (
        data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    )


# ---------------------------------------------------------------------------
# either_exists property
# ---------------------------------------------------------------------------


def test_either_exists_true_when_legacy_only(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    _make_file(data_root.processed_articles / f"{STEM}.parquet")

    assert wikipedia_source_paths(data_root, STEM).either_exists


def test_either_exists_true_when_canonical_only(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    _make_file(data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet")

    assert wikipedia_source_paths(data_root, STEM).either_exists


def test_either_exists_true_when_both_present(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    _make_file(data_root.processed_articles / f"{STEM}.parquet")
    _make_file(data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet")

    assert wikipedia_source_paths(data_root, STEM).either_exists


def test_either_exists_false_when_neither_present(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)

    assert not wikipedia_source_paths(data_root, STEM).either_exists


# ---------------------------------------------------------------------------
# Stem validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_stem",
    ["", ".", "..", "a/b", "a\\b", "../escape"],
)
def test_invalid_stem_raises(tmp_path: Path, bad_stem: str) -> None:
    data_root = DataRoot(tmp_path)

    with pytest.raises((ValueError, OSError)):
        wikipedia_source_paths(data_root, bad_stem)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_result_is_frozen(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    result = wikipedia_source_paths(data_root, STEM)

    with pytest.raises((AttributeError, TypeError)):
        result.canonical = tmp_path / "other"  # type: ignore[misc]


def test_wikipedia_source_paths_class_is_exposed(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    sources = wikipedia_source_paths(data_root, STEM)
    assert isinstance(sources, WikipediaSourcePaths)
