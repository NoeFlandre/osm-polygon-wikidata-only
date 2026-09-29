"""Step definitions for ``language_splits.feature`` (#26, #27, #115)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, scenarios, then, when

from osm_polygon_wikidata_only.v2.language_splits import (
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    V2LanguageSplitError,
    build_v2_language_splits,
)
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from tests.v2.language_splits_support import (
    _document,
    _manifest,
    _release_snapshot,
    _rows,
    _shard,
    _write_table,
    _write_v2_fixture,
)

scenarios("language_splits.feature")

# The fixture's rows, by the language partition each must land in.
_EXPECTED_PARTITIONS: dict[str, dict[str, set[str]]] = {
    "wikipedia_documents": {
        "fr": {"doc-fr-a", "doc-fr-z"},
        "de": {"doc-de-a"},
        "unknown": {"doc-unknown-a"},
    },
    "wikipedia_sections": {
        "fr": {"section-fr-a", "section-fr-z"},
        "de": {"section-de-a"},
        "unknown": {"section-unknown-a"},
    },
}
_ID_COLUMN = {"wikipedia_documents": "document_id", "wikipedia_sections": "section_id"}
_STALE_SHARD_PATH = (
    "language_splits/wikipedia_documents_by_language/lang-de/part-00000-of-00001.parquet"
)


@dataclass
class _State:
    root: Path | None = None
    source_rows: dict[str, set[str]] = field(default_factory=dict)
    source_bytes: dict[str, bytes] = field(default_factory=dict)
    first_outputs: dict[str, bytes] = field(default_factory=dict)
    error: Exception | None = None


def _source_ids(root: Path, directory: str, column: str) -> set[str]:
    return {
        str(row[column])
        for path in sorted((root / directory).glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    }


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.parquet"))
        if "language_splits" not in path.parts
    }


@pytest.fixture
def state() -> _State:
    return _State()


@given("a V2 region with French, German and unusable-language rows")
def v2_region(state: _State, tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    state.root = root
    state.source_rows = {
        "wikipedia_documents": _source_ids(root, "wikipedia/documents", "document_id"),
        "wikipedia_sections": _source_ids(root, "wikipedia/sections", "section_id"),
    }
    state.source_bytes = _source_snapshot(root)


@given("the language splits were built once")
@when("I build the language splits")
def build_once(state: _State) -> None:
    assert state.root is not None
    build_v2_language_splits(state.root)
    state.first_outputs = _release_snapshot(state.root)


@when("I build the language splits again")
def build_again(state: _State) -> None:
    assert state.root is not None
    build_v2_language_splits(state.root)


@when("the German document disappears and I rebuild the language splits")
def drop_german_and_rebuild(state: _State) -> None:
    assert state.root is not None
    assert _shard(state.root, "wikipedia_documents", "de", "part-00000-of-00001").is_file()
    _write_table(
        state.root / "wikipedia/documents/a-latest.parquet",
        [_document("doc-fr-a", "fr", "French A")],
        wikipedia_document_v2_schema(),
    )
    build_v2_language_splits(state.root)


@when("I build the language splits into the processed root itself")
def build_overlapping(state: _State) -> None:
    assert state.root is not None
    try:
        build_v2_language_splits(state.root, output_root=state.root)
    except V2LanguageSplitError as error:
        state.error = error


@then("each language shard contains only rows of its own language")
def shards_are_pure(state: _State) -> None:
    assert state.root is not None
    for table, partitions in _EXPECTED_PARTITIONS.items():
        for language, expected_ids in partitions.items():
            rows: list[dict[str, Any]] = _rows(state.root, table, language)
            assert {str(row[_ID_COLUMN[table]]) for row in rows} == expected_ids


@then("the union of the shards equals the input rows")
def union_is_input(state: _State) -> None:
    assert state.root is not None
    for table, partitions in _EXPECTED_PARTITIONS.items():
        observed = [
            str(row[_ID_COLUMN[table]])
            for language in partitions
            for row in _rows(state.root, table, language)
        ]
        assert len(observed) == len(set(observed))
        assert set(observed) == state.source_rows[table]


@then("the language split outputs are byte-identical")
def byte_identical(state: _State) -> None:
    assert state.root is not None
    assert _release_snapshot(state.root) == state.first_outputs


@then("the stale German document shard is removed")
def stale_removed(state: _State) -> None:
    assert state.root is not None
    stale = _shard(state.root, "wikipedia_documents", "de", "part-00000-of-00001")
    assert not stale.exists()
    assert not stale.parent.exists()


@then("the manifest no longer references the German document shard")
def manifest_forgets_stale(state: _State) -> None:
    assert state.root is not None
    manifest = _manifest(state.root)
    paths = {
        file["path"]
        for table in cast(list[dict[str, object]], manifest["tables"])
        for bucket in cast(list[dict[str, object]], table["buckets"])
        for file in cast(list[dict[str, object]], bucket["files"])
    }
    assert _STALE_SHARD_PATH not in paths


@then("the build is rejected for overlapping its source")
def overlap_rejected(state: _State) -> None:
    assert state.error is not None
    assert "must not overlap" in str(state.error)


@then("the source Parquet files are unchanged")
def sources_unchanged(state: _State) -> None:
    assert state.root is not None
    assert _source_snapshot(state.root) == state.source_bytes
    assert not (state.root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).exists()
