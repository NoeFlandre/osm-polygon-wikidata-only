"""Resuming an interrupted V2 language-split release must never reuse stale shards."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf.language_splits import LanguageTable, LanguageTableInventory
from osm_polygon_wikidata_only.v2 import language_splits
from osm_polygon_wikidata_only.v2.language_split_models import (
    V2LanguageSplitError,
    V2LanguageSplitFile,
)
from osm_polygon_wikidata_only.v2.language_split_writer import TableWriteContext
from osm_polygon_wikidata_only.v2.language_splits import build_v2_language_splits
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from tests.v2.language_splits_support import _document, _write_table, _write_v2_fixture

_FRENCH_DOCUMENTS = "wikipedia/documents/z-latest.parquet"


def _interrupt_before_install(root: Path) -> None:
    """Run a release that stages every table and then crashes before installing."""

    def crash(*_args: object) -> None:
        raise RuntimeError("crash before install")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(language_splits, "_install_staged_files", crash)
        with pytest.raises(RuntimeError, match="crash before install"):
            build_v2_language_splits(root)


def _french_titles(root: Path) -> list[str]:
    directory = root / "language_splits/wikipedia_documents_by_language/lang-fr"
    return sorted(
        row["title"]
        for path in directory.glob("*.parquet")
        for row in pq.read_table(path).to_pylist()
    )


def _rewrite_french_document(root: Path, title: str) -> None:
    _write_table(
        root / _FRENCH_DOCUMENTS,
        [_document("doc-fr-z", " FR ", title)],
        wikipedia_document_v2_schema(),
    )


def test_resume_rebuilds_shards_whose_source_content_changed(tmp_path: Path) -> None:
    """An edit that keeps row counts must not be papered over by the old staging tree."""
    root = _write_v2_fixture(tmp_path)
    _interrupt_before_install(root)

    _rewrite_french_document(root, "French Z CORRECTED")
    build_v2_language_splits(root)

    assert _french_titles(root) == ["French A", "French Z CORRECTED"]


def test_resume_rebuilds_when_content_changes_and_mtime_is_restored(tmp_path: Path) -> None:
    """Content identity, not file metadata, decides whether a staged shard is reused."""
    root = _write_v2_fixture(tmp_path)
    source = root / _FRENCH_DOCUMENTS
    before = source.stat()
    _interrupt_before_install(root)

    _rewrite_french_document(root, "French Z CORRECTED")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    build_v2_language_splits(root)

    assert _french_titles(root) == ["French A", "French Z CORRECTED"]


def test_resume_reuses_staged_shards_when_sources_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged sources still resume: no table is restaged after an interrupted run."""
    root = _write_v2_fixture(tmp_path)
    _interrupt_before_install(root)

    def restage(*_args: object) -> None:
        raise AssertionError("unchanged completed tables must not be restaged")

    monkeypatch.setattr(language_splits, "_write_table", restage)
    build_v2_language_splits(root)

    assert _french_titles(root) == ["French A", "French Z"]


def test_resume_rebuilds_shards_when_max_rows_per_shard_changes(tmp_path: Path) -> None:
    """A resume with a different shard size must not reuse the old shard layout."""
    root = _write_v2_fixture(tmp_path)
    _interrupt_before_install(root)

    build_v2_language_splits(root, max_rows_per_shard=1)

    directory = root / "language_splits/wikipedia_documents_by_language/lang-fr"
    shard_rows = [pq.read_table(path).num_rows for path in directory.glob("*.parquet")]
    assert shard_rows
    assert max(shard_rows) == 1
    assert _french_titles(root) == ["French A", "French Z"]


def test_source_edited_during_staging_is_rebuilt_on_resume(tmp_path: Path) -> None:
    """A source edited while its table is staged must not be recorded as the old bytes."""
    root = _write_v2_fixture(tmp_path)
    real_write_table = language_splits._write_table

    def write_then_edit(
        root_arg: Path,
        context: TableWriteContext,
        table: LanguageTableInventory,
        batch_size: int,
    ) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
        result = real_write_table(root_arg, context, table, batch_size)
        if table.table is LanguageTable.WIKIPEDIA_DOCUMENTS:
            _rewrite_french_document(root, "French Z CORRECTED")
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(language_splits, "_write_table", write_then_edit)
        with pytest.raises(V2LanguageSplitError, match="changed during generation"):
            build_v2_language_splits(root)

    build_v2_language_splits(root)

    assert _french_titles(root) == ["French A", "French Z CORRECTED"]
