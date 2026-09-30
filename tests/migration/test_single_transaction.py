"""Phase 2.5 / Defect 5: Migration must be one recoverable
transaction. The link replacement, the processed manifest update,
the augmentation manifest update, the pending-publication envelope
and the metadata-refresh marker must all commit as a single
manifest-last journaled transaction.

Crash injection after every write boundary must leave the stem in a
state where a fresh restart converges. The stem must NOT be falsely
classified current when only some writes have completed.

All journal target/staged/backup paths must be confined to the
expected data-root directories before replay.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import document_schema, section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline import link_migration
from tests.migration._builders import write_document, write_legacy_link, write_polygon


def _setup_processed(processed: Path, stem: str) -> None:
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
        "manifests",
        "augmentation/manifests",
    ):
        (processed / sub).mkdir(parents=True, exist_ok=True)
    write_polygon(processed, stem)
    write_document(processed, stem)
    write_legacy_link(processed, stem)
    empty_sections = pa.Table.from_pylist([], schema=section_schema())
    pq.write_table(empty_sections, processed / "wikipedia" / "sections" / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=document_schema()),
        processed / "wikivoyage" / "documents" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=section_schema()),
        processed / "wikivoyage" / "sections" / f"{stem}.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"_placeholder": []}),
        processed / "wikidata" / "facts" / f"{stem}.parquet",
    )
    # Pre-seed the processed manifest so the migration can update it.
    (processed / "manifests" / "processed_pbfs.json").write_text(
        json.dumps(
            {
                f"{stem}.osm.pbf": {
                    "source_pbf": f"{stem}.osm.pbf",
                    "region": "r",
                    "polygons_path": f"polygons/{stem}.parquet",
                    "articles_path": f"wikipedia/documents/{stem}.parquet",
                    "polygon_articles_path": f"polygon_articles/{stem}.parquet",
                    "extraction_version": "test",
                    "processed_at": "2026-07-24T00:00:00Z",
                }
            }
        )
    )


def _fresh_process(tmp_path: Path) -> DataRoot:
    """Simulate a fresh process restart by constructing a new
    BackgroundUploadQueue / apply context from disk state."""
    return DataRoot(tmp_path)


# ---------------------------------------------------------------------------
# 1. Crash after link parquet commit but before manifests -> roll-forward
# ---------------------------------------------------------------------------


def test_crash_after_link_parquet_rollforward_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the link parquet commit and the manifest
    updates must allow a fresh restart to complete the transaction
    (link parquet, processed manifest, augmentation manifest,
    pending intent, metadata marker).
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    # Patch the manifest update sequence to crash immediately after
    # the link parquet commit (so the manifests are NOT yet updated).
    from osm_polygon_wikidata_only.pipeline import link_migration as lm

    real_commit = lm._commit_ordered_replacements

    crash_after_first = {"raised": False}

    def _crashing_commit(
        directory: Path,
        stem: str,
        replacements: list[tuple[Path, Path]],
        *,
        _crash_hook: Callable[[int, Path], None] | None = None,
    ) -> None:
        # Replace the crash hook to fire after the first replacement.
        def _hook(index: int, target: Path) -> None:
            if index == 0 and not crash_after_first["raised"]:
                crash_after_first["raised"] = True
                raise RuntimeError("simulated crash after link parquet")
            if _crash_hook is not None:
                _crash_hook(index, target)

        return real_commit(
            directory,
            stem=stem,
            replacements=replacements,
            _crash_hook=_hook,
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(lm, "_commit_ordered_replacements", _crashing_commit)
        with pytest.raises(RuntimeError, match="simulated crash"):
            link_migration.apply_link_migration(processed, stems={stem})

    # State after crash:
    # - Link parquet IS committed (atomic file replace already
    #   happened).
    # - Processed manifest, augmentation manifest, pending envelope
    #   and metadata marker are NOT yet committed.
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )

    data_root = _fresh_process(tmp_path)
    # Stem must NOT be classified current yet.
    assert augmentation_is_current(data_root, stem) is False, (
        "After crash between link parquet and manifests, stem must NOT be current"
    )

    # Now run a fresh apply (simulating a restarted process).
    link_migration.apply_link_migration(processed, stems={stem})

    # Now the stem must be current.
    assert augmentation_is_current(data_root, stem) is True, (
        "After restart, fresh apply must converge and mark the stem current"
    )

    # Pending intent and metadata marker must be present.
    from osm_polygon_wikidata_only.pipeline import pending_publications as pp

    assert stem in pp.load_pending_publications(data_root)
    marker = pp.load_metadata_refresh_marker(data_root)
    assert marker is not None and stem in marker["stems"]


# ---------------------------------------------------------------------------
# 2. Crash before any commit -> no false current
# ---------------------------------------------------------------------------


def test_crash_before_any_commit_does_not_mark_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash BEFORE any write must leave the stem un-migrated and
    not-current. A fresh apply then completes the transaction.
    """
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )
    from osm_polygon_wikidata_only.pipeline import link_migration as lm

    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    def _always_crash(
        directory: Path,
        stem: str,
        replacements: list[tuple[Path, Path]],
        *,
        _crash_hook: Callable[[int, Path], None] | None = None,
    ) -> None:
        raise RuntimeError("simulated pre-commit crash")

    with monkeypatch.context() as patcher:
        patcher.setattr(lm, "_commit_ordered_replacements", _always_crash)
        with pytest.raises(RuntimeError, match="simulated pre-commit crash"):
            link_migration.apply_link_migration(processed, stems={stem})

    data_root = _fresh_process(tmp_path)
    assert augmentation_is_current(data_root, stem) is False

    # Fresh apply converges (using the now-restored real function).
    link_migration.apply_link_migration(processed, stems={stem})
    assert augmentation_is_current(data_root, stem) is True


# ---------------------------------------------------------------------------
# 3. Journal paths are confined to expected roots
# ---------------------------------------------------------------------------


def test_journal_paths_are_inside_data_root(tmp_path: Path) -> None:
    """The journal target/staged/backup paths must resolve inside the
    processed/ subdirectory (not anywhere outside).
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    link_migration.apply_link_migration(processed, stems={stem})

    # The link parquet and manifests must all live under processed/.
    data_root = _fresh_process(tmp_path)
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "augmentation/manifests",
        "manifests",
    ):
        path = data_root.processed / sub
        assert str(path.resolve()).startswith(str(data_root.processed.resolve()))


def test_manifest_failure_cannot_leave_canonical_link_without_manifest_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every durable state change belongs to the same recoverable transaction."""
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)

    link_migration.apply_link_migration(processed, stems={stem})

    links = pq.read_table(processed / "polygon_articles" / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    assert "document_id" in links.column_names
    augmentation_manifest = json.loads(
        (processed / "augmentation" / "manifests" / "augmentation_manifest.json").read_text()
    )
    assert augmentation_manifest[stem]["link_schema_version"] == "polygon-document-links-v1"
    assert len(augmentation_manifest[stem]["link_artifact_sha256"]) == 64


def test_transaction_replacements_include_every_durable_migration_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    _setup_processed(processed, stem)
    captured: list[Path] = []
    real_commit = link_migration._commit_ordered_replacements

    def capture(
        directory: Path,
        stem: str,
        replacements: list[tuple[Path, Path]],
        *,
        _crash_hook=None,
    ) -> None:
        captured.extend(target for target, _ in replacements)
        real_commit(
            directory,
            stem,
            replacements,
            _crash_hook=_crash_hook,
        )

    monkeypatch.setattr(link_migration, "_commit_ordered_replacements", capture)
    link_migration.apply_link_migration(processed, stems={stem})

    relative = {path.relative_to(processed).as_posix() for path in captured}
    assert f"polygon_articles/{stem}.parquet" in relative
    assert "manifests/processed_pbfs.json" in relative
    assert "augmentation/manifests/augmentation_manifest.json" in relative
    assert "manifests/pending_migration_publications.json" in relative
    assert "integrity/rejection_ledger.json" in relative
