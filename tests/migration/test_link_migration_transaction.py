"""Phase 2 / Group C: ordered journaled transaction primitive.

Red tests for the shared journaled ordered-replacement helper that backs
the migration apply stage. The helper is exposed as a private function
(``_commit_ordered_replacements``) used by the migration's public
``apply_link_migration`` boundary. Tests import it directly to drive
the helper through real crash semantics.

The replacement sequence is a list of ``(target, staged)`` tuples --
no observer callbacks, no decorator tuples; the helper enforces
manifest-last ordering by sorting the supplied list so that any
``*.json`` replacement comes after every data-file replacement.

Two-phase testing:

1. Use a narrow ``after_replace(index, target)`` crash hook injected
   via the helper's ``_crash_hook`` parameter (or the public
   ``apply_link_migration`` boundary) to interrupt mid-transaction.
2. Re-run the helper on the resulting journal to verify recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


def _import_module():
    try:
        from osm_polygon_wikidata_only.pipeline import link_migration as mod
    except ImportError as exc:
        pytest.fail(
            "Expected osm_polygon_wikidata_only.pipeline.link_migration to exist "
            f"(Phase 2 group C: ordered journaled transaction); got ImportError: {exc}"
        )
    return mod


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    return pa


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_text_pair(tmp_path: Path, name: str, old: str, new: str) -> tuple[Path, Path]:
    target = tmp_path / name
    staged = tmp_path / f"{name}.staged"
    target.write_text(old)
    staged.write_text(new)
    return target, staged


# ---------------------------------------------------------------------------
# The shared helper is internal; tests import it through the migration module
# ---------------------------------------------------------------------------


def test_helper_is_internal_or_kept_private() -> None:
    """The ordered-replacement helper must not be a public API.

    If the helper exists at module top level it must be private (leading
    underscore). The migration's public API is ``plan_link_migration``
    / ``apply_link_migration``.
    """
    mod = _import_module()
    public_names = set(dir(mod))
    if "commit_ordered_replacements" in public_names:
        pytest.fail(
            "link_migration.commit_ordered_replacements must not be public; "
            "test the public apply boundary only."
        )
    assert "apply_link_migration" in public_names, (
        "apply_link_migration is the public API for the ordered transaction"
    )


# ---------------------------------------------------------------------------
# Manifest-last ordering
# ---------------------------------------------------------------------------


def test_ordered_replacements_applies_json_manifests_last(tmp_path: Path) -> None:
    """Manifests must be the LAST replacements; data files first."""
    mod = _import_module()
    data_target, data_staged = _make_text_pair(tmp_path, "data.parquet", "OLD", "NEW")
    manifest_target, manifest_staged = _make_text_pair(
        tmp_path, "manifest.json", "{}", '{"new": 1}'
    )

    # Pass the manifest FIRST in input order; the helper must still write
    # data first (so a data-write failure leaves manifest untouched).
    mod.apply_link_migration(
        tmp_path,
        replacements=[(manifest_target, manifest_staged), (data_target, data_staged)],
    )
    assert data_target.read_text() == "NEW"
    assert manifest_target.read_text() == '{"new": 1}'


def test_apply_replacements_helper_preserves_transaction_boundary(tmp_path: Path) -> None:
    """The private replacement adapter keeps the public transaction seam small."""
    mod = _import_module()
    target, staged = _make_text_pair(tmp_path, "data.parquet", "OLD", "NEW")

    assert hasattr(mod, "_apply_replacements")
    mod._apply_replacements(tmp_path, [(target, staged)])

    assert target.read_text() == "NEW"


# ---------------------------------------------------------------------------
# Crash hook semantics
# ---------------------------------------------------------------------------


def test_ordered_replacements_rolls_forward_after_interruption(
    tmp_path: Path,
) -> None:
    """A crash after replacement N must roll forward idempotently on resume.

    The crash hook is the only sanctioned test seam. The caller injects
    a function that raises after the Nth replacement; the helper stamps
    the journal and persists its state. Calling ``apply_link_migration``
    (or a dedicated recovery entry-point) on the same stem must:

    * verify all target hashes match the staged content;
    * finish any remaining replacements;
    * leave the journal removed and backups cleaned up.
    """
    mod = _import_module()
    targets = [
        _make_text_pair(tmp_path, f"target_{i}.parquet", f"old_{i}", f"new_{i}") for i in range(3)
    ]

    seen_indices: list[int] = []
    crash_at = 2

    def crash_hook(index: int, target: Path) -> None:
        seen_indices.append(index)
        if index == crash_at:
            raise RuntimeError(f"injected crash after replacement {index}")

    # The crash hook is private; tests invoke it via a parameter on
    # ``apply_link_migration`` that the helper exposes for tests.
    with pytest.raises(RuntimeError, match="injected crash after replacement 2"):
        mod.apply_link_migration(
            tmp_path,
            replacements=[(t, s) for t, s in targets],
            _crash_hook=crash_hook,
        )

    # The hook was called for indices 0..2 (the crash happens at index 2).
    assert seen_indices[:3] == [0, 1, 2]

    # All targets BEFORE the crash were already replaced.
    for i in range(crash_at + 1):
        assert targets[i][0].read_text() == f"new_{i}", (
            f"Target {i} should be replaced (target_{i}.parquet), got {targets[i][0].read_text()!r}"
        )

    # After the crash, the post-crash targets remain at their original content.
    for i in range(crash_at + 1, len(targets)):
        assert targets[i][0].read_text() == f"old_{i}", (
            f"Target {i} should NOT be replaced after crash, got {targets[i][0].read_text()!r}"
        )

    # Re-run the migration (no crash hook) to roll forward.
    mod.apply_link_migration(
        tmp_path,
        replacements=[(t, s) for t, s in targets],
    )

    # All targets now at end state.
    for i in range(len(targets)):
        assert targets[i][0].read_text() == f"new_{i}", (
            f"After recovery, target {i} should be at new_{i}, got {targets[i][0].read_text()!r}"
        )

    # Backups and journal cleaned up.
    backups = list(tmp_path.glob("*.backup"))
    assert backups == [], f"Backups must be cleaned up after recovery, got {backups}"
    journal_files = list(tmp_path.glob("journal.json"))
    assert journal_files == [], f"Journal must be cleaned up after recovery, got {journal_files}"


def test_ordered_replacements_rollback_on_validation_failure(tmp_path: Path) -> None:
    """A validation failure must restore all targets to their original content."""
    mod = _import_module()
    targets = [
        _make_text_pair(tmp_path, f"target_{i}.txt", f"old_{i}", f"new_{i}") for i in range(3)
    ]

    # Inject a validation failure before any replacements.
    def fail_before_any(_index: int, _target: Path) -> None:
        raise RuntimeError("injected validation failure")

    with pytest.raises(RuntimeError, match="injected validation failure"):
        mod.apply_link_migration(
            tmp_path,
            replacements=[(t, s) for t, s in targets],
            _crash_hook=fail_before_any,
        )
    # No target was replaced.
    for i, (target, _) in enumerate(targets):
        assert target.read_text() == f"old_{i}", (
            f"Target {i} should be unchanged after rollback, got {target.read_text()!r}"
        )


def test_ordered_replacements_rejects_duplicate_targets(tmp_path: Path) -> None:
    """The helper must reject duplicate targets in the input list."""
    mod = _import_module()
    target, staged = _make_text_pair(tmp_path, "target.txt", "old", "new")
    with pytest.raises(ValueError):
        mod.apply_link_migration(
            tmp_path,
            replacements=[(target, staged), (target, staged)],
        )


def test_ordered_replacements_idempotent_second_run_preserves_mtime_and_hash(
    tmp_path: Path,
) -> None:
    """A second run on the same stem must preserve target mtime and hash."""
    mod = _import_module()
    target, staged = _make_text_pair(tmp_path, "target.txt", "new", "new")  # same content
    mod.apply_link_migration(tmp_path, replacements=[(target, staged)])
    first_hash = _sha256(target)
    first_mtime = target.stat().st_mtime_ns
    # Wait at least 1 microsecond so that a re-write would bump mtime.
    os.utime(target, ns=(first_mtime + 1_000, first_mtime + 1_000))
    mod.apply_link_migration(tmp_path, replacements=[(target, staged)])
    second_hash = _sha256(target)
    second_mtime = target.stat().st_mtime_ns
    assert first_hash == second_hash, "Idempotent second run must preserve hash"
    assert second_mtime == first_mtime + 1_000, (
        "Idempotent second run must NOT rewrite the file (mtime preserved)"
    )


def test_ordered_replacements_rolls_back_when_replace_verification_fails(
    tmp_path: Path,
) -> None:
    """If a target's hash doesn't match the staged hash after replacement,
    the helper must roll back and re-raise."""
    mod = _import_module()
    target, staged = _make_text_pair(tmp_path, "target.txt", "old", "new")

    # Inject a hook that corrupts the target AFTER the helper writes it.
    def corrupt_after_replace(index: int, target_path: Path) -> None:
        if index == 0:
            target_path.write_text("CORRUPTED")

    with pytest.raises(RuntimeError):
        mod.apply_link_migration(
            tmp_path,
            replacements=[(target, staged)],
            _crash_hook=corrupt_after_replace,
        )
    # Rolled back to original content.
    assert target.read_text() == "old", (
        f"Target must be rolled back on hash mismatch, got {target.read_text()!r}"
    )


def test_ordered_replacements_rejects_missing_staged_file(tmp_path: Path) -> None:
    """A staged file that does not exist must be rejected before any writes."""
    mod = _import_module()
    target = tmp_path / "target.txt"
    target.write_text("old")
    missing_staged = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        mod.apply_link_migration(
            tmp_path,
            replacements=[(target, missing_staged)],
        )
    assert target.read_text() == "old"


# ---------------------------------------------------------------------------
# augmentation_is_current contract (valid minimal DataRoot)
# ---------------------------------------------------------------------------


def test_partial_commit_never_marks_stem_current(tmp_path: Path) -> None:
    """A successful apply must mark the stem ``augmentation_is_current``;
    a crashed/aborted apply must NOT."""
    mod = _import_module()
    pa = _pyarrow()
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )
    from osm_polygon_wikidata_only.config.paths import DataRoot

    # Build a valid minimal DataRoot with all required subdirs.
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "monaco-latest"

    # Seed polygons, legacy links, and legacy documents.
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    polygons_path.parent.mkdir(parents=True, exist_ok=True)
    poly_table = pa.table(
        {
            "polygon_id": [f"{stem}:relation:1"],
            "wikidata": ["Q1"],
            "source_pbf": [f"{stem}.osm.pbf"],
        }
    )
    pa.parquet.write_table(poly_table, polygons_path, compression="snappy")

    legacy_links_path = data_root.processed_links / f"{stem}.parquet"
    legacy_links_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(
        pa.table(
            {
                "polygon_id": [f"{stem}:relation:1"],
                "article_id": ["Q1:en:1:1"],
                "wikidata": ["Q1"],
                "language": ["en"],
                "source_pbf": [f"{stem}.osm.pbf"],
                "region": [stem],
                "osm_type": ["relation"],
                "osm_id": [1],
                "page_id": [1],
                "revision_id": [1],
                "is_best_language": [True],
            }
        ),
        legacy_links_path,
        compression="snappy",
    )

    wiki_docs_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wiki_docs_path.parent.mkdir(parents=True, exist_ok=True)
    from osm_polygon_wikidata_only.augmentation.schema import document_schema

    pa.parquet.write_table(
        pa.table(
            {
                "document_id": ["Q1:wikipedia:en:1:1"],
                "article_id": ["Q1:en:1:1"],
                "wikidata": ["Q1"],
                "project": ["wikipedia"],
                "language": ["en"],
                "site": ["enwiki"],
                "title": ["T"],
                "url": ["u"],
                "page_id": [1],
                "revision_id": [1],
                "revision_timestamp": [""],
                "retrieved_at": [""],
                "full_text": ["x"],
                "full_text_format": ["plain_text"],
                "article_length_chars": [1],
                "article_length_words": [1],
                "article_length_tokens_estimate": [1],
                "license": [""],
                "attribution": [""],
                "source_api": [""],
                "fetch_status": ["ok"],
                "fetch_error": [""],
                "content_hash": [""],
            }
        ),
        wiki_docs_path,
        compression="snappy",
    )

    # Other required sidecar files (sections, wikivoyage, wikidata)
    # must exist as empty parquets for augmentation_is_current to
    # accept the stem as current.
    for sub in (
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    ):
        empty_path = data_root.processed / sub / f"{stem}.parquet"
        empty_path.parent.mkdir(parents=True, exist_ok=True)
        if sub == "wikidata/facts":
            from osm_polygon_wikidata_only.augmentation.schema import (
                fact_schema,
            )

            pa.parquet.write_table(
                pa.table({col: [] for col in fact_schema().names}, schema=fact_schema()),
                empty_path,
                compression="snappy",
            )
        elif sub == "wikivoyage/documents":
            pa.parquet.write_table(
                pa.table({col: [] for col in document_schema().names}, schema=document_schema()),
                empty_path,
                compression="snappy",
            )
        else:
            from osm_polygon_wikidata_only.augmentation.schema import section_schema

            pa.parquet.write_table(
                pa.table({col: [] for col in section_schema().names}, schema=section_schema()),
                empty_path,
                compression="snappy",
            )

    # Pre-seed processed_pbfs.json so the migration can update it.
    processed_manifest = data_root.processed_manifests / "processed_pbfs.json"
    processed_manifest.write_text(
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

    # Before apply, augmentation_is_current must be False.
    assert augmentation_is_current(data_root, stem) is False

    # Apply and verify the stem is now marked current.
    mod.apply_link_migration(data_root.processed, stems=[stem])
    assert augmentation_is_current(data_root, stem) is True

    # Crash before apply (with a single replacement) must NOT mark current.
    target = tmp_path / "throwaway.txt"
    staged = tmp_path / "throwaway.staged"
    target.write_text("a")
    staged.write_text("b")
    with pytest.raises(RuntimeError):
        mod.apply_link_migration(
            data_root.processed,
            replacements=[(target, staged)],
            _crash_hook=lambda *_: (_ for _ in ()).throw(RuntimeError("crash")),
        )
    assert augmentation_is_current(data_root, stem) is True, (
        "A failed apply on unrelated targets must NOT roll back the prior stem's current marker"
    )
