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

import codecs
import json
import os
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline._link_migration import application
from osm_polygon_wikidata_only.pipeline._link_migration import transaction as transaction_module
from tests.helpers import sha256_file as _sha256


def _pyarrow():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    return pa


def _make_text_pair(tmp_path: Path, name: str, old: str, new: str) -> tuple[Path, Path]:
    target = tmp_path / name
    staged = tmp_path / f"{name}.staged"
    target.write_text(old)
    staged.write_text(new)
    return target, staged


def _capture_journal_writes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    original_write = transaction_module.atomic_write_journal_json
    journal_payloads: list[dict[str, Any]] = []

    def capture(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "journal.json":
            journal_payloads.append(json.loads(json.dumps(payload)))
        original_write(path, payload)

    monkeypatch.setattr(transaction_module, "atomic_write_journal_json", capture)
    return journal_payloads


# ---------------------------------------------------------------------------
# The shared helper is internal; tests import it through the migration module
# ---------------------------------------------------------------------------


def test_helper_is_internal_or_kept_private() -> None:
    """The ordered-replacement helper must not be a public API.

    If the helper exists at module top level it must be private (leading
    underscore). The migration's public API is ``plan_link_migration``
    / ``apply_link_migration``.
    """
    public_names = set(dir(link_migration))
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
    data_target, data_staged = _make_text_pair(tmp_path, "z_data.parquet", "OLD", "NEW")
    manifest_target, manifest_staged = _make_text_pair(
        tmp_path, "a_manifest.json", "{}", '{"new": 1}'
    )
    seen_targets: list[Path] = []

    # Pass the manifest FIRST in input order; the helper must still write
    # data first (so a data-write failure leaves manifest untouched).
    link_migration.apply_link_migration(
        tmp_path,
        replacements=[(manifest_target, manifest_staged), (data_target, data_staged)],
        _crash_hook=lambda _index, target: seen_targets.append(target),
    )
    assert seen_targets == [data_target, manifest_target]
    assert data_target.read_text() == "NEW"
    assert manifest_target.read_text() == '{"new": 1}'


def test_replacement_order_uses_target_paths_not_staged_paths(tmp_path: Path) -> None:
    target_a, staged_z = _make_text_pair(tmp_path, "target-a.parquet", "old-a", "new-a")
    target_b, staged_a = _make_text_pair(tmp_path, "target-b.parquet", "old-b", "new-b")
    staged_z = staged_z.with_name("z-staged.parquet")
    staged_a = staged_a.with_name("a-staged.parquet")
    staged_z.write_text("new-a")
    staged_a.write_text("new-b")
    seen: list[Path] = []

    transaction_module.commit_ordered_replacements(
        tmp_path / "txn",
        "alpha",
        [(target_b, staged_a), (target_a, staged_z)],
        data_root=tmp_path,
        _crash_hook=lambda _index, target: seen.append(target),
    )

    assert seen == [target_a, target_b]


def test_apply_replacements_helper_preserves_transaction_boundary(tmp_path: Path) -> None:
    """The private replacement adapter keeps the public transaction seam small."""
    target, staged = _make_text_pair(tmp_path, "data.parquet", "OLD", "NEW")

    application._apply_replacements(tmp_path, [(target, staged)])

    assert target.read_text() == "NEW"


# ---------------------------------------------------------------------------
# Crash hook semantics
# ---------------------------------------------------------------------------


def test_ordered_replacements_rolls_forward_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    targets = [
        _make_text_pair(tmp_path, f"target_{i}.parquet", f"old_{i}", f"new_{i}") for i in range(3)
    ]
    journal_payloads = _capture_journal_writes(monkeypatch)
    expected_entries = [
        {
            "target": str(target),
            "staged": str(staged),
            "backup": str(tmp_path / ".link_migration_journal" / f"{target.name}.backup"),
            "existed": True,
            "original_hash": _sha256(target),
            "staged_hash": _sha256(staged),
        }
        for target, staged in targets
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
        link_migration.apply_link_migration(
            tmp_path,
            replacements=[(t, s) for t, s in targets],
            _crash_hook=crash_hook,
        )

    prepared, interrupted = journal_payloads
    assert prepared == {
        "contract_version": transaction_module.TRANSACTION_VERSION,
        "stem": "__replacements__",
        "phase": "prepared",
        "entries": expected_entries,
    }
    assert interrupted == {
        **prepared,
        "phase": "interrupted",
        "interrupted_at_index": crash_at,
    }

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
    link_migration.apply_link_migration(
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


def test_ordered_replacements_rollback_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validation failure must restore all targets to their original content."""
    targets = [
        _make_text_pair(tmp_path, f"target_{i}.txt", f"old_{i}", f"new_{i}") for i in range(3)
    ]
    journal_payloads = _capture_journal_writes(monkeypatch)

    # Inject a validation failure before any replacements.
    def fail_before_any(_index: int, _target: Path) -> None:
        raise RuntimeError("injected validation failure")

    with pytest.raises(RuntimeError, match="injected validation failure"):
        link_migration.apply_link_migration(
            tmp_path,
            replacements=[(t, s) for t, s in targets],
            _crash_hook=fail_before_any,
        )
    assert [payload["phase"] for payload in journal_payloads] == ["prepared", "rolled_back"]
    assert "interrupted_at_index" not in journal_payloads[1]
    # No target was replaced.
    for i, (target, _) in enumerate(targets):
        assert target.read_text() == f"old_{i}", (
            f"Target {i} should be unchanged after rollback, got {target.read_text()!r}"
        )


def test_journal_json_serialization_is_stable_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    payload = {"z": "München", "a": {"phase": "prepared"}}

    transaction_module.atomic_write_journal_json(path, payload)

    assert path.read_bytes() == (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def test_file_content_hash_uses_an_empty_sentinel_for_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    existing = tmp_path / "existing.txt"
    existing.write_bytes(b"content")

    assert transaction_module.file_content_hash(missing) == ""
    assert transaction_module.file_content_hash(existing) == _sha256(existing)


def test_ordered_replacements_rejects_duplicate_targets(tmp_path: Path) -> None:
    """The helper must reject duplicate targets in the input list."""
    target, staged = _make_text_pair(tmp_path, "target.txt", "old", "new")
    with pytest.raises(ValueError) as error:
        link_migration.apply_link_migration(
            tmp_path,
            replacements=[(target, staged), (target, staged)],
        )
    assert str(error.value) == "Link migration transaction contains duplicate targets"


def test_first_replacement_failure_rolls_back_and_marks_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, staged = _make_text_pair(tmp_path, "target.txt", "old", "new")
    journal_payloads = _capture_journal_writes(monkeypatch)

    def fail_after_first_write(_index: int, _target: Path) -> None:
        raise RuntimeError("first replacement interrupted")

    with pytest.raises(RuntimeError, match="first replacement interrupted"):
        link_migration.apply_link_migration(
            tmp_path,
            replacements=[(target, staged)],
            _crash_hook=fail_after_first_write,
        )

    assert target.read_text() == "old"
    assert [payload["phase"] for payload in journal_payloads] == ["prepared", "rolled_back"]
    assert not (tmp_path / ".link_migration_journal" / "journal.json").exists()


def test_recovery_with_missing_entries_defaults_to_empty_list(tmp_path: Path) -> None:
    journal_dir = tmp_path / "transaction"
    journal_dir.mkdir()
    journal = journal_dir / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "contract_version": transaction_module.TRANSACTION_VERSION,
                "stem": "alpha",
                "phase": "prepared",
            }
        ),
        encoding="utf-8",
    )

    transaction_module._recover_directory(journal_dir, "alpha", data_root=tmp_path)

    assert not journal.exists()


def test_recovery_journal_reader_requests_utf8_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "contract_version": transaction_module.TRANSACTION_VERSION,
                "stem": "alpha",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        encodings.append(encoding)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert (
        transaction_module._load_recovery_journal(journal, "alpha", data_root=tmp_path)["entries"]
        == []
    )
    assert len(encodings) == 1
    assert encodings[0] is not None
    assert codecs.lookup(encodings[0]).name == "utf-8"


def test_ordered_replacements_idempotent_second_run_preserves_mtime_and_hash(
    tmp_path: Path,
) -> None:
    """A second run on the same stem must preserve target mtime and hash."""
    target, staged = _make_text_pair(tmp_path, "target.txt", "new", "new")  # same content
    link_migration.apply_link_migration(tmp_path, replacements=[(target, staged)])
    first_hash = _sha256(target)
    first_mtime = target.stat().st_mtime_ns
    # Wait at least 1 microsecond so that a re-write would bump mtime.
    os.utime(target, ns=(first_mtime + 1_000, first_mtime + 1_000))
    link_migration.apply_link_migration(tmp_path, replacements=[(target, staged)])
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
    target, staged = _make_text_pair(tmp_path, "target.txt", "old", "new")

    # Inject a hook that corrupts the target AFTER the helper writes it.
    def corrupt_after_replace(index: int, target_path: Path) -> None:
        if index == 0:
            target_path.write_text("CORRUPTED")

    with pytest.raises(RuntimeError, match="post-hook hash mismatch"):
        link_migration.apply_link_migration(
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
    target = tmp_path / "target.txt"
    target.write_text("old")
    missing_staged = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="Staged link migration file is missing"):
        link_migration.apply_link_migration(
            tmp_path,
            replacements=[(target, missing_staged)],
        )
    assert target.read_text() == "old"


def test_new_targets_have_no_backup_in_the_prepared_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "new-target.txt"
    staged = tmp_path / "new-target.staged"
    staged.write_text("new", encoding="utf-8")
    journal_payloads = _capture_journal_writes(monkeypatch)

    link_migration.apply_link_migration(tmp_path, replacements=[(target, staged)])

    prepared = journal_payloads[0]
    assert prepared["phase"] == "prepared"
    assert prepared["entries"] == [
        {
            "target": str(target),
            "staged": str(staged),
            "backup": "",
            "existed": False,
            "original_hash": "",
            "staged_hash": _sha256(target),
        }
    ]
    assert journal_payloads[-1]["phase"] == "committed"


def test_recovery_journal_rejects_wrong_version_and_stem(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({"contract_version": "wrong", "stem": "expected"}))

    with pytest.raises(RuntimeError, match="Invalid link migration journal"):
        transaction_module._load_recovery_journal(journal, "expected", data_root=tmp_path)

    journal.write_text(
        json.dumps(
            {
                "contract_version": transaction_module.TRANSACTION_VERSION,
                "stem": "actual",
            }
        )
    )
    with pytest.raises(
        RuntimeError,
        match=r"Link migration journal stem mismatch: 'actual' vs 'expected'",
    ):
        transaction_module._load_recovery_journal(journal, "expected", data_root=tmp_path)


def test_recovery_journal_rejects_malformed_json_and_loads_a_matching_journal(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.json"
    journal.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        transaction_module._load_recovery_journal(journal, "expected", data_root=tmp_path)

    matching = {"contract_version": transaction_module.TRANSACTION_VERSION, "stem": "expected"}
    journal.write_text(json.dumps(matching), encoding="utf-8")
    loaded = transaction_module._load_recovery_journal(journal, "expected", data_root=tmp_path)
    assert loaded["stem"] == "expected"


def test_cleanup_keeps_nonempty_nested_directories_without_raising(tmp_path: Path) -> None:
    transaction_dir = tmp_path / "transaction"
    nested = transaction_dir / "nested"
    nested.mkdir(parents=True)

    transaction_module._cleanup(transaction_dir, tmp_path)

    assert nested.is_dir()
    assert transaction_dir.is_dir()


def test_recovery_creates_missing_target_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "new" / "nested" / "target.txt"
    staged = tmp_path / "staged.txt"
    staged.write_text("new", encoding="utf-8")

    transaction_module._recover_entry(
        {
            "target": str(target),
            "staged": str(staged),
            "staged_hash": _sha256(staged),
            "backup": "",
            "existed": False,
        },
        data_root=tmp_path,
    )

    assert target.read_text(encoding="utf-8") == "new"


def test_move_recovery_staged_moves_the_staged_file_into_a_missing_target(tmp_path: Path) -> None:
    staged = tmp_path / "staged.txt"
    target = tmp_path / "target.txt"
    staged.write_text("staged-payload", encoding="utf-8")

    transaction_module._move_recovery_staged(staged, target, data_root=tmp_path)

    assert target.read_text(encoding="utf-8") == "staged-payload"
    assert not staged.exists()


# ---------------------------------------------------------------------------
# augmentation_is_current contract (valid minimal DataRoot)
# ---------------------------------------------------------------------------


def test_partial_commit_never_marks_stem_current(tmp_path: Path) -> None:
    """A successful apply must mark the stem ``augmentation_is_current``;
    a crashed/aborted apply must NOT."""
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
    link_migration.apply_link_migration(data_root.processed, stems={stem})
    assert augmentation_is_current(data_root, stem) is True

    # Crash before apply (with a single replacement) must NOT mark current.
    target = tmp_path / "throwaway.txt"
    staged = tmp_path / "throwaway.staged"
    target.write_text("a")
    staged.write_text("b")
    with pytest.raises(RuntimeError):
        link_migration.apply_link_migration(
            data_root.processed,
            replacements=[(target, staged)],
            _crash_hook=lambda *_: (_ for _ in ()).throw(RuntimeError("crash")),
        )
    assert augmentation_is_current(data_root, stem) is True, (
        "A failed apply on unrelated targets must NOT roll back the prior stem's current marker"
    )
