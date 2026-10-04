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
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline._link_migration import transaction
from tests.migration._builders import seed_processed_migration_stem


def _fresh_process(tmp_path: Path) -> DataRoot:
    """Simulate a fresh process restart by constructing a new
    BackgroundUploadQueue / apply context from disk state."""
    return DataRoot(tmp_path)


# ---------------------------------------------------------------------------
# 1. Crash after link parquet commit but before manifests -> roll-forward
# ---------------------------------------------------------------------------


def test_crash_after_link_parquet_rollforward_completes(tmp_path: Path) -> None:
    """A crash between the link parquet commit and the manifest
    updates must allow a fresh restart to complete the transaction
    (link parquet, processed manifest, augmentation manifest,
    pending intent, metadata marker).
    """
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    seed_processed_migration_stem(processed, stem, region="r")

    # The public crash hook fires immediately after the link parquet
    # replacement, before any manifest replacement.
    crash_after_first = {"raised": False}

    def _crash_hook(index: int, _target: Path) -> None:
        if index == 0 and not crash_after_first["raised"]:
            crash_after_first["raised"] = True
            raise RuntimeError("simulated crash after link parquet")

    with pytest.raises(RuntimeError, match="simulated crash"):
        link_migration.apply_link_migration(
            processed,
            stems={stem},
            _crash_hook=_crash_hook,
        )

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


def test_crash_before_any_commit_does_not_mark_current(tmp_path: Path) -> None:
    """A crash BEFORE any write must leave the stem un-migrated and
    not-current. A fresh apply then completes the transaction.
    """
    from osm_polygon_wikidata_only.augmentation.orchestrator import (
        augmentation_is_current,
    )

    processed = tmp_path / "processed"
    stem = "alpha-latest"
    seed_processed_migration_stem(processed, stem, region="r")

    def _always_crash(_index: int, _target: Path) -> None:
        raise RuntimeError("simulated pre-commit crash")

    with pytest.raises(RuntimeError, match="simulated pre-commit crash"):
        link_migration.apply_link_migration(
            processed,
            stems={stem},
            _crash_hook=_always_crash,
        )

    data_root = _fresh_process(tmp_path)
    assert augmentation_is_current(data_root, stem) is False

    # Fresh apply converges (using the now-restored real function).
    link_migration.apply_link_migration(processed, stems={stem})
    assert augmentation_is_current(data_root, stem) is True


# ---------------------------------------------------------------------------
# 3. Journal paths are confined to expected roots
# ---------------------------------------------------------------------------


def _create_interrupted_path_test_journal(
    tmp_path: Path, stem: str = "alpha-latest"
) -> tuple[Path, Path, dict[str, Any]]:
    """Create a real interrupted journal with two entries still to replay."""
    processed = tmp_path / "processed"
    target_dir = processed / "targets"
    staged_dir = processed / "staging"
    target_dir.mkdir(parents=True)
    staged_dir.mkdir()
    replacements = []
    for index in range(4):
        target = target_dir / f"target_{index}.bin"
        staged = staged_dir / f"target_{index}.bin"
        target.write_text(f"original-{index}", encoding="utf-8")
        staged.write_text(f"staged-{index}", encoding="utf-8")
        replacements.append((target, staged))

    journal_dir = processed / ".link_migration_journal" / stem

    def crash_after_second(index: int, _target: Path) -> None:
        if index == 1:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        transaction.commit_ordered_replacements(
            journal_dir,
            stem,
            replacements,
            data_root=processed,
            _crash_hook=crash_after_second,
        )

    journal_path = journal_dir / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["phase"] == "interrupted"
    assert replacements[0][0].read_text(encoding="utf-8") == "staged-0"
    assert replacements[1][0].read_text(encoding="utf-8") == "staged-1"
    assert replacements[2][0].read_text(encoding="utf-8") == "original-2"
    assert replacements[3][0].read_text(encoding="utf-8") == "original-3"
    return processed, journal_path, journal


@pytest.mark.parametrize("unsafe_field", ["target", "staged", "backup"])
def test_recovery_rejects_unsafe_journal_paths_before_writes(
    tmp_path: Path, unsafe_field: str
) -> None:
    """Reject sibling, prefix, and symlink escapes before replay mutates state."""
    processed, journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    unsafe_entry = journal["entries"][3]

    if unsafe_field == "target":
        escaped_path = tmp_path / "outside-targets" / "target_3.bin"
        escaped_path.parent.mkdir()
        escaped_path.write_text("target-canary", encoding="utf-8")
        unsafe_entry["target"] = str(escaped_path)
        canaries = [(escaped_path, "target-canary")]
    elif unsafe_field == "staged":
        processed_other = tmp_path / "processed-other"
        escaped_path = processed_other / "staging" / "target_3.bin"
        escaped_path.parent.mkdir(parents=True)
        escaped_path.write_text("staged-3", encoding="utf-8")
        canary = processed_other / "independent-canary.txt"
        canary.write_text("processed-other-canary", encoding="utf-8")
        assert str(escaped_path).startswith(str(processed))
        unsafe_entry["staged"] = str(escaped_path)
        canaries = [(escaped_path, "staged-3"), (canary, "processed-other-canary")]
    else:
        outside_backup_dir = tmp_path / "outside-backups"
        outside_backup_dir.mkdir()
        escaped_path = outside_backup_dir / "target_3.bin.backup"
        escaped_path.write_text("backup-canary", encoding="utf-8")
        backup_link = processed / "backup-link"
        backup_link.symlink_to(outside_backup_dir, target_is_directory=True)
        unsafe_entry["backup"] = str(backup_link / escaped_path.name)
        canaries = [(escaped_path, "backup-canary")]

    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    untouched_target = processed / "targets" / "target_2.bin"

    with pytest.raises(
        RuntimeError,
        match=rf"^Link migration journal {unsafe_field} path escapes the data root:",
    ):
        transaction._recover_directory(journal_path.parent, "alpha-latest", data_root=processed)

    # Validate the complete entry set before any earlier safe entry can roll forward.
    assert untouched_target.read_text(encoding="utf-8") == "original-2"
    for canary, expected in canaries:
        assert canary.read_text(encoding="utf-8") == expected
    assert journal_path.is_file()


@pytest.mark.parametrize("field", ["target", "staged", "backup"])
def test_recovery_allows_symlink_aliases_confined_to_data_root(tmp_path: Path, field: str) -> None:
    """Keep in-root aliases valid when both their entry and referent stay rooted."""
    processed, _journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    entry = journal["entries"][3]
    original_path = Path(entry[field])
    aliases = processed / "aliases"
    aliases.mkdir()
    alias = aliases / f"{field}-alias.bin"
    alias.symlink_to(original_path.resolve())
    entry[field] = str(alias)

    transaction._validate_recovery_journal_paths(journal, processed.resolve())

    assert alias.is_symlink()
    assert alias.resolve().is_relative_to(processed.resolve())


@pytest.mark.parametrize("unsafe_field", ["target", "staged", "backup"])
def test_recovery_rejects_outside_symlink_aliases_before_writes(
    tmp_path: Path, unsafe_field: str
) -> None:
    """Validate both the symlink entry location and the path it resolves to."""
    processed, journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    unsafe_entry = journal["entries"][3]
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    alias = outside_dir / f"{unsafe_field}-alias.bin"

    if unsafe_field == "target":
        resolved_inside_path = processed / "targets" / "target_3.bin"
        untouched_inside_value = "original-3"
    elif unsafe_field == "staged":
        resolved_inside_path = processed / "staging" / "target_3.bin"
        untouched_inside_value = "staged-3"
    else:
        resolved_inside_path = Path(unsafe_entry["backup"])
        untouched_inside_value = "original-3"

    alias.symlink_to(resolved_inside_path)
    unsafe_entry[unsafe_field] = str(alias)
    outside_canary = outside_dir / "independent-canary.txt"
    outside_canary.write_text("outside-canary", encoding="utf-8")
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    untouched_target = processed / "targets" / "target_2.bin"

    recovery_error: RuntimeError | None = None
    try:
        transaction._recover_directory(journal_path.parent, "alpha-latest", data_root=processed)
    except RuntimeError as error:
        recovery_error = error

    safe_before_replay = (
        untouched_target.read_text(encoding="utf-8") == "original-2"
        and alias.is_symlink()
        and alias.resolve() == resolved_inside_path.resolve()
        and outside_canary.read_text(encoding="utf-8") == "outside-canary"
        and journal_path.is_file()
    )
    assert recovery_error is not None and safe_before_replay, (
        f"recovery_error={recovery_error!r}; alias_is_symlink={alias.is_symlink()}; "
        f"alias_exists={alias.exists()}; alias_target={alias.resolve() if alias.exists() else None}; "
        f"inside_value={resolved_inside_path.read_text(encoding='utf-8') if resolved_inside_path.is_file() else None!r}; "
        f"expected_inside_value={untouched_inside_value!r}; "
        f"earlier_safe_target={untouched_target.read_text(encoding='utf-8')!r}; "
        f"journal_exists={journal_path.is_file()}"
    )


@pytest.mark.parametrize("unsafe_field", ["target", "staged", "backup"])
def test_recovery_rejects_in_root_symlinks_to_outside_before_writes(
    tmp_path: Path, unsafe_field: str
) -> None:
    """A rooted directory entry must not dereference to a path outside the root."""
    processed, journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    entry = journal["entries"][3]
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_path = outside_dir / f"{unsafe_field}-target.bin"
    outside_contents = {
        "target": "original-3",
        "staged": "staged-3",
        "backup": "original-3",
    }[unsafe_field]
    outside_path.write_text(outside_contents, encoding="utf-8")

    alias = Path(entry[unsafe_field])
    if alias.exists():
        alias.unlink()
    alias.symlink_to(outside_path)
    outside_canary = outside_dir / "independent-canary.txt"
    outside_canary.write_text("outside-canary", encoding="utf-8")
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    untouched_target = processed / "targets" / "target_2.bin"

    with pytest.raises(
        RuntimeError,
        match=rf"^Link migration journal {unsafe_field} path escapes the data root:",
    ):
        transaction._recover_directory(journal_path.parent, "alpha-latest", data_root=processed)

    assert (
        alias.is_symlink()
        and alias.resolve() == outside_path.resolve()
        and outside_path.read_text(encoding="utf-8") == outside_contents
        and outside_canary.read_text(encoding="utf-8") == "outside-canary"
        and untouched_target.read_text(encoding="utf-8") == "original-2"
        and journal_path.is_file()
    ), (
        f"alias_is_symlink={alias.is_symlink()}; "
        f"outside_value={outside_path.read_text(encoding='utf-8')!r}; "
        f"earlier_safe_target={untouched_target.read_text(encoding='utf-8')!r}; "
        f"journal_exists={journal_path.is_file()}"
    )


def test_recovery_rejects_symlinked_journal_directory_before_cleanup(
    tmp_path: Path,
) -> None:
    """A journal directory symlink must not make cleanup unlink outside files."""
    processed = tmp_path / "processed"
    journal_parent = processed / ".link_migration_journal"
    journal_parent.mkdir(parents=True)
    outside_journal_dir = tmp_path / "outside-journal"
    outside_journal_dir.mkdir()
    journal_dir = journal_parent / "alpha-latest"
    journal_dir.symlink_to(outside_journal_dir, target_is_directory=True)
    journal_path = outside_journal_dir / "journal.json"
    journal = {
        "contract_version": transaction.TRANSACTION_VERSION,
        "stem": "alpha-latest",
        "phase": "interrupted",
        "entries": [],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    outside_canary = outside_journal_dir / "outside-canary.txt"
    outside_canary.write_text("outside-canary", encoding="utf-8")

    recovery_error: RuntimeError | None = None
    try:
        transaction._recover_directory(journal_dir, "alpha-latest", data_root=processed)
    except RuntimeError as error:
        recovery_error = error

    assert (
        recovery_error is not None
        and journal_dir.is_symlink()
        and (
            journal_path.read_text(encoding="utf-8") == json.dumps(journal)
            and outside_canary.read_text(encoding="utf-8") == "outside-canary"
        )
    ), (
        f"recovery_error={recovery_error!r}; journal_dir_is_symlink={journal_dir.is_symlink()}; "
        f"outside_journal_exists={journal_path.exists()}; "
        f"outside_canary_exists={outside_canary.exists()}"
    )


def test_recovery_rejects_journal_directory_symlink_to_data_before_cleanup(
    tmp_path: Path,
) -> None:
    """Cleanup must not erase ordinary files through an in-root directory alias."""
    processed = tmp_path / "processed"
    data_dir = processed / "targets"
    data_dir.mkdir(parents=True)
    journal_parent = processed / ".link_migration_journal"
    journal_parent.mkdir()
    journal_dir = journal_parent / "alpha-latest"
    journal_dir.symlink_to(data_dir, target_is_directory=True)
    journal_path = data_dir / "journal.json"
    journal = {
        "contract_version": transaction.TRANSACTION_VERSION,
        "stem": "alpha-latest",
        "phase": "interrupted",
        "entries": [],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    data_canary = data_dir / "target-canary.txt"
    data_canary.write_text("target-canary", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=r"^Link migration journal directory cannot contain symlinks:",
    ):
        transaction._recover_directory(journal_dir, "alpha-latest", data_root=processed)

    assert (
        journal_dir.is_symlink()
        and journal_path.read_text(encoding="utf-8") == json.dumps(journal)
        and data_canary.read_text(encoding="utf-8") == "target-canary"
    ), (
        f"journal_dir_is_symlink={journal_dir.is_symlink()}; "
        f"journal_exists={journal_path.exists()}; data_canary_exists={data_canary.exists()}"
    )


def test_recovery_accepts_stem_named_like_journal_directory(tmp_path: Path) -> None:
    """A valid stem must not be confused with a journal-root path component."""
    stem = ".link_migration_journal"
    processed, journal_path, _journal = _create_interrupted_path_test_journal(tmp_path, stem)

    transaction._recover_directory(journal_path.parent, stem, data_root=processed)

    assert not journal_path.exists()
    assert (processed / "targets" / "target_2.bin").read_text(encoding="utf-8") == "staged-2"
    assert (processed / "targets" / "target_3.bin").read_text(encoding="utf-8") == "staged-3"


@pytest.mark.parametrize("malformation", ["entries", "entry"])
def test_recovery_rejects_malformed_journal_before_writes(
    tmp_path: Path, malformation: str
) -> None:
    """Reject malformed journal structure before any pending target is changed."""
    processed, journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    if malformation == "entries":
        journal["entries"] = None
    else:
        journal["entries"][3] = None
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    untouched_target = processed / "targets" / "target_2.bin"

    with pytest.raises(RuntimeError, match=r"^Invalid link migration journal (entries|entry)$"):
        transaction._recover_directory(journal_path.parent, "alpha-latest", data_root=processed)

    assert untouched_target.read_text(encoding="utf-8") == "original-2"
    assert journal_path.is_file()


def test_recovery_accepts_empty_optional_backup_path(tmp_path: Path) -> None:
    """A missing backup is valid for an entry without a preexisting target."""
    processed, _journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    journal["entries"][3]["backup"] = ""

    transaction._validate_recovery_journal_paths(journal, processed.resolve())


@pytest.mark.parametrize("target_value", [None, 17], ids=["missing", "non-string"])
def test_recovery_rejects_invalid_required_path_before_writes(
    tmp_path: Path, target_value: Any
) -> None:
    """Reject malformed required targets before replaying earlier pending entries."""
    processed, journal_path, journal = _create_interrupted_path_test_journal(tmp_path)
    journal["entries"][3]["target"] = target_value
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    untouched_target = processed / "targets" / "target_2.bin"

    with pytest.raises(RuntimeError, match="Invalid link migration journal target path"):
        transaction._recover_directory(journal_path.parent, "alpha-latest", data_root=processed)

    assert untouched_target.read_text(encoding="utf-8") == "original-2"
    assert journal_path.is_file()


def test_manifest_failure_cannot_leave_canonical_link_without_manifest_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every durable state change belongs to the same recoverable transaction."""
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    seed_processed_migration_stem(processed, stem, region="r")

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
) -> None:
    processed = tmp_path / "processed"
    stem = "alpha-latest"
    seed_processed_migration_stem(processed, stem, region="r")
    captured: list[Path] = []

    def capture(_index: int, target: Path) -> None:
        captured.append(target)

    link_migration.apply_link_migration(processed, stems={stem}, _crash_hook=capture)

    relative = {path.relative_to(processed).as_posix() for path in captured}
    assert f"polygon_articles/{stem}.parquet" in relative
    assert "manifests/processed_pbfs.json" in relative
    assert "augmentation/manifests/augmentation_manifest.json" in relative
    assert "manifests/pending_migration_publications.json" in relative
    assert "integrity/rejection_ledger.json" in relative
