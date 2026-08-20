"""Direct safety contracts for the low-level recovery transaction journal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery import transaction as transaction_module
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.transaction import (
    commit_replacements,
    recover_interrupted_transactions,
    transaction_directory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_journal(
    root: Path,
    *,
    stem: str,
    phase: str,
    target: Path,
    staged: Path,
    backup: Path,
    existed: bool,
    original_hash: str,
) -> Path:
    directory = transaction_directory(root, stem)
    directory.mkdir(parents=True)
    payload = {
        "contract_version": "wikidata-recovery-transaction-v1",
        "stem": stem,
        "phase": phase,
        "entries": [
            {
                "target": str(target),
                "staged": str(staged),
                "backup": str(backup),
                "existed": existed,
                "original_hash": original_hash,
                "staged_hash": _sha256(staged),
            }
        ],
    }
    journal = directory / "journal.json"
    journal.write_text(json.dumps(payload), encoding="utf-8")
    return journal


@pytest.mark.parametrize("stem", ["", ".", "..", "nested/name", r"nested\\name"])
def test_transaction_directory_rejects_path_escape_stems(tmp_path: Path, stem: str) -> None:
    with pytest.raises(ValueError, match="Invalid recovery transaction stem"):
        transaction_directory(tmp_path, stem)


def test_recover_missing_transaction_root_is_a_noop(tmp_path: Path) -> None:
    assert recover_interrupted_transactions(tmp_path / "missing") == ()


def test_commit_replacements_with_no_files_is_a_noop(tmp_path: Path) -> None:
    directory = tmp_path / "transactions" / "empty"

    commit_replacements(directory, "empty", [])

    assert not directory.exists()


def test_commit_replacements_rejects_duplicate_targets_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    staged.write_text("new", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate targets"):
        commit_replacements(
            tmp_path / "transactions" / "duplicate",
            "duplicate",
            [(target, staged), (target, staged)],
        )

    assert not target.exists()
    assert not (tmp_path / "transactions" / "duplicate" / "journal.json").exists()


def test_commit_replacements_cleans_journal_and_staging_after_success(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    directory = tmp_path / "transactions" / "success"

    commit_replacements(directory, "success", [(target, staged)])

    assert target.read_text(encoding="utf-8") == "new"
    assert not staged.exists()
    assert not directory.exists()


def test_commit_replacements_rolls_back_when_commit_hook_fails(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    created = tmp_path / "created.txt"
    staged_existing = tmp_path / "staged-existing.txt"
    staged_created = tmp_path / "staged-created.txt"
    existing.write_text("old", encoding="utf-8")
    staged_existing.write_text("new", encoding="utf-8")
    staged_created.write_text("created", encoding="utf-8")
    directory = tmp_path / "transactions" / "rollback"

    def fail() -> None:
        raise RuntimeError("simulated crash before commit")

    with pytest.raises(RuntimeError, match="simulated crash"):
        commit_replacements(
            directory,
            "rollback",
            [(existing, staged_existing), (created, staged_created)],
            before_commit=fail,
        )

    assert existing.read_text(encoding="utf-8") == "old"
    assert not created.exists()
    assert not directory.exists()


def test_recover_prepared_journal_rolls_back_and_cleans_files(tmp_path: Path) -> None:
    root = tmp_path / "transactions"
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    backup = tmp_path / "backup.txt"
    target.write_text("new", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    backup.write_text("old", encoding="utf-8")
    journal = _write_journal(
        root,
        stem="prepared",
        phase="prepared",
        target=target,
        staged=staged,
        backup=backup,
        existed=True,
        original_hash=_sha256(backup),
    )

    assert recover_interrupted_transactions(root) == ("prepared",)
    assert target.read_text(encoding="utf-8") == "old"
    assert not journal.parent.exists()


@pytest.mark.parametrize("phase", ["committing", "committed"])
def test_recover_committing_or_committed_journal_rolls_forward(tmp_path: Path, phase: str) -> None:
    root = tmp_path / "transactions"
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    backup = tmp_path / "backup.txt"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    backup.write_text("old", encoding="utf-8")
    journal = _write_journal(
        root,
        stem=phase,
        phase=phase,
        target=target,
        staged=staged,
        backup=backup,
        existed=True,
        original_hash=_sha256(backup),
    )

    assert recover_interrupted_transactions(root) == (phase,)
    assert target.read_text(encoding="utf-8") == "new"
    assert not journal.parent.exists()


def test_recover_rejects_unknown_transaction_phase(tmp_path: Path) -> None:
    root = tmp_path / "transactions"
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    backup = tmp_path / "backup.txt"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    backup.write_text("old", encoding="utf-8")
    _write_journal(
        root,
        stem="unknown",
        phase="unknown",
        target=target,
        staged=staged,
        backup=backup,
        existed=True,
        original_hash=_sha256(backup),
    )

    with pytest.raises(RuntimeError, match="Unknown recovery transaction phase"):
        recover_interrupted_transactions(root)


def test_rollback_removes_target_created_by_transaction(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    target.write_text("created", encoding="utf-8")

    transaction_module._rollback_entry(
        {
            "target": str(target),
            "backup": str(tmp_path / "missing.backup"),
            "existed": False,
            "original_hash": "",
        }
    )

    assert not target.exists()


def test_rollback_rejects_missing_backup_for_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("current", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backup is unavailable"):
        transaction_module._rollback_entry(
            {
                "target": str(target),
                "backup": str(tmp_path / "missing.backup"),
                "existed": True,
                "original_hash": "unused",
            }
        )


def test_rollback_rejects_hash_mismatch_after_restore(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    backup = tmp_path / "backup.txt"
    target.write_text("current", encoding="utf-8")
    backup.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="rollback verification failed"):
        transaction_module._rollback_entry(
            {
                "target": str(target),
                "backup": str(backup),
                "existed": True,
                "original_hash": "wrong-hash",
            }
        )

    assert target.read_text(encoding="utf-8") == "original"


def test_roll_forward_skips_target_with_matching_staged_hash(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    target.write_text("new", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")

    transaction_module._roll_forward_entry(
        {
            "target": str(target),
            "staged": str(staged),
            "staged_hash": _sha256(staged),
        }
    )

    assert target.read_text(encoding="utf-8") == "new"


def test_roll_forward_rejects_changed_staged_file(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.txt"
    staged.write_text("new", encoding="utf-8")

    with pytest.raises(RuntimeError, match="staged file is unavailable"):
        transaction_module._roll_forward_entry(
            {
                "target": str(target),
                "staged": str(staged),
                "staged_hash": "wrong-hash",
            }
        )


def test_journal_entries_require_a_list_of_mappings(tmp_path: Path) -> None:
    journal = tmp_path / "journal.json"

    with pytest.raises(RuntimeError, match="Invalid recovery transaction entries"):
        transaction_module._validate_journal_entries({"entry": "not-a-list"}, journal)

    with pytest.raises(RuntimeError, match="Invalid recovery transaction entries"):
        transaction_module._validate_journal_entries(["not-a-mapping"], journal)
