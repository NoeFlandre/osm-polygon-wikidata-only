from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline._link_migration import transaction as transaction_module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    target: Path,
    staged: Path,
    backup: Path | None,
    *,
    existed: bool,
    original_hash: str = "",
) -> transaction_module._TransactionEntry:
    return transaction_module._TransactionEntry(
        target=target,
        staged=staged,
        backup=backup,
        existed=existed,
        original_hash=original_hash,
        staged_hash="",
    )


def test_recovery_backup_is_removed_only_for_existing_targets(tmp_path: Path) -> None:
    backup = tmp_path / "target.backup"
    backup.write_text("old", encoding="utf-8")

    transaction_module._remove_recovery_backup({"backup": str(backup), "existed": True})
    assert not backup.exists()

    backup.write_text("old", encoding="utf-8")
    transaction_module._remove_recovery_backup({"backup": str(backup), "existed": False})
    assert backup.exists()


def test_rollback_entry_removes_new_target(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    target.write_text("new", encoding="utf-8")
    entry = _entry(target, tmp_path / "staged.txt", None, existed=False)

    transaction_module._rollback_entry(entry)

    assert not target.exists()


def test_restore_existing_entry_rejects_missing_backup(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("current", encoding="utf-8")
    entry = _entry(
        target,
        tmp_path / "staged.txt",
        tmp_path / "missing.backup",
        existed=True,
    )

    with pytest.raises(RuntimeError, match="backup is unavailable"):
        transaction_module._restore_existing_entry(entry)


def test_restore_existing_entry_verifies_restored_hash(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    backup = tmp_path / "target.backup"
    target.write_text("current", encoding="utf-8")
    backup.write_text("old", encoding="utf-8")
    entry = _entry(
        target,
        tmp_path / "staged.txt",
        backup,
        existed=True,
        original_hash="wrong-hash",
    )

    with pytest.raises(RuntimeError, match="rollback verification failed"):
        transaction_module._restore_existing_entry(entry)

    assert target.read_text(encoding="utf-8") == "old"
    assert _sha256(target) != entry.original_hash
