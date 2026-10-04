from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.pipeline._link_migration import transaction as transaction_module
from tests.helpers import sha256_file as _sha256


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

    transaction_module._remove_recovery_backup(
        {"backup": str(backup), "existed": True}, data_root=tmp_path
    )
    assert not backup.exists()

    backup.write_text("old", encoding="utf-8")
    transaction_module._remove_recovery_backup(
        {"backup": str(backup), "existed": False}, data_root=tmp_path
    )
    assert backup.exists()


def test_recovery_without_a_backup_path_is_a_noop(tmp_path: Path) -> None:
    transaction_module._remove_recovery_backup({"existed": False}, data_root=tmp_path)


def test_recovery_entry_rejects_an_unresolvable_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    target = root / "target.txt"
    original_resolve = Path.resolve

    def fail_for_root(path: Path) -> Path:
        if path == root:
            raise OSError("parent cannot be resolved")
        return original_resolve(path)

    monkeypatch.setattr(Path, "resolve", fail_for_root)

    with pytest.raises(RuntimeError, match="Invalid link migration journal target path"):
        transaction_module._ensure_recovery_entry_within_root(target, "target", root)


def test_cleanup_preserves_symlinked_directory_entries(tmp_path: Path) -> None:
    transaction_dir = tmp_path / "txn"
    transaction_dir.mkdir()
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    target_canary = target_dir / "canary.txt"
    target_canary.write_text("canary", encoding="utf-8")
    nested_alias = transaction_dir / "nested-alias"
    nested_alias.symlink_to(target_dir, target_is_directory=True)

    transaction_module._cleanup(transaction_dir, tmp_path)

    assert nested_alias.is_symlink()
    assert target_canary.read_text(encoding="utf-8") == "canary"
    assert transaction_dir.is_dir()


def test_cleanup_unlinks_file_symlinks_without_touching_referents(tmp_path: Path) -> None:
    transaction_dir = tmp_path / "txn"
    transaction_dir.mkdir()
    referent = tmp_path / "referent.txt"
    referent.write_text("canary", encoding="utf-8")
    file_alias = transaction_dir / "journal-alias.json"
    file_alias.symlink_to(referent)

    transaction_module._cleanup(transaction_dir, tmp_path)

    assert not file_alias.exists()
    assert referent.read_text(encoding="utf-8") == "canary"
    assert not transaction_dir.exists()


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
