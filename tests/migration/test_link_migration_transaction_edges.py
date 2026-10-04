from __future__ import annotations

from collections.abc import Callable
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


def _assert_path_escape(role: str, operation: Callable[[], object]) -> None:
    with pytest.raises(
        RuntimeError,
        match=rf"^Link migration journal {role} path escapes the data root:",
    ):
        operation()


def test_commit_rejects_external_journal_directory_and_journal_symlink(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    _assert_path_escape(
        "journal directory",
        lambda: transaction_module.commit_ordered_replacements(
            outside / "transaction",
            "alpha",
            [(target, staged)],
            data_root=data_root,
        ),
    )

    journal_dir = data_root / "transaction"
    journal_dir.mkdir()
    outside_journal = outside / "journal.json"
    outside_journal.write_text("outside-canary", encoding="utf-8")
    (journal_dir / "journal.json").symlink_to(outside_journal)
    _assert_path_escape(
        "journal",
        lambda: transaction_module.commit_ordered_replacements(
            journal_dir,
            "alpha",
            [(target, staged)],
            data_root=data_root,
        ),
    )
    assert outside_journal.read_text(encoding="utf-8") == "outside-canary"


@pytest.mark.parametrize("unsafe_path", ["journal directory", "journal"])
def test_recovery_reports_the_escaping_journal_path_role(tmp_path: Path, unsafe_path: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    if unsafe_path == "journal directory":
        journal_directory = outside / "transaction"
        journal_directory.mkdir()
        canary = journal_directory / "journal.json"
        canary.write_text("outside-canary", encoding="utf-8")
    else:
        journal_directory = data_root / "transaction"
        journal_directory.mkdir()
        canary = outside / "journal.json"
        canary.write_text("outside-canary", encoding="utf-8")
        (journal_directory / "journal.json").symlink_to(canary)

    _assert_path_escape(
        unsafe_path,
        lambda: transaction_module._recover_directory(
            journal_directory, "alpha", data_root=data_root
        ),
    )
    assert canary.read_text(encoding="utf-8") == "outside-canary"


@pytest.mark.parametrize("role", ["target", "staged"])
def test_replacement_validation_identifies_each_external_path(tmp_path: Path, role: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    inside = data_root / "inside.bin"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_text("outside", encoding="utf-8")
    replacement = (outside, inside) if role == "target" else (inside, outside)

    _assert_path_escape(
        role,
        lambda: transaction_module._validate_replacement_targets([replacement], data_root),
    )


def test_new_transaction_checks_journal_path_before_writing(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    journal_dir = data_root / "transaction"
    journal_dir.mkdir(parents=True)
    outside = tmp_path / "outside-journal.json"
    outside.write_text("outside-canary", encoding="utf-8")
    (journal_dir / "journal.json").symlink_to(outside)

    _assert_path_escape(
        "journal",
        lambda: transaction_module._commit_new_transaction(
            journal_dir, "alpha", [], data_root, None
        ),
    )
    assert outside.read_text(encoding="utf-8") == "outside-canary"


def test_new_transaction_checks_journal_again_after_apply(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    journal_dir = data_root / "transaction"
    journal_dir.mkdir(parents=True)
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    outside_journal = tmp_path / "outside-journal.json"
    outside_journal.write_text("outside-canary", encoding="utf-8")

    def replace_journal_with_symlink(_index: int, _target: Path) -> None:
        journal = journal_dir / "journal.json"
        journal.unlink()
        journal.symlink_to(outside_journal)

    _assert_path_escape(
        "journal",
        lambda: transaction_module._commit_new_transaction(
            journal_dir,
            "alpha",
            [(target, staged)],
            data_root,
            replace_journal_with_symlink,
        ),
    )
    assert outside_journal.read_text(encoding="utf-8") == "outside-canary"


def test_interrupted_failure_checks_journal_before_rewriting(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    journal_dir = data_root / "transaction"
    journal_dir.mkdir(parents=True)
    outside_journal = tmp_path / "outside-journal.json"
    outside_journal.write_text("outside-canary", encoding="utf-8")
    journal = journal_dir / "journal.json"
    journal.symlink_to(outside_journal)

    _assert_path_escape(
        "journal",
        lambda: transaction_module._record_transaction_failure(
            journal_dir, journal, {}, [], 1, data_root
        ),
    )
    assert outside_journal.read_text(encoding="utf-8") == "outside-canary"


@pytest.mark.parametrize("role", ["target", "staged", "backup"])
def test_entry_preparation_identifies_each_external_path(tmp_path: Path, role: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    journal_dir = data_root / "journal"

    if role == "target":
        target = outside / "target.bin"
        target.write_text("old", encoding="utf-8")
    elif role == "staged":
        staged = outside / "staged.bin"
        staged.write_text("new", encoding="utf-8")
    else:
        journal_dir.symlink_to(outside, target_is_directory=True)

    _assert_path_escape(
        role,
        lambda: transaction_module._prepare_entry(journal_dir, target, staged, data_root),
    )


def test_journal_loader_rejects_external_journal_symlink(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    journal_dir = data_root / "transaction"
    journal_dir.mkdir(parents=True)
    outside = tmp_path / "outside-journal.json"
    outside.write_text("outside-canary", encoding="utf-8")
    journal = journal_dir / "journal.json"
    journal.symlink_to(outside)

    _assert_path_escape(
        "journal",
        lambda: transaction_module._load_recovery_journal(journal, "alpha", data_root=data_root),
    )
    assert outside.read_text(encoding="utf-8") == "outside-canary"


@pytest.mark.parametrize("role", ["target", "staged"])
def test_recovery_entry_identifies_each_external_path(tmp_path: Path, role: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_text("outside", encoding="utf-8")
    paths = {"target": str(target), "staged": str(staged), "staged_hash": "unused"}
    paths[role] = str(outside)

    _assert_path_escape(role, lambda: transaction_module._recover_entry(paths, data_root))


@pytest.mark.parametrize("role", ["target", "staged"])
def test_recovery_target_preparation_identifies_each_external_path(
    tmp_path: Path, role: str
) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_text("outside", encoding="utf-8")
    entry = {"backup": "", "existed": False}
    if role == "target":
        target = outside
    else:
        staged = outside

    _assert_path_escape(
        role,
        lambda: transaction_module._prepare_recovery_target(entry, target, staged, data_root),
    )


@pytest.mark.parametrize("role", ["target", "staged"])
def test_recovery_move_identifies_each_external_path(tmp_path: Path, role: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    target = data_root / "target.bin"
    staged = data_root / "staged.bin"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_text("outside", encoding="utf-8")
    if role == "target":
        target = outside
    else:
        staged = outside

    _assert_path_escape(
        role,
        lambda: transaction_module._move_recovery_staged(staged, target, data_root),
    )


def test_cleanup_rejects_external_journal_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()

    _assert_path_escape(
        "journal directory",
        lambda: transaction_module._cleanup(outside_directory, data_root),
    )


@pytest.mark.parametrize("entry_kind", ["file", "symlink"])
def test_cleanup_entry_rejects_external_entries(tmp_path: Path, entry_kind: str) -> None:
    data_root = tmp_path / "processed"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    entry = outside / "entry.json"
    if entry_kind == "file":
        entry.write_text("outside-canary", encoding="utf-8")
    else:
        referent = tmp_path / "referent.json"
        referent.write_text("outside-canary", encoding="utf-8")
        entry.symlink_to(referent)

    _assert_path_escape(
        "cleanup entry",
        lambda: transaction_module._cleanup_entry(entry, data_root),
    )
    assert entry.exists()


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
