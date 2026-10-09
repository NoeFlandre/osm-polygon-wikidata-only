"""Ordered, journaled file replacement for link migration."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file_uncached

TRANSACTION_VERSION = "link-migration-transaction-v1"


@dataclass
class _TransactionEntry:
    target: Path
    staged: Path
    backup: Path | None
    existed: bool
    original_hash: str
    staged_hash: str


def commit_ordered_replacements(
    directory: Path,
    stem: str,
    replacements: list[tuple[Path, Path]],
    *,
    data_root: Path,
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Replace data before manifests and recover interrupted commits."""
    if not replacements:
        return
    root = data_root.resolve()
    _ensure_recovery_path_within_root(str(directory), "journal directory", root)
    _ensure_recovery_journal_directory_has_no_symlinks(directory, data_root)
    _validate_replacement_targets(replacements, root)
    ordered = sorted(
        replacements,
        key=lambda item: (1 if item[0].suffix == ".json" else 0, str(item[0])),
    )

    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "journal.json"
    _ensure_recovery_path_within_root(str(journal_path), "journal", root)
    if journal_path.exists():
        # Keep the owner's original spelling for the lexical symlink check.
        _recover_directory(directory, stem, data_root=data_root)
        return

    _commit_new_transaction(directory, stem, ordered, data_root, _crash_hook)


def _validate_replacement_targets(replacements: list[tuple[Path, Path]], data_root: Path) -> None:
    """Reject duplicate targets or paths outside the owning processed-data root."""
    targets = [target for target, _ in replacements]
    if len(set(targets)) != len(targets):
        raise ValueError("Link migration transaction contains duplicate targets")
    for target, staged in replacements:
        _ensure_recovery_path_within_root(str(target), "target", data_root)
        _ensure_recovery_path_within_root(str(staged), "staged", data_root)


def _commit_new_transaction(
    directory: Path,
    stem: str,
    ordered: list[tuple[Path, Path]],
    data_root: Path,
    crash_hook: Callable[[int, Path], None] | None,
) -> None:
    """Prepare, apply, and clean a new ordered replacement transaction."""
    _ensure_recovery_journal_directory_has_no_symlinks(directory, data_root)
    journal_path = directory / "journal.json"
    _ensure_recovery_path_within_root(str(journal_path), "journal", data_root)

    entries = [_prepare_entry(directory, target, staged, data_root) for target, staged in ordered]
    journal: dict[str, Any] = {
        "contract_version": TRANSACTION_VERSION,
        "stem": stem,
        "phase": "prepared",
        "entries": [
            {
                "target": str(entry.target),
                "staged": str(entry.staged),
                "backup": str(entry.backup) if entry.backup else "",
                "existed": entry.existed,
                "original_hash": entry.original_hash,
                "staged_hash": entry.staged_hash,
            }
            for entry in entries
        ],
    }
    atomic_write_journal_json(journal_path, journal)

    index_ref = [0]
    try:
        _apply_entries(entries, crash_hook, index_ref)
        journal["phase"] = "committed"
        _ensure_recovery_path_within_root(str(journal_path), "journal", data_root)
        atomic_write_journal_json(journal_path, journal)
    except BaseException:
        _record_transaction_failure(
            directory, journal_path, journal, entries, index_ref[0], data_root
        )
        raise
    _cleanup(directory, data_root)


def _record_transaction_failure(
    directory: Path,
    journal_path: Path,
    journal: dict[str, Any],
    entries: list[_TransactionEntry],
    index: int,
    data_root: Path,
) -> None:
    """Record whether a failed apply can be rolled back immediately."""
    if index == 0:
        _rollback_entries(entries)
        journal["phase"] = "rolled_back"
        atomic_write_journal_json(journal_path, journal)
        _cleanup(directory, data_root)
        return
    journal["phase"] = "interrupted"
    journal["interrupted_at_index"] = int(index)
    _ensure_recovery_path_within_root(str(journal_path), "journal", data_root)
    atomic_write_journal_json(journal_path, journal)


def _apply_entries(
    entries: list[_TransactionEntry],
    crash_hook: Callable[[int, Path], None] | None,
    index_ref: list[int],
) -> None:
    """Apply ordered entries and verify each optional post-hook boundary."""
    for index, entry in enumerate(entries):
        index_ref[0] = index
        _apply_single(entry)
        if crash_hook is not None:
            crash_hook(index, entry.target)
        if file_content_hash(entry.target) != entry.staged_hash:
            raise RuntimeError(f"Link migration post-hook hash mismatch for {entry.target}")


def _prepare_entry(
    directory: Path, target: Path, staged: Path, data_root: Path
) -> _TransactionEntry:
    _ensure_recovery_path_within_root(str(target), "target", data_root)
    _ensure_recovery_path_within_root(str(staged), "staged", data_root)
    if not staged.is_file():
        raise FileNotFoundError(f"Staged link migration file is missing: {staged}")
    backup = directory / f"{target.name}.backup"
    _ensure_recovery_path_within_root(str(backup), "backup", data_root)
    existed = target.is_file()
    original_hash = ""
    if existed:
        shutil.copyfile(target, backup)
        original_hash = file_content_hash(target)
    return _TransactionEntry(
        target=target,
        staged=staged,
        backup=backup if existed else None,
        existed=existed,
        original_hash=original_hash,
        staged_hash=file_content_hash(staged),
    )


def _apply_single(entry: _TransactionEntry) -> None:
    if _file_matches_hash(entry.target, entry.staged_hash):
        return
    if not _file_matches_hash(entry.staged, entry.staged_hash):
        raise RuntimeError(f"Link migration staged file is unavailable: {entry.staged}")
    entry.target.parent.mkdir(parents=True, exist_ok=True)
    if entry.target.is_file():
        os.replace(entry.staged, entry.target)
    else:
        shutil.move(str(entry.staged), str(entry.target))
    if file_content_hash(entry.target) != entry.staged_hash:
        raise RuntimeError(f"Link migration verification failed: {entry.target}")


def _file_matches_hash(path: Path, expected_hash: str) -> bool:
    """Return whether a regular file has the expected content hash."""
    return path.is_file() and file_content_hash(path) == expected_hash


def _rollback_entries(entries: list[_TransactionEntry]) -> None:
    for entry in reversed(entries):
        _rollback_entry(entry)


def _rollback_entry(entry: _TransactionEntry) -> None:
    """Restore one target from backup or remove a newly created target."""
    if entry.existed:
        _restore_existing_entry(entry)
    elif entry.target.is_file():
        entry.target.unlink()


def _restore_existing_entry(entry: _TransactionEntry) -> None:
    """Restore and verify an entry that existed before the transaction."""
    backup = entry.backup
    if backup is None or not backup.is_file():
        raise RuntimeError(f"Link migration backup is unavailable: {backup}")
    os.replace(backup, entry.target)
    if file_content_hash(entry.target) != entry.original_hash:
        raise RuntimeError(f"Link migration rollback verification failed: {entry.target}")


def _recover_directory(directory: Path, stem: str, *, data_root: Path) -> None:
    root = data_root.resolve()
    _ensure_recovery_path_within_root(str(directory), "journal directory", root)
    _ensure_recovery_journal_directory_has_no_symlinks(directory, data_root)
    journal_path = directory / "journal.json"
    _ensure_recovery_path_within_root(str(journal_path), "journal", root)
    if not journal_path.is_file():
        return
    raw = _load_recovery_journal(journal_path, stem, data_root=root)
    _validate_recovery_journal_paths(raw, root)
    _recover_entries(raw.get("entries", []), root)
    _cleanup(directory, data_root)


def _load_recovery_journal(path: Path, stem: str, *, data_root: Path) -> dict[str, Any]:
    """Load and validate a link migration recovery journal."""
    _ensure_recovery_path_within_root(str(path), "journal", data_root.resolve())
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid link migration journal: {path}")
    if raw.get("stem") != stem:
        raise RuntimeError(f"Link migration journal stem mismatch: {raw.get('stem')!r} vs {stem!r}")
    return raw


def _validate_recovery_journal_paths(raw: dict[str, Any], data_root: Path) -> None:
    """Validate every persisted path before replay can modify any entry."""
    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("Invalid link migration journal entries")
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid link migration journal entry")
        for name in ("target", "staged", "backup"):
            _validate_recovery_path(entry, name, data_root)


def _validate_recovery_path(entry: dict[str, Any], name: str, data_root: Path) -> Path | None:
    """Reject one journal path that resolves outside the processed-data root."""
    value = entry.get(name)
    if name == "backup" and not value:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid link migration journal {name} path")
    _ensure_recovery_path_within_root(value, name, data_root)
    return Path(value)


def _ensure_recovery_path_within_root(value: str, name: str, data_root: Path) -> None:
    """Reject a path whose mutation entry or dereferenced target escapes the root."""
    candidate = Path(value)
    root = data_root.resolve()
    _ensure_recovery_entry_within_root(candidate, name, root)
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        raise RuntimeError(f"Invalid link migration journal {name} path: {value}") from None
    try:
        resolved.relative_to(root)
    except ValueError:
        raise RuntimeError(
            f"Link migration journal {name} path escapes the data root: {value}"
        ) from None


def _ensure_recovery_entry_within_root(path: Path, name: str, data_root: Path) -> None:
    """Reject the directory entry that an unlink/replace would mutate outside root."""
    try:
        entry_path = Path(os.path.normpath(path.parent.resolve() / path.name))
    except (OSError, RuntimeError):
        raise RuntimeError(f"Invalid link migration journal {name} path: {path}") from None
    try:
        entry_path.relative_to(data_root)
    except ValueError:
        raise RuntimeError(
            f"Link migration journal {name} path escapes the data root: {path}"
        ) from None


def _ensure_recovery_journal_directory_has_no_symlinks(directory: Path, data_root: Path) -> None:
    """Keep directory-wide journal cleanup away from aliased data directories."""
    lexical_root = data_root.absolute()
    lexical_directory = directory.absolute()
    resolved_root = data_root.resolve()
    resolved_directory = directory.resolve()
    try:
        lexical_relative = lexical_directory.relative_to(lexical_root)
        resolved_relative = resolved_directory.relative_to(resolved_root)
    except ValueError:
        raise RuntimeError(
            f"Link migration journal directory escapes the data root: {directory}"
        ) from None
    if not lexical_relative.parts or lexical_relative != resolved_relative:
        raise RuntimeError(f"Link migration journal directory cannot contain symlinks: {directory}")


def _recover_entries(entries: list[dict[str, Any]], data_root: Path) -> None:
    """Roll forward all entries from an interrupted migration journal."""
    for entry in entries:
        _recover_entry(entry, data_root)


def _recover_entry(entry: dict[str, Any], data_root: Path) -> None:
    """Roll forward one interrupted migration entry."""
    target = _validate_recovery_path(entry, "target", data_root)
    staged = _validate_recovery_path(entry, "staged", data_root)
    if target is None or staged is None:
        raise RuntimeError("Link migration recovery requires target and staged paths")
    staged_hash = str(entry["staged_hash"])
    if _file_matches_hash(target, staged_hash):
        return
    if not _file_matches_hash(staged, staged_hash):
        raise RuntimeError(f"Link migration recovery: staged file unavailable: {staged}")
    _prepare_recovery_target(entry, target, staged, data_root)
    if file_content_hash(target) != staged_hash:
        raise RuntimeError(f"Link migration recovery verification failed: {target}")


def _prepare_recovery_target(
    entry: dict[str, Any], target: Path, staged: Path, data_root: Path
) -> None:
    """Move one staged file into place, removing a superseded backup."""
    _ensure_recovery_path_within_root(str(target), "target", data_root)
    _ensure_recovery_path_within_root(str(staged), "staged", data_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_recovery_backup(entry, data_root)
    _move_recovery_staged(staged, target, data_root)


def _remove_recovery_backup(entry: dict[str, Any], data_root: Path) -> None:
    """Remove a superseded backup when the original target existed."""
    backup = _validate_recovery_path(entry, "backup", data_root)
    if bool(entry["existed"]) and backup is not None and backup.is_file():
        _ensure_recovery_path_within_root(str(backup), "backup", data_root)
        backup.unlink()


def _move_recovery_staged(staged: Path, target: Path, data_root: Path) -> None:
    """Replace or create a recovered target from its staged file."""
    _ensure_recovery_path_within_root(str(staged), "staged", data_root)
    _ensure_recovery_path_within_root(str(target), "target", data_root)
    if target.is_file():
        os.replace(staged, target)
    else:
        shutil.move(str(staged), str(target))


def _cleanup(directory: Path, data_root: Path) -> None:
    """Remove journal files only from its verified, root-confined directory."""
    root = data_root.resolve()
    _ensure_recovery_path_within_root(str(directory), "journal directory", root)
    _ensure_recovery_journal_directory_has_no_symlinks(directory, data_root)
    for entry in directory.iterdir():
        _cleanup_entry(entry, root)
    with suppress(OSError):
        directory.rmdir()


def _cleanup_entry(entry: Path, data_root: Path) -> None:
    """Remove a regular journal file or file symlink, and skip directories."""
    if entry.is_symlink():
        if entry.is_dir():
            return
        _ensure_recovery_entry_within_root(entry, "cleanup entry", data_root)
        entry.unlink()
        return
    if entry.is_file():
        _ensure_recovery_path_within_root(str(entry), "cleanup entry", data_root)
        entry.unlink()


def file_content_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return sha256_file_uncached(path)


def atomic_write_journal_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a migration journal in its readable, indented JSON format."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["atomic_write_journal_json", "commit_ordered_replacements", "file_content_hash"]
