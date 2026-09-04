"""Ordered, journaled file replacement for link migration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.io.atomic import atomic_write_text

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
    _crash_hook: Callable[[int, Path], None] | None = None,
) -> None:
    """Replace data before manifests and recover interrupted commits."""
    if not replacements:
        return
    _validate_replacement_targets(replacements)
    ordered = sorted(
        replacements,
        key=lambda item: (1 if item[0].suffix == ".json" else 0, str(item[0])),
    )

    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "journal.json"
    if journal_path.exists():
        _recover_directory(directory, stem)
        return

    _commit_new_transaction(directory, stem, ordered, _crash_hook)


def _validate_replacement_targets(replacements: list[tuple[Path, Path]]) -> None:
    """Reject duplicate target paths before creating a transaction journal."""
    targets = [target for target, _ in replacements]
    if len(set(targets)) != len(targets):
        raise ValueError("Link migration transaction contains duplicate targets")


def _commit_new_transaction(
    directory: Path,
    stem: str,
    ordered: list[tuple[Path, Path]],
    crash_hook: Callable[[int, Path], None] | None,
) -> None:
    """Prepare, apply, and clean a new ordered replacement transaction."""
    journal_path = directory / "journal.json"

    entries = [_prepare_entry(directory, target, staged) for target, staged in ordered]
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
    _atomic_write_json(journal_path, journal)

    index_ref = [0]
    try:
        _apply_entries(entries, crash_hook, index_ref)
        journal["phase"] = "committed"
        _atomic_write_json(journal_path, journal)
    except BaseException:
        _record_transaction_failure(directory, journal_path, journal, entries, index_ref[0])
        raise
    _cleanup(directory)


def _record_transaction_failure(
    directory: Path,
    journal_path: Path,
    journal: dict[str, Any],
    entries: list[_TransactionEntry],
    index: int,
) -> None:
    """Record whether a failed apply can be rolled back immediately."""
    if index == 0:
        _rollback_entries(entries)
        journal["phase"] = "rolled_back"
        _atomic_write_json(journal_path, journal)
        _cleanup(directory)
        return
    journal["phase"] = "interrupted"
    journal["interrupted_at_index"] = int(index)
    _atomic_write_json(journal_path, journal)


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
        if _file_content_hash(entry.target) != entry.staged_hash:
            raise RuntimeError(f"Link migration post-hook hash mismatch for {entry.target}")


def _prepare_entry(directory: Path, target: Path, staged: Path) -> _TransactionEntry:
    if not staged.is_file():
        raise FileNotFoundError(f"Staged link migration file is missing: {staged}")
    backup = directory / f"{target.name}.backup"
    existed = target.is_file()
    original_hash = ""
    if existed:
        shutil.copyfile(target, backup)
        original_hash = _file_content_hash(target)
    return _TransactionEntry(
        target=target,
        staged=staged,
        backup=backup if existed else None,
        existed=existed,
        original_hash=original_hash,
        staged_hash=_file_content_hash(staged),
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
    if _file_content_hash(entry.target) != entry.staged_hash:
        raise RuntimeError(f"Link migration verification failed: {entry.target}")


def _file_matches_hash(path: Path, expected_hash: str) -> bool:
    """Return whether a regular file has the expected content hash."""
    return path.is_file() and _file_content_hash(path) == expected_hash


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
    if _file_content_hash(entry.target) != entry.original_hash:
        raise RuntimeError(f"Link migration rollback verification failed: {entry.target}")


def _recover_directory(directory: Path, stem: str) -> None:
    journal_path = directory / "journal.json"
    if not journal_path.is_file():
        return
    raw = _load_recovery_journal(journal_path, stem)
    _recover_entries(raw.get("entries", []))
    _cleanup(directory)


def _load_recovery_journal(path: Path, stem: str) -> dict[str, Any]:
    """Load and validate a link migration recovery journal."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid link migration journal: {path}")
    if raw.get("stem") != stem:
        raise RuntimeError(f"Link migration journal stem mismatch: {raw.get('stem')!r} vs {stem!r}")
    return raw


def _recover_entries(entries: list[dict[str, Any]]) -> None:
    """Roll forward all entries from an interrupted migration journal."""
    for entry in entries:
        _recover_entry(entry)


def _recover_entry(entry: dict[str, Any]) -> None:
    """Roll forward one interrupted migration entry."""
    target = Path(entry["target"])
    staged = Path(entry["staged"])
    staged_hash = str(entry["staged_hash"])
    if _file_matches_hash(target, staged_hash):
        return
    if not _file_matches_hash(staged, staged_hash):
        raise RuntimeError(f"Link migration recovery: staged file unavailable: {staged}")
    _prepare_recovery_target(entry, target, staged)
    if _file_content_hash(target) != staged_hash:
        raise RuntimeError(f"Link migration recovery verification failed: {target}")


def _prepare_recovery_target(entry: dict[str, Any], target: Path, staged: Path) -> None:
    """Move one staged file into place, removing a superseded backup."""
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_recovery_backup(entry)
    _move_recovery_staged(staged, target)


def _remove_recovery_backup(entry: dict[str, Any]) -> None:
    """Remove a superseded backup when the original target existed."""
    backup = Path(entry["backup"]) if entry.get("backup") else None
    if bool(entry["existed"]) and backup is not None and backup.is_file():
        backup.unlink()


def _move_recovery_staged(staged: Path, target: Path) -> None:
    """Replace or create a recovered target from its staged file."""
    if target.is_file():
        os.replace(staged, target)
    else:
        shutil.move(str(staged), str(target))


def _cleanup(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.is_file():
            entry.unlink()
    with suppress(OSError):
        directory.rmdir()


def _file_content_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a migration journal in its readable, indented JSON format."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["commit_ordered_replacements"]
