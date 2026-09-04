"""Provide transactional filesystem recovery for Wikidata repairs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.augmentation.steps import sha256_file
from osm_polygon_wikidata_only.io.atomic import atomic_copy_file, atomic_write_json
from osm_polygon_wikidata_only.utils.json import loads

_TRANSACTION_VERSION = "wikidata-recovery-transaction-v1"


def transaction_directory(root: Path, stem: str) -> Path:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid recovery transaction stem: {stem!r}")
    return root / stem


def recover_interrupted_transactions(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    recovered: list[str] = []
    for directory in _transaction_directories(root):
        stem = _recover_transaction_directory(directory)
        if stem is not None:
            recovered.append(stem)
    return tuple(recovered)


def _transaction_directories(root: Path) -> list[Path]:
    """Return transaction subdirectories in stable order."""
    return sorted(path for path in root.iterdir() if path.is_dir())


def _recover_transaction_directory(directory: Path) -> str | None:
    """Recover one transaction directory when it contains a journal."""
    journal_path = directory / "journal.json"
    if not journal_path.is_file():
        return None
    journal = _read_journal(journal_path)
    phase = journal["phase"]
    if phase == "prepared":
        _rollback(journal)
    elif phase in {"committing", "committed"}:
        _roll_forward(journal)
        journal["phase"] = "committed"
        _write_journal(journal_path, journal)
    else:
        raise RuntimeError(f"Unknown recovery transaction phase {phase!r}")
    stem = str(journal["stem"])
    _cleanup(directory, journal)
    return stem


def commit_replacements(
    directory: Path,
    stem: str,
    replacements: list[tuple[Path, Path]],
    *,
    before_commit: Callable[[], None] | None = None,
) -> None:
    if not replacements:
        return
    targets = [target for target, _ in replacements]
    if len(set(targets)) != len(targets):
        raise ValueError("Recovery transaction contains duplicate targets")
    journal_path = directory / "journal.json"
    directory.mkdir(parents=True, exist_ok=True)
    if journal_path.exists():
        raise RuntimeError(f"Recovery transaction is already prepared: {directory}")
    entries = _prepare_entries(directory, replacements)
    journal: dict[str, Any] = {
        "contract_version": _TRANSACTION_VERSION,
        "stem": stem,
        "phase": "prepared",
        "entries": entries,
    }
    _write_journal(journal_path, journal)
    _commit_or_rollback(directory, journal_path, journal, before_commit)


def _prepare_entries(
    directory: Path,
    replacements: list[tuple[Path, Path]],
) -> list[dict[str, Any]]:
    """Hash staged files and create backups before a transaction is committed."""
    entries: list[dict[str, Any]] = []
    for index, (target, staged) in enumerate(sorted(replacements, key=lambda item: str(item[0]))):
        if not staged.is_file():
            raise FileNotFoundError(f"Staged recovery file is missing: {staged}")
        backup = directory / f"{index:03d}.backup"
        existed = target.is_file()
        original_hash = ""
        if existed:
            shutil.copyfile(target, backup)
            original_hash = sha256_file(target)
        entries.append(
            {
                "target": str(target),
                "staged": str(staged),
                "backup": str(backup),
                "existed": existed,
                "original_hash": original_hash,
                "staged_hash": sha256_file(staged),
            }
        )
    return entries


def _commit_journal(
    journal_path: Path,
    journal: dict[str, Any],
    before_commit: Callable[[], None] | None,
) -> None:
    """Advance a prepared journal through the commit phases."""
    if before_commit is not None:
        before_commit()
    journal["phase"] = "committing"
    _write_journal(journal_path, journal)
    _roll_forward(journal)
    journal["phase"] = "committed"
    _write_journal(journal_path, journal)


def _commit_or_rollback(
    directory: Path,
    journal_path: Path,
    journal: dict[str, Any],
    before_commit: Callable[[], None] | None,
) -> None:
    """Commit a prepared journal or restore its backups on failure."""
    try:
        _commit_journal(journal_path, journal, before_commit)
    except BaseException:
        _rollback(journal)
        _cleanup(directory, journal)
        raise
    _cleanup(directory, journal)


def _read_journal(path: Path) -> dict[str, Any]:
    raw: object = loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("contract_version") != _TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid recovery transaction journal: {path}")
    _validate_journal_entries(raw.get("entries"), path)
    return cast(dict[str, Any], raw)


def _validate_journal_entries(entries: object, path: Path) -> None:
    """Validate the journal entry collection shape."""
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise RuntimeError(f"Invalid recovery transaction entries: {path}")


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    atomic_write_json(path, journal)


def _roll_forward(journal: dict[str, Any]) -> None:
    for entry in journal["entries"]:
        _roll_forward_entry(entry)


def _roll_forward_entry(entry: dict[str, Any]) -> None:
    """Apply and verify one staged replacement."""
    target = Path(entry["target"])
    staged = Path(entry["staged"])
    staged_hash = str(entry["staged_hash"])
    if _file_matches_hash(target, staged_hash):
        return
    if not _file_matches_hash(staged, staged_hash):
        raise RuntimeError(f"Recovery transaction staged file is unavailable: {staged}")
    atomic_copy_file(staged, target)
    if sha256_file(target) != staged_hash:
        raise RuntimeError(f"Recovery transaction verification failed: {target}")


def _file_matches_hash(path: Path, expected_hash: str) -> bool:
    """Return whether a regular file exists with the expected digest."""
    return path.is_file() and sha256_file(path) == expected_hash


def _rollback(journal: dict[str, Any]) -> None:
    for entry in reversed(journal["entries"]):
        _rollback_entry(entry)


def _rollback_entry(entry: dict[str, Any]) -> None:
    """Restore or remove one target according to its transaction backup."""
    target = Path(entry["target"])
    backup = Path(entry["backup"])
    if bool(entry["existed"]):
        if not backup.is_file():
            raise RuntimeError(f"Recovery transaction backup is unavailable: {backup}")
        atomic_copy_file(backup, target)
        if sha256_file(target) != str(entry["original_hash"]):
            raise RuntimeError(f"Recovery transaction rollback verification failed: {target}")
    elif target.exists():
        target.unlink()


def _cleanup(directory: Path, journal: dict[str, Any]) -> None:
    for entry in journal["entries"]:
        Path(entry["staged"]).unlink(missing_ok=True)
        Path(entry["backup"]).unlink(missing_ok=True)
    (directory / "journal.json").unlink(missing_ok=True)
    directory.rmdir()


__all__ = [
    "commit_replacements",
    "recover_interrupted_transactions",
    "transaction_directory",
]
