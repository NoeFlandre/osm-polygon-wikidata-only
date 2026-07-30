"""Ordered, journaled file replacement for link migration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    targets = [target for target, _ in replacements]
    if len(set(targets)) != len(targets):
        raise ValueError("Link migration transaction contains duplicate targets")
    ordered = sorted(
        replacements,
        key=lambda item: (1 if item[0].suffix == ".json" else 0, str(item[0])),
    )

    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "journal.json"
    if journal_path.exists():
        _recover_directory(directory, stem)
        return

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

    index = 0
    try:
        for index, entry in enumerate(entries):
            _apply_single(entry)
            if _crash_hook is not None:
                _crash_hook(index, entry.target)
            if _file_content_hash(entry.target) != entry.staged_hash:
                raise RuntimeError(f"Link migration post-hook hash mismatch for {entry.target}")
        journal["phase"] = "committed"
        _atomic_write_json(journal_path, journal)
    except BaseException:
        if index == 0:
            _rollback_entries(entries)
            journal["phase"] = "rolled_back"
            _atomic_write_json(journal_path, journal)
            _cleanup(directory)
        else:
            journal["phase"] = "interrupted"
            journal["interrupted_at_index"] = int(index)
            _atomic_write_json(journal_path, journal)
        raise
    _cleanup(directory)


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
    if entry.target.is_file() and _file_content_hash(entry.target) == entry.staged_hash:
        return
    if not entry.staged.is_file() or _file_content_hash(entry.staged) != entry.staged_hash:
        raise RuntimeError(f"Link migration staged file is unavailable: {entry.staged}")
    entry.target.parent.mkdir(parents=True, exist_ok=True)
    if entry.target.is_file():
        os.replace(entry.staged, entry.target)
    else:
        shutil.move(str(entry.staged), str(entry.target))
    if _file_content_hash(entry.target) != entry.staged_hash:
        raise RuntimeError(f"Link migration verification failed: {entry.target}")


def _rollback_entries(entries: list[_TransactionEntry]) -> None:
    for entry in reversed(entries):
        if entry.existed and entry.backup is not None:
            if not entry.backup.is_file():
                raise RuntimeError(f"Link migration backup is unavailable: {entry.backup}")
            os.replace(entry.backup, entry.target)
            if _file_content_hash(entry.target) != entry.original_hash:
                raise RuntimeError(f"Link migration rollback verification failed: {entry.target}")
        elif entry.target.is_file():
            entry.target.unlink()


def _recover_directory(directory: Path, stem: str) -> None:
    journal_path = directory / "journal.json"
    if not journal_path.is_file():
        return
    raw = json.loads(journal_path.read_text(encoding="utf-8"))
    if raw.get("contract_version") != TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid link migration journal: {journal_path}")
    if raw.get("stem") != stem:
        raise RuntimeError(f"Link migration journal stem mismatch: {raw.get('stem')!r} vs {stem!r}")
    for entry in raw.get("entries", []):
        target = Path(entry["target"])
        staged = Path(entry["staged"])
        staged_hash = entry["staged_hash"]
        if target.is_file() and _file_content_hash(target) == staged_hash:
            continue
        if not staged.is_file() or _file_content_hash(staged) != staged_hash:
            raise RuntimeError(f"Link migration recovery: staged file unavailable: {staged}")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = Path(entry["backup"]) if entry.get("backup") else None
        if bool(entry["existed"]) and backup is not None and backup.is_file():
            backup.unlink()
        if target.is_file():
            os.replace(staged, target)
        else:
            shutil.move(str(staged), str(target))
        if _file_content_hash(target) != staged_hash:
            raise RuntimeError(f"Link migration recovery verification failed: {target}")
    _cleanup(directory)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_tmp)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["commit_ordered_replacements"]
