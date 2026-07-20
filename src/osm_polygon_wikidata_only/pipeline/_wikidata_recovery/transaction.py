from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.steps import sha256_file
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps, loads

_TRANSACTION_VERSION = "wikidata-recovery-transaction-v1"


def transaction_directory(root: Path, stem: str) -> Path:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid recovery transaction stem: {stem!r}")
    return root / stem


def recover_interrupted_transactions(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    recovered: list[str] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        journal_path = directory / "journal.json"
        if not journal_path.is_file():
            continue
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
        recovered.append(str(journal["stem"]))
        _cleanup(directory, journal)
    return tuple(recovered)


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
    journal: dict[str, Any] = {
        "contract_version": _TRANSACTION_VERSION,
        "stem": stem,
        "phase": "prepared",
        "entries": entries,
    }
    _write_journal(journal_path, journal)
    try:
        if before_commit is not None:
            before_commit()
        journal["phase"] = "committing"
        _write_journal(journal_path, journal)
        _roll_forward(journal)
        journal["phase"] = "committed"
        _write_journal(journal_path, journal)
    except BaseException:
        _rollback(journal)
        _cleanup(directory, journal)
        raise
    _cleanup(directory, journal)


def _read_journal(path: Path) -> dict[str, Any]:
    raw: object = loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("contract_version") != _TRANSACTION_VERSION:
        raise RuntimeError(f"Invalid recovery transaction journal: {path}")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise RuntimeError(f"Invalid recovery transaction entries: {path}")
    return raw


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    atomic_write_text(path, dumps(journal) + "\n")


def _roll_forward(journal: dict[str, Any]) -> None:
    for entry in journal["entries"]:
        target = Path(entry["target"])
        staged = Path(entry["staged"])
        staged_hash = str(entry["staged_hash"])
        if target.is_file() and sha256_file(target) == staged_hash:
            continue
        if not staged.is_file() or sha256_file(staged) != staged_hash:
            raise RuntimeError(f"Recovery transaction staged file is unavailable: {staged}")
        _atomic_copy(staged, target)
        if sha256_file(target) != staged_hash:
            raise RuntimeError(f"Recovery transaction verification failed: {target}")


def _rollback(journal: dict[str, Any]) -> None:
    for entry in reversed(journal["entries"]):
        target = Path(entry["target"])
        backup = Path(entry["backup"])
        if bool(entry["existed"]):
            if not backup.is_file():
                raise RuntimeError(f"Recovery transaction backup is unavailable: {backup}")
            _atomic_copy(backup, target)
            if sha256_file(target) != str(entry["original_hash"]):
                raise RuntimeError(f"Recovery transaction rollback verification failed: {target}")
        elif target.exists():
            target.unlink()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(raw_tmp)
    os.close(fd)
    try:
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
