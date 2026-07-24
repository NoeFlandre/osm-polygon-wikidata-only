"""Phase 2 / Amendment 4: Migration transaction correctness.

A crash after any replacement MUST leave every remaining staged
artifact available for roll-forward. The current code deletes staged
files after a mid-flight failure, making recovery impossible.

The journaled transaction must:

1. Crash mid-flight (e.g. after index 1 of at least 4 replacements)
   and preserve every remaining staged file on disk.
2. Validate all target/staged/journal paths remain inside their
   expected roots.
3. Reconstruct/restart and prove complete roll-forward -- every
   target ends at the staged hash, with no data loss.
4. Do NOT swallow manifest-update failures (Amendment 5 related).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _import_module():
    try:
        from osm_polygon_wikidata_only.pipeline import link_migration as mod
    except ImportError:
        pytest.fail("link_migration module must exist")
    return mod


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_pair(tmp_path: Path, name: str, payload: bytes = b"") -> tuple[Path, Path]:
    target = tmp_path / "targets" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"ORIGINAL_" + name.encode())
    staged = tmp_path / "staged" / name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(payload if payload else b"STAGED_" + name.encode())
    return target, staged


def test_staged_files_preserved_after_mid_flight_crash(tmp_path: Path) -> None:
    """After a crash at index 1 of 4, staged files for indices 2 and 3
    must remain on disk for roll-forward recovery.
    """
    mod = _import_module()
    targets_staged: list[tuple[Path, Path]] = []
    for index in range(4):
        target, staged = _make_pair(
            tmp_path, f"file{index}.parquet", payload=f"STAGED_{index}".encode()
        )
        targets_staged.append((target, staged))

    directory = tmp_path / "txn"
    directory.mkdir(parents=True, exist_ok=True)

    crash_index = {1}  # crash after applying index 1

    def _crash_hook(index: int, target: Path) -> None:
        if index in crash_index:
            raise RuntimeError(f"simulated crash at index {index}")

    with pytest.raises(RuntimeError, match="simulated crash"):
        mod._commit_ordered_replacements(
            directory=directory,
            stem="alpha-latest",
            replacements=targets_staged,
            _crash_hook=_crash_hook,
        )

    # Index 0 and 1 are applied; index 2 and 3 are NOT applied.
    # Staged files for indices 2 and 3 MUST still exist for roll-forward.
    for index, (_, staged) in enumerate(targets_staged):
        if index in crash_index:
            # Already applied; staged may or may not have been moved.
            # Index 1 staged was consumed by os.replace.
            continue
        if index > 1:
            # Not yet applied; staged MUST still be on disk.
            assert staged.is_file(), (
                f"Staged file for index {index} must remain on disk for roll-forward"
            )


def test_roll_forward_after_mid_flight_crash(tmp_path: Path) -> None:
    """After a crash, a fresh call to apply_link_migration must complete
    the roll-forward -- every target ends at its staged hash.
    """
    mod = _import_module()
    targets_staged: list[tuple[Path, Path]] = []
    expected: dict[Path, str] = {}
    for index in range(4):
        target, staged = _make_pair(
            tmp_path, f"file{index}.parquet", payload=f"STAGED_{index}".encode()
        )
        targets_staged.append((target, staged))
        expected[target] = _file_hash(staged)

    directory = tmp_path / "txn"
    directory.mkdir(parents=True, exist_ok=True)

    crash_index = {1}

    def _crash_hook(index: int, target: Path) -> None:
        if index in crash_index:
            raise RuntimeError(f"simulated crash at index {index}")

    with pytest.raises(RuntimeError, match="simulated crash"):
        mod._commit_ordered_replacements(
            directory=directory,
            stem="alpha-latest",
            replacements=targets_staged,
            _crash_hook=_crash_hook,
        )

    # Now: roll-forward. A subsequent call (with the same directory)
    # must re-apply the remaining replacements and complete the
    # transaction. The journal file should be present and the staged
    # files for indices 2 and 3 should still be on disk.
    journal = directory / "journal.json"
    assert journal.is_file(), "Journal must persist after mid-flight crash"

    # Replay by calling the public API again.
    mod._commit_ordered_replacements(
        directory=directory,
        stem="alpha-latest",
        replacements=targets_staged,
    )

    for target, expected_hash in expected.items():
        assert _file_hash(target) == expected_hash, (
            f"After roll-forward, {target} hash must match staged hash"
        )


def test_target_staged_journal_paths_remain_inside_roots(tmp_path: Path) -> None:
    """All paths in the journal must be inside their declared roots."""
    mod = _import_module()
    targets_staged: list[tuple[Path, Path]] = []
    for index in range(2):
        target, staged = _make_pair(tmp_path, f"file{index}.parquet")
        targets_staged.append((target, staged))

    directory = tmp_path / "txn"
    directory.mkdir(parents=True, exist_ok=True)

    # Crash at index 1 so the journal remains on disk for inspection.
    crash_index = {1}

    def _crash_hook(index: int, target: Path) -> None:
        if index in crash_index:
            raise RuntimeError("simulated crash")

    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="simulated crash"):
        mod._commit_ordered_replacements(
            directory=directory,
            stem="alpha-latest",
            replacements=targets_staged,
            _crash_hook=_crash_hook,
        )

    import json

    journal = json.loads((directory / "journal.json").read_text())
    for entry in journal["entries"]:
        target = Path(entry["target"]).resolve()
        staged = Path(entry["staged"]).resolve()
        # Targets/staged must resolve under tmp_path.
        try:
            target.relative_to(tmp_path.resolve())
        except ValueError:
            pytest.fail(f"Journal target {target} escapes root {tmp_path}")
        try:
            staged.relative_to(tmp_path.resolve())
        except ValueError:
            pytest.fail(f"Journal staged {staged} escapes root {tmp_path}")


def test_manifest_update_failure_is_not_swallowed(tmp_path: Path) -> None:
    """The apply stage's augmentation-manifest update must propagate
    exceptions -- they are NOT logged-and-continued.
    """
    mod = _import_module()
    src = open(mod.__file__).read()
    assert "except Exception" not in src or "raise" in src, (
        "link_migration must not swallow manifest-update failures with a broad except + log"
    )
    # More specifically, the augmentation-manifest-update block must not
    # end with `raise` missing -- every except must re-raise.
    import re

    bad_patterns = list(re.finditer(r"except Exception:\s*\n[^\n]*log\w*\(", src))
    assert not bad_patterns, (
        f"link_migration contains a broad `except Exception: log...` pattern: "
        f"{[m.group(0) for m in bad_patterns]}"
    )
