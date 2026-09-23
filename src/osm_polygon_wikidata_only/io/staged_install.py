"""Multi-file staged installation with all-or-nothing rollback.

Responsibility:
    Install a set of already-staged files over their final paths as one
    transaction. Every existing target (and every stale path being retired)
    is first moved aside to a hidden backup next to it; the staged files are
    then moved into place, with the manifest last so it never names a file
    that is not installed yet. Any failure, including one during the backup
    phase, removes what was installed and restores every backup. Staged
    temporaries and backups never outlive the call.

    Moves use :func:`os.replace`, so staging must live on the same
    filesystem as the destination. A cross-filesystem move raises
    :class:`CrossFilesystemInstallError` rather than a bare ``EXDEV``.
"""

from __future__ import annotations

import errno
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


class CrossFilesystemInstallError(OSError):
    """Raised when a staged install would have to cross filesystems."""

    def __init__(self, source: Path, destination: Path) -> None:
        super().__init__(errno.EXDEV, "cannot cross filesystems (EXDEV)")
        self.source = source
        self.destination = destination


def install_staged_files(
    staged: dict[Path, Path],
    *,
    stale: Iterable[Path] = (),
    manifest_path: Path | None = None,
) -> None:
    """Install ``staged`` (final path -> staged path) and retire ``stale`` atomically."""
    targets = sorted(set(staged) | set(stale), key=lambda path: path.as_posix())
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        backup_targets(targets, backups)
        install_files(staged, installed, manifest_path=manifest_path)
    except BaseException:
        restore_files(installed, backups)
        raise
    finally:
        cleanup_transaction(staged, backups)


def backup_targets(targets: list[Path], backups: dict[Path, Path]) -> None:
    try:
        for target in targets:
            if target.exists():
                backups[target] = backup_existing(target)
    except BaseException:
        restore_files([], backups)
        raise


def install_files(
    staged: dict[Path, Path],
    installed: list[Path],
    *,
    manifest_path: Path | None = None,
) -> None:
    ordered = sorted(
        staged.items(),
        key=lambda item: (
            manifest_path is not None and item[0] == manifest_path,
            item[0].as_posix(),
        ),
    )
    for final, temporary in ordered:
        final.parent.mkdir(parents=True, exist_ok=True)
        replace_path(temporary, final)
        installed.append(final)


def restore_files(installed: list[Path], backups: dict[Path, Path]) -> None:
    for final in installed:
        final.unlink(missing_ok=True)
    for final, backup in sorted(backups.items(), key=lambda item: item[0].as_posix()):
        if backup.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            replace_path(backup, final)


def cleanup_transaction(staged: dict[Path, Path], backups: dict[Path, Path]) -> None:
    for temporary in staged.values():
        temporary.unlink(missing_ok=True)
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def backup_existing(path: Path) -> Path:
    descriptor, raw_backup = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".backup", dir=path.parent
    )
    os.close(descriptor)
    backup = Path(raw_backup)
    replace_path(path, backup)
    return backup


def replace_path(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise CrossFilesystemInstallError(source, destination) from error
        raise
