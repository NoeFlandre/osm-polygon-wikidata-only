"""Assemble tracked presentation assets into a built documentation site."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

PRESENTATION_FILES: tuple[Path, ...] = (
    Path("presentations/dataset.html"),
    Path("presentations/codebase.html"),
    Path("presentations/assets/coverage_map.png"),
    Path("presentations/assets/text_density.png"),
    Path("presentations/assets/text_presence.png"),
)


def _validate_sources(source_root: Path) -> None:
    missing = [
        relative_path
        for relative_path in PRESENTATION_FILES
        if not (source_root / relative_path).is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"presentation asset(s) missing: {formatted}")


def _stage_site(source_root: Path, site_dir: Path, staging_dir: Path) -> None:
    if site_dir.exists():
        shutil.copytree(site_dir, staging_dir, dirs_exist_ok=True)

    for relative_path in PRESENTATION_FILES:
        destination = staging_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, destination)


def _backup_site(site_dir: Path) -> Path | None:
    if not site_dir.exists():
        return None

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{site_dir.name}-backup-",
            dir=site_dir.parent,
        )
    )
    backup_dir.rmdir()
    site_dir.replace(backup_dir)
    return backup_dir


def _restore_site(site_dir: Path, backup_dir: Path | None) -> None:
    if site_dir.exists():
        shutil.rmtree(site_dir)
    if backup_dir is not None and backup_dir.exists():
        backup_dir.replace(site_dir)


def _publish_site(staging_dir: Path, site_dir: Path) -> None:
    backup_dir = _backup_site(site_dir)
    try:
        staging_dir.replace(site_dir)
    except OSError:
        _restore_site(site_dir, backup_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def assemble_public_presentations(source_root: Path, site_dir: Path) -> None:
    """Copy tracked public presentations into site_dir atomically.

    Existing site content is staged with the new presentations and replaced
    only after every source copy succeeds.
    """
    _validate_sources(source_root)
    site_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{site_dir.name}-",
        dir=site_dir.parent,
    ) as staging_name:
        _stage_site(source_root, site_dir, Path(staging_name))
        _publish_site(Path(staging_name), site_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the documentation asset assembly command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[1]
    assemble_public_presentations(source_root, args.site_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
