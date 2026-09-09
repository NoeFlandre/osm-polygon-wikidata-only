"""Assemble tracked presentation assets into a built documentation site."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

PRESENTATION_FILES: tuple[Path, ...] = (
    Path("presentations/dataset.html"),
    Path("presentations/codebase.html"),
    Path("presentations/assets/coverage_map.png"),
    Path("presentations/assets/text_density.png"),
    Path("presentations/assets/text_presence.png"),
)


def assemble_public_presentations(source_root: Path, site_dir: Path) -> None:
    """Copy all tracked public presentations into ``site_dir``.

    Validate every source first so a missing asset cannot leave a partially
    assembled site behind.
    """

    missing = [
        relative_path
        for relative_path in PRESENTATION_FILES
        if not (source_root / relative_path).is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"presentation asset(s) missing: {formatted}")

    for relative_path in PRESENTATION_FILES:
        destination = site_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, destination)


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
