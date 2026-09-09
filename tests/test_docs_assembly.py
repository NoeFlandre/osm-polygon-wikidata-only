"""Tests for deterministic assembly of public documentation assets."""

from pathlib import Path

import pytest

from scripts.assemble_docs_site import PRESENTATION_FILES, assemble_public_presentations


def test_assemble_public_presentations_copies_the_declared_assets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    site_dir = tmp_path / "site"
    for index, relative_path in enumerate(PRESENTATION_FILES):
        source = source_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"asset-{index}", encoding="utf-8")

    assemble_public_presentations(source_root, site_dir)

    for index, relative_path in enumerate(PRESENTATION_FILES):
        target = site_dir / relative_path
        assert target.read_text(encoding="utf-8") == f"asset-{index}"


def test_assemble_public_presentations_fails_before_publishing_partial_site(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    site_dir = tmp_path / "site"
    first_path = source_root / PRESENTATION_FILES[0]
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("first", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="presentation asset"):
        assemble_public_presentations(source_root, site_dir)

    assert not site_dir.exists()
