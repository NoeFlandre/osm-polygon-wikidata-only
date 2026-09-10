"""Tests for deterministic assembly of public documentation assets."""

from pathlib import Path

import pytest

from scripts import assemble_docs_site as assembly_module
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


def test_assemble_public_presentations_keeps_existing_site_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    site_dir = tmp_path / "site"
    for index, relative_path in enumerate(PRESENTATION_FILES):
        source = source_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"asset-{index}", encoding="utf-8")

    site_dir.mkdir()
    previous = site_dir / "previous.txt"
    previous.write_text("previous", encoding="utf-8")
    real_copy2 = assembly_module.shutil.copy2
    failing_source = source_root / PRESENTATION_FILES[1]

    def fail_copy(source: Path, destination: Path) -> str:
        if source == failing_source:
            raise OSError("copy failed")
        return real_copy2(source, destination)

    monkeypatch.setattr(assembly_module.shutil, "copy2", fail_copy)

    with pytest.raises(OSError, match="copy failed"):
        assemble_public_presentations(source_root, site_dir)

    assert previous.read_text(encoding="utf-8") == "previous"


def test_assemble_public_presentations_restores_site_when_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    site_dir = tmp_path / "site"
    for index, relative_path in enumerate(PRESENTATION_FILES):
        source = source_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"asset-{index}", encoding="utf-8")

    site_dir.mkdir()
    previous = site_dir / "previous.txt"
    previous.write_text("previous", encoding="utf-8")
    real_replace = Path.replace

    def fail_staging_replace(source: Path, target: Path) -> Path:
        if target == site_dir and "backup" not in source.name:
            raise OSError("publish failed")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_staging_replace)

    with pytest.raises(OSError, match="publish failed"):
        assemble_public_presentations(source_root, site_dir)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".site-*"))
    assert not (site_dir / PRESENTATION_FILES[0]).exists()


def test_assemble_public_presentations_cleans_backup_when_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    site_dir = tmp_path / "site"
    for index, relative_path in enumerate(PRESENTATION_FILES):
        source = source_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"asset-{index}", encoding="utf-8")

    site_dir.mkdir()
    previous = site_dir / "previous.txt"
    previous.write_text("previous", encoding="utf-8")
    real_replace = Path.replace

    def fail_backup_replace(source: Path, target: Path) -> Path:
        if "backup" in target.name:
            raise OSError("backup failed")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_backup_replace)

    with pytest.raises(OSError, match="backup failed"):
        assemble_public_presentations(source_root, site_dir)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".site-*"))
