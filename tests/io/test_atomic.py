"""Contract for the shared atomic-publication ritual.

`osm_polygon_wikidata_only.io.atomic` owns one mechanism: create a hidden
temporary sibling inside the destination directory, let a caller fill it, and
install it over the target with a single :func:`os.replace`. These tests pin
that mechanism directly so the writers built on it -- text, deterministic JSON,
Parquet, byte copies -- do not each need to re-prove durability.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.io.atomic import (
    atomic_copy_file,
    atomic_replacement,
    atomic_write_json,
)


def test_atomic_replacement_yields_a_hidden_sibling_of_the_target(tmp_path: Path) -> None:
    """The temporary file must sit beside the target so the rename is atomic."""
    target = tmp_path / "artifact.bin"

    with atomic_replacement(target) as temporary:
        assert temporary.parent == target.parent
        assert temporary.name.startswith(".artifact.bin.")
        assert temporary.name.endswith(".tmp")
        assert temporary.exists()
        temporary.write_bytes(b"payload")

    assert target.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_replacement_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "artifact.bin"

    with atomic_replacement(target) as temporary:
        temporary.write_bytes(b"payload")

    assert target.read_bytes() == b"payload"


def test_atomic_replacement_overwrites_an_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"stale")

    with atomic_replacement(target) as temporary:
        temporary.write_bytes(b"fresh")

    assert target.read_bytes() == b"fresh"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_replacement_leaves_the_target_untouched_when_the_body_raises(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="simulated failure"):
        with atomic_replacement(target) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("simulated failure")

    assert target.read_bytes() == b"stale"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_replacement_cleans_up_after_a_keyboard_interrupt(tmp_path: Path) -> None:
    """Cleanup catches ``BaseException``; Ctrl-C must not leak a temporary file."""
    target = tmp_path / "artifact.bin"

    with pytest.raises(KeyboardInterrupt):
        with atomic_replacement(target) as temporary:
            temporary.write_bytes(b"partial")
            raise KeyboardInterrupt

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_replacement_tolerates_a_body_that_removed_the_temporary_file(
    tmp_path: Path,
) -> None:
    """Cleanup is best-effort: a body may consume or move the sibling itself."""
    target = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="simulated failure"):
        with atomic_replacement(target) as temporary:
            temporary.unlink()
            raise RuntimeError("simulated failure")

    assert list(tmp_path.iterdir()) == []


def test_atomic_replacement_closes_the_descriptor_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``mkstemp`` descriptor must be closed; callers reopen by path."""
    from osm_polygon_wikidata_only.io import atomic

    closed: list[int] = []
    real_close = os.close

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(atomic.os, "close", recording_close)

    with atomic_replacement(tmp_path / "artifact.bin") as temporary:
        temporary.write_bytes(b"payload")

    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])


def test_atomic_write_json_uses_the_deterministic_encoding_with_a_trailing_newline(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.json"

    atomic_write_json(target, {"b": 1, "a": "é"})

    assert target.read_text(encoding="utf-8") == '{"a":"é","b":1}\n'
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_json_replaces_an_existing_document(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1})

    atomic_write_json(target, {"generation": 2})

    assert target.read_text(encoding="utf-8") == '{"generation":2}\n'


def test_atomic_copy_file_publishes_the_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "published" / "target.bin"
    source.write_bytes(b"payload\x00bytes")

    atomic_copy_file(source, target)

    assert target.read_bytes() == b"payload\x00bytes"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_copy_file_leaves_no_temporary_file_when_the_source_disappears(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.bin"
    target = tmp_path / "target.bin"

    with pytest.raises(FileNotFoundError):
        atomic_copy_file(source, target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
