"""Atomic-write cleanup contract.

The :func:`atomic_write_text` and :func:`atomic_save_png` helpers
write to a temporary file then atomically rename to the final path.
The ``except BaseException`` boundary they use is intentional: temp
files must be cleaned up even when the writer is interrupted
(``KeyboardInterrupt``, ``SystemExit``), not only on ordinary
exception types.

These tests pin that contract so any future narrowing is forced to
re-examine cleanup guarantees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_atomic_write_text_writes_atomically(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.io.atomic import atomic_write_text

    target = tmp_path / "x.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    # No leftover temp files in the directory.
    leftover = [p for p in tmp_path.iterdir() if p.name != "x.txt"]
    assert leftover == []


def test_atomic_write_text_cleans_up_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the inner write raises, the temp file must be unlinked.

    This proves the ``except BaseException`` cleanup branch is taken
    on exception, not only on success.
    """
    from osm_polygon_wikidata_only.io import atomic

    target = tmp_path / "x.txt"

    def _broken_fdopen(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(atomic.os, "fdopen", _broken_fdopen)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        atomic.atomic_write_text(target, "never written")
    # The temp file must NOT remain in the directory.
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"expected no leftover files, got {leftover}"


def test_atomic_write_text_cleans_up_temp_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``except BaseException`` must catch ``KeyboardInterrupt`` too.

    A narrow ``except Exception`` would leak the temp file on Ctrl-C;
    this test pins that the cleanup branch fires for that path.
    """
    from osm_polygon_wikidata_only.io import atomic

    target = tmp_path / "x.txt"

    def _interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic.os, "fdopen", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        atomic.atomic_write_text(target, "never written")
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"expected no leftover files, got {leftover}"


def test_atomic_save_png_cleans_up_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``atomic_save_png`` must unlink its temp file when savefig raises."""
    from osm_polygon_wikidata_only.hf._geographic import rendering

    target = tmp_path / "out.png"

    class _FakeFig:
        def savefig(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated savefig failure")

    with pytest.raises(RuntimeError, match="simulated savefig failure"):
        rendering.atomic_save_png(_FakeFig(), target)
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"expected no leftover files, got {leftover}"
