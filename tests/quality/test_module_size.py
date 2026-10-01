"""Production modules stay small enough to review (issue #107)."""

from __future__ import annotations

from pathlib import Path

MAX_MODULE_LINES = 600
SOURCE_ROOTS = (
    Path(__file__).resolve().parents[2] / "src",
    Path(__file__).resolve().parents[2] / "preprocessing" / "src",
)


def test_no_production_module_exceeds_the_line_limit() -> None:
    oversized = {
        path.as_posix(): lines
        for root in SOURCE_ROOTS
        for path in sorted(root.rglob("*.py"))
        if (lines := len(path.read_text(encoding="utf-8").splitlines())) > MAX_MODULE_LINES
    }

    assert oversized == {}
