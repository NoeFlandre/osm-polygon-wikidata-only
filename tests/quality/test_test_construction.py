"""Tests build objects through their constructors, never around them."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]


def test_no_test_builds_an_object_with_dunder_new() -> None:
    """``Cls.__new__(Cls)`` skips ``__init__``, so the object may not be constructible for real."""
    offenders = [
        f"{path.relative_to(TESTS).as_posix()}:{node.lineno}"
        for path in sorted(TESTS.rglob("*.py"))
        if path != Path(__file__).resolve()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__new__"
    ]

    assert offenders == []
