"""Regression contracts for source import boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def test_source_modules_do_not_import_private_symbols_across_module_boundaries() -> None:
    """Focused modules must consume named collaborators, not private seams."""
    violations: list[str] = []
    source_root = REPOSITORY / "src"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            private_names = [alias.name for alias in node.names if alias.name.startswith("_")]
            if private_names:
                violations.append(
                    f"{path.relative_to(REPOSITORY)}:{node.lineno}: {', '.join(private_names)}"
                )
    assert violations == []
