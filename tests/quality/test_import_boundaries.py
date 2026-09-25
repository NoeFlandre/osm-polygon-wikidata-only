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


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            modules.add(prefix)
    return modules


def test_sync_runner_is_a_pure_state_executor() -> None:
    """The sync runner receives collaborators; the CLI shell owns I/O wiring."""
    runner = REPOSITORY / "src/osm_polygon_wikidata_only/pipeline/sync_runner.py"
    forbidden = (
        "argparse",
        "osm_polygon_wikidata_only.cli",
        "osm_polygon_wikidata_only.hf",
        "osm_polygon_wikidata_only.config",
        "..cli",
        "..hf",
        "..config",
    )
    leaked = sorted(
        module
        for module in _imported_modules(runner)
        if any(module == name or module.startswith(f"{name}.") for name in forbidden)
    )
    assert leaked == []
