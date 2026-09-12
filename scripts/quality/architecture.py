"""Small AST-based import-boundary checker for the source package.

Only project-local imports are graph edges. Standard-library and third-party
imports, including PyArrow, are intentionally outside this check.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from itertools import pairwise
from pathlib import Path

DEFAULT_PACKAGE = "osm_polygon_wikidata_only"
DEFAULT_SOURCE_ROOT = Path("src") / DEFAULT_PACKAGE


@dataclass(frozen=True, slots=True)
class ImportEdge:
    """One local import, including the source location that created it."""

    source: str
    target: str
    path: Path
    line: int


@dataclass(frozen=True, slots=True)
class ArchitectureViolation:
    """One deterministic architecture-check failure."""

    rule: str
    source: str
    target: str
    path: Path
    line: int
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.rule}: {self.source} -> {self.target} ({self.detail})"
        )


def build_import_graph(source_root: Path, package: str) -> dict[str, tuple[ImportEdge, ...]]:
    """Return local import edges; external imports are deliberately ignored."""
    module_paths = _discover_modules(source_root, package)
    graph: dict[str, tuple[ImportEdge, ...]] = {}
    for module, path in module_paths.items():
        graph[module] = tuple(_imports_from(path, module, module_paths))
    return graph


def check_architecture(
    source_root: Path, package: str = DEFAULT_PACKAGE
) -> tuple[ArchitectureViolation, ...]:
    """Return cycles and forbidden local import-boundary violations."""
    graph = build_import_graph(source_root, package)
    violations = list(_cycle_violations(graph))
    violations.extend(_boundary_violations(graph, package))
    return tuple(violations)


def main(argv: list[str] | None = None) -> int:
    """Run the checker as a small executable quality command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    args = parser.parse_args(argv)

    violations = check_architecture(args.source_root, args.package)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print(f"Architecture checks passed for {args.package}.")
    return 0


def _discover_modules(source_root: Path, package: str) -> dict[str, Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    paths = sorted(source_root.rglob("*.py"))
    if not paths:
        raise ValueError(f"source root contains no Python modules: {source_root}")
    return {_module_name(path, source_root, package): path for path in paths}


def _module_name(path: Path, source_root: Path, package: str) -> str:
    parts = list(path.relative_to(source_root).parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join((package, *parts)) if parts else package


def _imports_from(
    path: Path,
    source: str,
    module_paths: dict[str, Path],
) -> list[ImportEdge]:
    # Parse bytes so each module's declared PEP 263 encoding is honoured.
    tree = ast.parse(path.read_bytes(), filename=str(path))
    return [
        edge
        for node in ast.walk(tree)
        for edge in _edges_for_node(node, source, path, module_paths)
    ]


def _edges_for_node(
    node: ast.AST,
    source: str,
    path: Path,
    module_paths: dict[str, Path],
) -> tuple[ImportEdge, ...]:
    match node:
        case ast.Import(names=names):
            return tuple(_absolute_edges(names, source, path, module_paths, node.lineno))
        case ast.ImportFrom() as import_from:
            return tuple(_from_edges(import_from, source, path, module_paths))
        case _:
            return ()


def _absolute_edges(
    names: list[ast.alias],
    source: str,
    path: Path,
    module_paths: dict[str, Path],
    line: int,
) -> list[ImportEdge]:
    return [
        ImportEdge(source, target, path, line)
        for alias in names
        if (target := _resolve_local(alias.name, module_paths)) is not None
    ]


def _from_edges(
    node: ast.ImportFrom,
    source: str,
    path: Path,
    module_paths: dict[str, Path],
) -> list[ImportEdge]:
    base = _import_base(node, source, path)
    candidates = (f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return [
        ImportEdge(source, target, path, node.lineno)
        for candidate in candidates
        if (target := _resolve_local(candidate, module_paths)) is not None
    ]


def _import_base(node: ast.ImportFrom, source: str, path: Path) -> str:
    """Return the absolute dotted prefix an ``import from`` statement resolves against."""
    package_parts = _relative_package_parts(node, source, path)
    return ".".join(part for part in (*package_parts, node.module) if part)


def _relative_package_parts(node: ast.ImportFrom, source: str, path: Path) -> tuple[str, ...]:
    """Return the package components a relative import counts back from."""
    if not node.level:
        return ()
    source_package = source if path.name == "__init__.py" else source.rpartition(".")[0]
    parts = source_package.split(".")
    # A relative import deeper than its own package counts back to no package.
    return tuple(parts[: max(0, len(parts) - node.level + 1)])


def _resolve_local(name: str, module_paths: dict[str, Path]) -> str | None:
    if not name:
        return None
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in module_paths:
            return candidate
    return None


def _sorted_edges(graph: dict[str, tuple[ImportEdge, ...]]) -> list[ImportEdge]:
    return sorted(
        (edge for edges in graph.values() for edge in edges),
        # Each discovered source module has one path; source already orders it.
        key=lambda edge: (edge.source, edge.target, edge.line),
    )


def _cycle_violations(
    graph: dict[str, tuple[ImportEdge, ...]],
) -> list[ArchitectureViolation]:
    predecessors = {
        module: tuple(sorted({edge.target for edge in graph[module]})) for module in sorted(graph)
    }
    try:
        TopologicalSorter(predecessors).prepare()
    except CycleError as error:
        return [_cycle_violation(error, graph)]
    return []


def _cycle_violation(
    error: CycleError,
    graph: dict[str, tuple[ImportEdge, ...]],
) -> ArchitectureViolation:
    cycle = tuple(str(module) for module in error.args[1])
    pairs = set(pairwise(cycle))
    edge = next(edge for edge in _sorted_edges(graph) if (edge.source, edge.target) in pairs)
    return ArchitectureViolation(
        "cycle",
        edge.source,
        edge.target,
        edge.path,
        edge.line,
        " -> ".join(cycle),
    )


def _boundary_violations(
    graph: dict[str, tuple[ImportEdge, ...]], package: str
) -> list[ArchitectureViolation]:
    violations: list[ArchitectureViolation] = []
    for edge in _sorted_edges(graph):
        violation = _boundary_violation(edge, package)
        if violation is not None:
            violations.append(violation)
    return violations


def _boundary_violation(edge: ImportEdge, package: str) -> ArchitectureViolation | None:
    source_layer = _local_layer(edge.source, package)
    target_layer = _local_layer(edge.target, package)
    if source_layer == "domain" and target_layer != "domain":
        return ArchitectureViolation(
            "domain-purity",
            edge.source,
            edge.target,
            edge.path,
            edge.line,
            "domain may import only local domain modules; external imports are not represented",
        )
    if source_layer == "pipeline" and target_layer == "cli":
        return ArchitectureViolation(
            "pipeline-cli",
            edge.source,
            edge.target,
            edge.path,
            edge.line,
            "pipeline must not import CLI modules",
        )
    return None


def _local_layer(module: str, package: str) -> str | None:
    prefix = f"{package}."
    if not module.startswith(prefix):
        return None
    return module[len(prefix) :].partition(".")[0]


if __name__ == "__main__":
    raise SystemExit(main())
