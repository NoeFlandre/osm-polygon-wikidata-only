"""Validate installed package metadata and resources without optional dependencies."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Sequence
from importlib import metadata, resources
from importlib.resources.abc import Traversable


class PackageSmokeError(RuntimeError):
    """Raised when an installed distribution is missing a declared artifact."""


def verify_resources(package_root: Traversable, resource_paths: Sequence[str]) -> None:
    """Require every declared resource to exist inside an installed package."""
    missing = [
        path for path in resource_paths if not package_root.joinpath(*path.split("/")).is_file()
    ]
    if missing:
        raise PackageSmokeError(f"Missing packaged resources: {', '.join(missing)}")


def _missing_entry_points(
    available: dict[str, metadata.EntryPoint],
    expected_names: Sequence[str],
) -> list[str]:
    return [name for name in expected_names if name not in available]


def _malformed_entry_points(
    available: dict[str, metadata.EntryPoint],
    expected_names: Sequence[str],
) -> list[str]:
    return [
        f"{name}={available[name].value}"
        for name in expected_names
        if ":" not in available[name].value
    ]


def verify_entry_points(
    entry_points: Iterable[metadata.EntryPoint],
    expected_names: Sequence[str],
) -> None:
    """Require named console entry points to have importable target syntax."""
    available = {entry_point.name: entry_point for entry_point in entry_points}
    missing = _missing_entry_points(available, expected_names)
    if missing:
        raise PackageSmokeError(f"Missing console entry points: {', '.join(missing)}")

    malformed = _malformed_entry_points(available, expected_names)
    if malformed:
        raise PackageSmokeError(f"Malformed console entry points: {', '.join(malformed)}")


def _module_candidate_parts(
    relative_parts: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if not relative_parts:
        return (("__init__.py",),)
    return (
        (*relative_parts[:-1], f"{relative_parts[-1]}.py"),
        (*relative_parts, "__init__.py"),
    )


def _module_resource(
    package_root: Traversable, package_name: str, module_name: str
) -> Traversable | None:
    package_parts = tuple(package_name.split("."))
    module_parts = tuple(module_name.split("."))
    if module_parts[: len(package_parts)] != package_parts:
        return None

    relative_parts = module_parts[len(package_parts) :]
    for candidate_parts in _module_candidate_parts(relative_parts):
        candidate = package_root.joinpath(*candidate_parts)
        if candidate.is_file():
            return candidate
    return None


def _names_from_targets(targets: Iterable[ast.expr]) -> set[str]:
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _assignment_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return _names_from_targets(node.targets)
    if isinstance(node, ast.AnnAssign):
        return _names_from_targets((node.target,))
    return set()


def _import_alias_name(alias: ast.alias) -> str:
    if alias.asname is not None:
        return alias.asname
    return alias.name.split(".")[0]


def _from_import_names(node: ast.ImportFrom) -> set[str]:
    return {_import_alias_name(alias) for alias in node.names if alias.name != "*"}


def _import_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {_import_alias_name(alias) for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return set()
    return _from_import_names(node)


def _node_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
        return {node.name}
    return _assignment_names(node) | _import_names(node)


def _top_level_names(source: str, module_name: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=module_name)
    except SyntaxError as error:
        raise PackageSmokeError(f"Invalid entry-point module: {module_name}") from error

    return set().union(*(_node_names(node) for node in tree.body))


def _read_entry_point_source(module_file: Traversable, entry_point: metadata.EntryPoint) -> str:
    try:
        return module_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PackageSmokeError(
            f"Entry point module unreadable: {entry_point.name}={entry_point.value}"
        ) from error


def _verify_entry_point_target(
    package_root: Traversable,
    package_name: str,
    entry_point: metadata.EntryPoint,
) -> None:
    module_file = _module_resource(package_root, package_name, entry_point.module)
    if module_file is None:
        raise PackageSmokeError(
            f"Entry point module unavailable: {entry_point.name}={entry_point.value}"
        )
    source = _read_entry_point_source(module_file, entry_point)
    target_name = (entry_point.attr or "").partition(".")[0]
    if target_name not in _top_level_names(source, entry_point.module):
        raise PackageSmokeError(
            f"Entry point target unavailable: {entry_point.name}={entry_point.value}"
        )


def _verify_entry_point_targets(
    package_root: Traversable,
    package_name: str,
    available: dict[str, metadata.EntryPoint],
    expected_names: Sequence[str],
) -> None:
    for name in expected_names:
        _verify_entry_point_target(package_root, package_name, available[name])


def verify_distribution(
    distribution_name: str,
    package_name: str,
    resource_paths: Sequence[str],
    entry_point_names: Sequence[str],
) -> None:
    """Validate one installed distribution against its artifact contract."""
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise PackageSmokeError(f"Installed distribution not found: {distribution_name}") from error

    try:
        package_root = resources.files(package_name)
    except ModuleNotFoundError as error:
        raise PackageSmokeError(f"Installed package not found: {package_name}") from error

    verify_entry_points(distribution.entry_points, entry_point_names)
    available = {entry_point.name: entry_point for entry_point in distribution.entry_points}
    _verify_entry_point_targets(package_root, package_name, available, entry_point_names)
    verify_resources(package_root, resource_paths)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed-artifact smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--entry-point", action="append", default=[])
    arguments = parser.parse_args(argv)

    try:
        verify_distribution(
            arguments.distribution,
            arguments.package,
            tuple(arguments.resource),
            tuple(arguments.entry_point),
        )
    except PackageSmokeError as error:
        print(f"Package smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
