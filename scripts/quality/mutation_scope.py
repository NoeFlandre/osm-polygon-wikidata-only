"""Derive and validate mutmut inputs from one source-to-test mapping."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mutmut.configuration as mutmut_configuration
from mutmut.__main__ import cli as mutmut_cli


class MutationScopeError(ValueError):
    """The configured mutation source-to-test mapping is invalid."""


@dataclass(frozen=True)
class MutationScope:
    """The source files and unique pytest selectors derived from one mapping."""

    source_paths: tuple[str, ...]
    test_selectors: tuple[str, ...]


def _repository_file(root: Path, relative: object, *, kind: str) -> Path:
    if not isinstance(relative, str):
        raise MutationScopeError(f"Mutation {kind} path must be a string: {relative!r}")
    path = Path(relative)
    repository = root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(repository) or resolved.suffix != ".py":
        raise MutationScopeError(
            f"Mutation {kind} path must be a relative Python file: {relative!r}"
        )
    if not resolved.is_file():
        raise MutationScopeError(f"Mutation {kind} path does not exist: {relative}")
    return path


def _validate_test_selector(root: Path, source: str, selector: object) -> str:
    if not isinstance(selector, str) or not selector.strip():
        raise MutationScopeError(
            f"Mutation source {source} has an invalid test selector: {selector!r}"
        )
    path = _repository_file(root, selector.partition("::")[0], kind="test selector")
    if not path.parts or path.parts[0] != "tests":
        raise MutationScopeError(f"Mutation test selector must be under tests/: {selector}")
    return selector


def _source_test_selectors(root: Path, source: str, selectors: object) -> tuple[str, ...]:
    if not isinstance(selectors, list) or not selectors:
        raise MutationScopeError(f"Mutation source {source} must select at least one test")
    return tuple(_validate_test_selector(root, source, selector) for selector in selectors)


def build_scope(source_to_tests: object, *, root: Path) -> MutationScope:
    """Validate mapping paths and derive stable source and test lists."""
    if not isinstance(source_to_tests, dict) or not source_to_tests:
        raise MutationScopeError("tool.mutmut.source_to_tests must be a non-empty table")

    source_paths: list[str] = []
    test_selectors: dict[str, None] = {}
    for source, selectors in source_to_tests.items():
        source_path = _repository_file(root, source, kind="source")
        source_paths.append(source_path.as_posix())
        for selector in _source_test_selectors(root, source, selectors):
            test_selectors.setdefault(selector, None)

    return MutationScope(tuple(source_paths), tuple(test_selectors))


def load_scope(pyproject: Path, *, root: Path) -> MutationScope:
    """Read and validate the mutation mapping from pyproject.toml."""
    try:
        config: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        source_to_tests = config["tool"]["mutmut"]["source_to_tests"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        raise MutationScopeError(
            f"Could not read tool.mutmut.source_to_tests from {pyproject}"
        ) from error
    return build_scope(source_to_tests, root=root)


def validate_collection(scope: MutationScope, *, root: Path) -> None:
    """Fail before mutation if pytest cannot collect the selected tests."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--no-cov",
        *scope.test_selectors,
    ]
    try:
        result = subprocess.run(command, cwd=root, check=False)  # noqa: S603
    except OSError as error:
        raise MutationScopeError("Could not start pytest collection for mutation tests") from error
    if result.returncode:
        raise MutationScopeError(
            f"pytest collection failed for mutation tests (exit {result.returncode})"
        )


def configure_mutmut(scope: MutationScope, *, root: Path) -> None:
    """Inject derived paths into mutmut's pinned runtime configuration."""
    current = mutmut_configuration.config()
    repository = root.resolve(strict=True)
    mutmut_configuration._config = replace(
        current,
        source_paths=[Path(path) for path in scope.source_paths],
        resolved_mutated_source_paths=[
            repository / "mutants" / path for path in map(Path, scope.source_paths)
        ],
        pytest_add_cli_args_test_selection=list(scope.test_selectors),
    )


def run_mutation(*, max_children: int, root: Path) -> None:
    """Validate scope, clear stale mutants, then run mutmut with derived inputs."""
    scope = load_scope(root / "pyproject.toml", root=root)
    validate_collection(scope, root=root)
    print(
        f"Validated {len(scope.source_paths)} mutation sources and {len(scope.test_selectors)} test selectors"
    )
    shutil.rmtree(root / "mutants", ignore_errors=True)
    configure_mutmut(scope, root=root)

    mutmut_cli(["run", "--max-children", str(max_children)], standalone_mode=False)


def run_results(arguments: list[str], *, root: Path) -> None:
    """Print mutmut results using the same derived scope as the mutation run."""
    scope = load_scope(root / "pyproject.toml", root=root)
    configure_mutmut(scope, root=root)
    mutmut_cli(["results", *arguments], standalone_mode=False)


def main(argv: list[str] | None = None) -> int:
    """Run the mutation scope used by the Justfile mutation gate."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "results":
        run_results(arguments[1:], root=Path.cwd())
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-children", type=int, required=True)
    args = parser.parse_args(arguments)
    if args.max_children < 1:
        parser.error("--max-children must be positive")
    run_mutation(max_children=args.max_children, root=Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
