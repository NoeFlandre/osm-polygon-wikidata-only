"""Derive and validate mutmut inputs from one source-to-test mapping."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import mutmut.configuration as mutmut_configuration
from mutmut.__main__ import cli as mutmut_cli


class MutationScopeError(ValueError):
    """The configured mutation source-to-test mapping is invalid."""


@dataclass(frozen=True)
class MutationScope:
    """The source files and unique pytest selectors derived from one mapping."""

    source_paths: tuple[str, ...]
    test_selectors: tuple[str, ...]


def _is_safe_repository_python_file(path: Path, resolved: Path, repository: Path) -> bool:
    """Return whether a relative path resolves to a Python file inside the repository."""
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and resolved.is_relative_to(repository)
        and resolved.suffix == ".py"
    )


def _repository_file(root: Path, relative: object, *, kind: str) -> Path:
    if not isinstance(relative, str):
        raise MutationScopeError(f"Mutation {kind} path must be a string: {relative!r}")
    path = Path(relative)
    repository = root.resolve()
    resolved = (root / path).resolve()
    if not _is_safe_repository_python_file(path, resolved, repository):
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
    _run_mutmut_in_subprocess(max_children=max_children, root=root)


def _run_mutmut_in_subprocess(*, max_children: int, root: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_mutmut-worker",
        "--max-children",
        str(max_children),
    ]
    try:
        result = subprocess.run(command, cwd=root, check=False)  # noqa: S603
    except OSError as error:
        raise MutationScopeError("Could not start mutmut worker process") from error
    if result.returncode:
        raise MutationScopeError(f"mutmut worker process failed (exit {result.returncode})")


def _exit_mutmut_worker(
    exit_code: int, *, exit_process: Callable[[int], NoReturn] = os._exit
) -> NoReturn:
    """Flush mutmut output, then exit before Python waits on leftover threads."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        exit_process(exit_code)


def _system_exit_code(error: SystemExit) -> int:
    if error.code is None:
        return 0
    if isinstance(error.code, int):
        return error.code
    print(error.code, file=sys.stderr)
    return 1


def _run_mutmut_cli(
    arguments: list[str], *, exit_process: Callable[[int], NoReturn] = os._exit
) -> NoReturn:
    """Run mutmut in its dedicated worker and exit without interpreter teardown."""
    exit_code = 0
    try:
        mutmut_cli(arguments, standalone_mode=False)
    except SystemExit as error:
        exit_code = _system_exit_code(error)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:  # noqa: BLE001 - exit the worker after reporting its failure
        traceback.print_exc()
        exit_code = 1
    _exit_mutmut_worker(exit_code, exit_process=exit_process)


def _run_mutmut_worker_command(arguments: list[str]) -> NoReturn:
    parser = argparse.ArgumentParser(description="Run mutmut in an isolated worker process")
    parser.add_argument("--max-children", type=int, required=True)
    args = parser.parse_args(arguments)
    root = Path.cwd()
    scope = load_scope(root / "pyproject.toml", root=root)
    configure_mutmut(scope, root=root)
    _run_mutmut_cli(["run", "--max-children", str(args.max_children)])


def run_results(arguments: list[str], *, root: Path) -> None:
    """Print mutmut results using the same derived scope as the mutation run."""
    scope = load_scope(root / "pyproject.toml", root=root)
    configure_mutmut(scope, root=root)
    mutmut_cli(["results", *arguments], standalone_mode=False)


def main(argv: list[str] | None = None) -> int:
    """Run the mutation scope used by the Justfile mutation gate."""
    arguments = sys.argv[1:] if argv is None else argv
    return _main(arguments)


def _main(arguments: list[str]) -> int:
    if arguments[:1] == ["_mutmut-worker"]:
        return _run_mutmut_worker_command(arguments[1:])
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
