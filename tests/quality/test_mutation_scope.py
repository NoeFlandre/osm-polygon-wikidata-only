"""Contracts for the single source-to-test mutation scope."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mutmut.configuration
import pytest

from scripts.quality import mutation_scope

REPOSITORY = Path(__file__).resolve().parents[2]


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_scope_derives_source_paths_and_deduplicated_test_selectors(tmp_path: Path) -> None:
    mapping = {
        "src/first.py": ["tests/first.py", "tests/shared.py::test_shared"],
        "src/second.py": ["tests/second.py", "tests/shared.py::test_shared"],
    }
    for relative in (
        "src/first.py",
        "src/second.py",
        "tests/first.py",
        "tests/second.py",
        "tests/shared.py",
    ):
        _touch(tmp_path, relative)

    scope = mutation_scope.build_scope(mapping, root=tmp_path)

    assert scope.source_paths == ("src/first.py", "src/second.py")
    assert scope.test_selectors == (
        "tests/first.py",
        "tests/shared.py::test_shared",
        "tests/second.py",
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"src/missing.py": ["tests/test_present.py"]}, "source.*does not exist"),
        ({"src/present.py": []}, "must select at least one test"),
        ({"src/present.py": ["tests/test_missing.py"]}, "test selector.*does not exist"),
    ],
)
def test_scope_rejects_missing_or_unmapped_paths(
    tmp_path: Path, mapping: dict[str, list[str]], message: str
) -> None:
    _touch(tmp_path, "src/present.py")
    _touch(tmp_path, "tests/test_present.py")

    with pytest.raises(mutation_scope.MutationScopeError, match=message):
        mutation_scope.build_scope(mapping, root=tmp_path)


@pytest.mark.parametrize(
    ("source", "selector", "message"),
    [
        (None, "tests/test_present.py", "source path must be a string"),
        ("src/present.py", "README.md", "relative Python file"),
        ("src/present.py", "src/present.py", "under tests/"),
    ],
)
def test_scope_rejects_invalid_source_test_pairs(
    tmp_path: Path, source: object, selector: str, message: str
) -> None:
    _touch(tmp_path, "src/present.py")
    _touch(tmp_path, "tests/test_present.py")
    _touch(tmp_path, "README.md")

    with pytest.raises(mutation_scope.MutationScopeError, match=message):
        mutation_scope.build_scope({source: [selector]}, root=tmp_path)


def test_scope_rejects_absolute_and_traversing_paths_that_resolve_inside_repo(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "src/present.py")
    _touch(tmp_path, "tests/test_present.py")

    with pytest.raises(mutation_scope.MutationScopeError, match="relative Python file"):
        mutation_scope.build_scope(
            {str(tmp_path / "src/present.py"): ["tests/test_present.py"]}, root=tmp_path
        )

    traversal = f"../{tmp_path.name}/tests/test_present.py"
    with pytest.raises(mutation_scope.MutationScopeError, match="relative Python file"):
        mutation_scope.build_scope({"src/present.py": [traversal]}, root=tmp_path)


def test_scope_rejects_test_selectors_that_pytest_cannot_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = mutation_scope.MutationScope(
        source_paths=("src/present.py",), test_selectors=("tests/test_present.py::test_missing",)
    )
    completed = subprocess.CompletedProcess(args=[], returncode=4)
    monkeypatch.setattr(mutation_scope.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(mutation_scope.MutationScopeError, match="pytest collection failed"):
        mutation_scope.validate_collection(scope, root=tmp_path)


def test_scope_validates_collection_with_derived_test_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = mutation_scope.MutationScope(
        source_paths=("src/present.py",),
        test_selectors=("tests/test_present.py", "tests/test_other.py::test_case"),
    )
    completed = subprocess.CompletedProcess(args=[], returncode=0)
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        mutation_scope.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or completed,
    )

    mutation_scope.validate_collection(scope, root=tmp_path)

    command, options = calls[0]
    assert command[1:8] == [
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--no-cov",
    ]
    assert command[8:] == list(scope.test_selectors)
    assert options["cwd"] == tmp_path
    assert options["check"] is False


def test_scope_reports_when_pytest_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = mutation_scope.MutationScope(
        source_paths=("src/present.py",), test_selectors=("tests/test_present.py",)
    )
    monkeypatch.setattr(
        mutation_scope.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing python")),
    )

    with pytest.raises(
        mutation_scope.MutationScopeError, match="Could not start pytest collection"
    ):
        mutation_scope.validate_collection(scope, root=tmp_path)


def test_mutmut_receives_paths_derived_from_the_mapping(tmp_path: Path) -> None:
    scope = mutation_scope.MutationScope(
        source_paths=("src/first.py", "scripts/second.py"),
        test_selectors=("tests/first.py", "tests/second.py::test_second"),
    )

    try:
        mutation_scope.configure_mutmut(scope, root=tmp_path)
        configured = mutmut.configuration.config()

        assert configured.source_paths == [Path("src/first.py"), Path("scripts/second.py")]
        assert configured.resolved_mutated_source_paths == [
            tmp_path.resolve() / "mutants/src/first.py",
            tmp_path.resolve() / "mutants/scripts/second.py",
        ]
        assert configured.pytest_add_cli_args_test_selection == list(scope.test_selectors)
    finally:
        mutmut.configuration.reset_config()


def test_load_scope_reports_missing_configuration(tmp_path: Path) -> None:
    with pytest.raises(mutation_scope.MutationScopeError, match="Could not read"):
        mutation_scope.load_scope(tmp_path / "missing.toml", root=tmp_path)


def test_run_mutation_validates_before_cleaning_and_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scope = mutation_scope.MutationScope(("src/example.py",), ("tests/test_example.py",))
    calls: list[tuple[str, object, object]] = []
    monkeypatch.setattr(mutation_scope, "load_scope", lambda pyproject, root: scope)
    monkeypatch.setattr(
        mutation_scope,
        "validate_collection",
        lambda current, root: calls.append(("collect", current, root)),
    )
    monkeypatch.setattr(
        mutation_scope.shutil,
        "rmtree",
        lambda path, ignore_errors: calls.append(("remove", path, ignore_errors)),
    )
    monkeypatch.setattr(
        mutation_scope,
        "configure_mutmut",
        lambda current, root: calls.append(("configure", current, root)),
    )
    monkeypatch.setattr(
        mutation_scope,
        "mutmut_cli",
        lambda args, standalone_mode: calls.append(("mutmut", args, standalone_mode)),
    )

    mutation_scope.run_mutation(max_children=3, root=tmp_path)

    assert [call[0] for call in calls] == ["collect", "remove", "configure", "mutmut"]
    assert calls[-1] == ("mutmut", ["run", "--max-children", "3"], False)
    assert "Validated 1 mutation sources and 1 test selectors" in capsys.readouterr().out


def test_run_results_configures_derived_scope_before_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = mutation_scope.MutationScope(("src/example.py",), ("tests/test_example.py",))
    calls: list[tuple[str, object, object]] = []
    monkeypatch.setattr(mutation_scope, "load_scope", lambda pyproject, root: scope)
    monkeypatch.setattr(
        mutation_scope,
        "configure_mutmut",
        lambda current, root: calls.append(("configure", current, root)),
    )
    monkeypatch.setattr(
        mutation_scope,
        "mutmut_cli",
        lambda args, standalone_mode: calls.append(("mutmut", args, standalone_mode)),
    )

    mutation_scope.run_results(["--all=true"], root=tmp_path)

    assert [call[0] for call in calls] == ["configure", "mutmut"]
    assert calls[-1] == ("mutmut", ["results", "--all=true"], False)


def test_main_runs_the_requested_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, Path]] = []
    monkeypatch.setattr(
        mutation_scope,
        "run_mutation",
        lambda *, max_children, root: calls.append((max_children, root)),
    )

    assert mutation_scope.main(["--max-children", "4"]) == 0
    assert calls == [(4, Path.cwd())]


def test_main_routes_results_to_configured_mutmut(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        mutation_scope,
        "run_results",
        lambda arguments, *, root: calls.append((arguments, root)),
    )

    assert mutation_scope.main(["results", "--all=true"]) == 0
    assert calls == [(["--all=true"], Path.cwd())]


def test_main_rejects_non_positive_worker_counts() -> None:
    with pytest.raises(SystemExit, match="2"):
        mutation_scope.main(["--max-children", "0"])
