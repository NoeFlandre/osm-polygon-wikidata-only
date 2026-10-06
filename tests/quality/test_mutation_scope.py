"""Contracts for the single source-to-test mutation scope."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import NoReturn

import mutmut.configuration
import pytest

from scripts.quality import mutation_scope

REPOSITORY = Path(__file__).resolve().parents[2]


class _WorkerExit(Exception):
    def __init__(self, exit_code: int) -> None:
        super().__init__(exit_code)
        self.exit_code = exit_code


def _raise_worker_exit(exit_code: int) -> NoReturn:
    raise _WorkerExit(exit_code)


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


def test_run_mutation_validates_before_cleaning_and_running_in_a_child_process(
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
        mutation_scope.subprocess,
        "run",
        lambda command, **options: (
            calls.append(("mutmut", command, options)) or subprocess.CompletedProcess(command, 0)
        ),
    )
    monkeypatch.setattr(
        mutation_scope,
        "mutmut_cli",
        lambda *args, **kwargs: pytest.fail("mutmut must run in a child process"),
    )

    mutation_scope.run_mutation(max_children=3, root=tmp_path)

    assert [call[0] for call in calls] == ["collect", "remove", "mutmut"]
    assert calls[-1] == (
        "mutmut",
        [
            mutation_scope.sys.executable,
            str(Path(mutation_scope.__file__).resolve()),
            "_mutmut-worker",
            "--max-children",
            "3",
        ],
        {"cwd": tmp_path, "check": False},
    )
    assert "Validated 1 mutation sources and 1 test selectors" in capsys.readouterr().out


def test_mutmut_worker_exits_after_cli_returns_with_a_live_thread() -> None:
    script = "\n".join(
        [
            "from threading import Event, Thread",
            "from scripts.quality import mutation_scope",
            "mutation_scope.mutmut_cli = lambda *args, **kwargs: Thread(",
            "    target=Event().wait, daemon=False).start()",
            "mutation_scope._run_mutmut_cli(['run'])",
        ]
    )

    result = subprocess.run(
        [mutation_scope.sys.executable, "-c", script],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [(SystemExit(7), 7), (KeyboardInterrupt(), 130), (RuntimeError("worker failed"), 1)],
)
def test_mutmut_cli_exits_the_worker_with_its_status(
    failure: BaseException,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(mutation_scope, "mutmut_cli", fail)

    with pytest.raises(_WorkerExit) as captured:
        mutation_scope._run_mutmut_cli(["run"], exit_process=_raise_worker_exit)

    assert captured.value.exit_code == exit_code
    if isinstance(failure, RuntimeError):
        assert "RuntimeError: worker failed" in capsys.readouterr().err


def test_mutmut_worker_configures_scope_and_invokes_cli(tmp_path: Path) -> None:
    marker = tmp_path / "worker.json"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["src"]\n', encoding="utf-8"
    )
    script = "\n".join(
        [
            "import json, sys",
            "from pathlib import Path",
            "sys.path.insert(0, sys.argv[1])",
            "import mutmut.configuration as configuration",
            "from scripts.quality import mutation_scope",
            "scope = mutation_scope.MutationScope(",
            "    ('src/tiny.py',), ('tests/test_tiny.py',))",
            "mutation_scope.load_scope = lambda *args, **kwargs: scope",
            "mutation_scope.mutmut_cli = lambda args, standalone_mode: Path(",
            "    sys.argv[2]).write_text(json.dumps((args, standalone_mode,",
            "    [str(path) for path in configuration.config().source_paths],",
            "    configuration.config().pytest_add_cli_args_test_selection)))",
            "mutation_scope.main(['_mutmut-worker', '--max-children', '2'])",
        ]
    )
    result = subprocess.run(
        [mutation_scope.sys.executable, "-c", script, str(REPOSITORY), str(marker)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(marker.read_text(encoding="utf-8")) == [
        ["run", "--max-children", "2"],
        False,
        ["src/tiny.py"],
        ["tests/test_tiny.py"],
    ]


@pytest.mark.parametrize(("code", "expected"), [(None, 0), (7, 7), ("mutation failed", 1)])
def test_mutmut_worker_preserves_click_exit_codes(
    code: object, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mutation_scope._system_exit_code(SystemExit(code)) == expected
    if isinstance(code, str):
        assert "mutation failed" in capsys.readouterr().err


def test_run_mutmut_in_subprocess_reports_worker_start_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mutation_scope.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("python unavailable")),
    )

    with pytest.raises(mutation_scope.MutationScopeError, match="Could not start mutmut worker"):
        mutation_scope._run_mutmut_in_subprocess(max_children=2, root=tmp_path)


def test_run_mutmut_in_subprocess_reports_worker_exit_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mutation_scope.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7),
    )

    with pytest.raises(
        mutation_scope.MutationScopeError, match=r"worker process failed \(exit 7\)"
    ):
        mutation_scope._run_mutmut_in_subprocess(max_children=2, root=tmp_path)


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
