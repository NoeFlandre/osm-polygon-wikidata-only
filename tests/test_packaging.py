"""Public distribution metadata and typing marker tests."""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path


def test_project_metadata_is_public_ready() -> None:
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata["authors"] == [{"name": "Noé Flandre"}]
    assert metadata["license"] == {"file": "LICENSE"}
    assert metadata["urls"]["Source"].endswith("NoeFlandre/osm-polygon-wikidata-only")
    assert "Programming Language :: Python :: 3.12" in metadata["classifiers"]
    assert metadata["scripts"]["osm-polygon-wikidata-only"].endswith(":run")
    description = metadata["description"].lower()
    assert all(source in description for source in ("osm", "wikidata", "wikipedia", "wikivoyage"))
    assert "wikivoyage" in metadata["keywords"]


def test_package_declares_inline_typing_support() -> None:
    marker = Path(__file__).parents[1] / "src/osm_polygon_wikidata_only/py.typed"
    assert marker.is_file()


def test_hatch_force_includes_reference_existing_public_resources() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    for source, target in force_include.items():
        assert (root / source).is_file(), source
        assert target.startswith("osm_polygon_wikidata_only/"), target


def test_project_uses_ty_as_its_only_static_type_checker() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    development = config["dependency-groups"]["dev"]

    assert "ty==0.0.64" in development
    assert not any(dependency.startswith("mypy") for dependency in development)
    assert "mypy" not in config["tool"]

    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    justfile = (root / "Justfile").read_text(encoding="utf-8")
    assert "run: just quality-gauntlet" in workflow
    assert "uv run ty check src scripts" in justfile
    assert "uv run mypy" not in workflow


def test_project_declares_operator_and_quality_tooling_directly() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_names = {
        dependency.split("=", 1)[0].split(">", 1)[0]
        for dependency in config["project"]["dependencies"]
    }
    development_names = {
        dependency.split("=", 1)[0].split(">", 1)[0]
        for dependency in config["dependency-groups"]["dev"]
    }

    assert len(config["dependency-groups"]["dev"]) == len(set(config["dependency-groups"]["dev"]))
    assert {"typer", "rich", "tqdm", "trackio"} <= runtime_names
    assert {
        "mutmut",
        "pytest",
        "pytest-cov",
        "ruff",
        "ty",
        "pre-commit",
    } <= development_names
    assert (
        config["project"]["scripts"]["osm-polygon-wikidata-only-audit-remote"]
        == "osm_polygon_wikidata_only.cli.audit_remote:run"
    )
    assert (
        config["project"]["scripts"]["osm-polygon-wikidata-only-trackio"]
        == "osm_polygon_wikidata_only.hf.trackio_snapshot:run"
    )
    assert config["project"]["optional-dependencies"]["sentence-splitting"] == [
        "wtpsplit[onnx-cpu]==2.2.1"
    ]
    assert config["project"]["optional-dependencies"]["sentence-splitting-gpu"] == [
        "wtpsplit[onnx-gpu]==2.2.1",
        "onnxruntime-gpu[cuda,cudnn]==1.29.0",
    ]


def test_justfile_is_the_uv_managed_quality_command_catalog() -> None:
    root = Path(__file__).parents[1]
    listed = subprocess.run(
        ["just", "--list"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stderr
    for recipe in (
        "sync",
        "test",
        "coverage",
        "baseline",
        "ruff",
        "lint",
        "tests",
        "property-tests",
        "acceptance-tests",
        "architecture-checks",
        "format",
        "format-check",
        "typecheck",
        "ty",
        "build",
        "package-smoke",
        "preprocessing-package-smoke",
        "docs",
        "trackio",
        "mutation",
        "crap",
        "crap-report",
        "crap-quality",
        "crap-all",
        "crap-preprocessing",
        "crap-upload",
        "quality-strength",
        "smoke-test",
        "diff-review",
        "qa-gauntlet",
        "quality-gauntlet",
        "check",
    ):
        assert recipe in listed.stdout

    for recipe in (
        "baseline",
        "ruff",
        "ty",
        "tests",
        "property-tests",
        "acceptance-tests",
        "architecture-checks",
        "crap-all",
        "mutation",
        "smoke-test",
        "diff-review",
    ):
        rendered = subprocess.run(
            ["just", "--dry-run", recipe],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rendered.returncode == 0, rendered.stderr

    justfile = (root / "Justfile").read_text(encoding="utf-8")
    assert "scripts/quality/qa_gauntlet.py" in justfile
    assert "mypy" not in justfile


def test_github_actions_runs_the_canonical_gauntlet_once() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert workflow.count("run: just quality-gauntlet") == 1
    assert "run: just qa-gauntlet" not in workflow
    assert "run: just quality-strength" not in workflow


def test_pre_commit_runs_fast_uv_managed_quality_hooks() -> None:
    root = Path(__file__).parents[1]
    config = (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "repo: local" in config
    assert "uv run ruff check src tests scripts" in config
    assert "uv run ruff format --check src tests scripts" in config
    assert "uv run ty check src scripts" in config
    assert "mypy" not in config


def test_github_actions_delegates_quality_commands_to_just() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "taiki-e/install-action@just" in workflow
    assert "run: just quality-gauntlet" in workflow
    for recipe in ("coverage", "lint", "format-check", "typecheck", "build"):
        assert f"run: just {recipe}" not in workflow
    assert "uv sync --frozen" not in workflow


def test_quality_gauntlet_is_the_single_canonical_completion_gate() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "quality-gauntlet:" in justfile
    assert "check: quality-gauntlet" in justfile
    assert "just build" in justfile
    assert "just docs" in justfile
    assert "docker-help" in justfile
    assert workflow.count("run: just quality-gauntlet") == 1
    assert "run: just qa-gauntlet" not in workflow


def test_coverage_recipes_use_configured_runtime_paths(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    runtime = tmp_path / "quality runtime"
    environment = os.environ.copy()
    environment["QUALITY_RUNTIME_DIR"] = str(runtime)
    environment["TMPDIR"] = "/var/folders/hostile"
    for name in (
        "QUALITY_REPORT_DIR",
        "QUALITY_CACHE_DIR",
        "QUALITY_TMP_DIR",
        "UV_CACHE_DIR",
    ):
        environment.pop(name, None)
    rendered = subprocess.run(
        ["just", "--dry-run", "tests"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rendered.returncode == 0, rendered.stderr
    output = rendered.stdout + rendered.stderr
    assert str(runtime / "reports" / "coverage.json") in output
    assert str(runtime / "tmp" / "tests-pytest") in output
    assert "/var/folders/hostile" not in output


def test_crap_report_covers_root_and_preprocessing_sources(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    runtime = tmp_path / "quality runtime"
    environment = os.environ.copy()
    environment["QUALITY_RUNTIME_DIR"] = str(runtime)
    for name in (
        "QUALITY_REPORT_DIR",
        "QUALITY_CACHE_DIR",
        "QUALITY_TMP_DIR",
        "UV_CACHE_DIR",
    ):
        environment.pop(name, None)
    rendered = subprocess.run(
        ["just", "--dry-run", "crap-report"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rendered.returncode == 0, rendered.stderr
    output = rendered.stdout + rendered.stderr
    assert "radon cc --show-closures -j src scripts" in output
    assert "radon cc --show-closures -j preprocessing/src" in output
    assert str(runtime / "reports" / "coverage.json") in output
    assert str(runtime / "reports" / "preprocessing-coverage.json") in output
    assert "crap4py" not in output


def test_crap_all_refreshes_reports_before_running_the_canonical_reporter() -> None:
    root = Path(__file__).parents[1]
    rendered = subprocess.run(
        ["just", "--dry-run", "crap-all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rendered.returncode == 0, rendered.stderr
    output = rendered.stdout + rendered.stderr
    assert output.index("just tests") < output.index("just preprocessing-check")
    assert output.index("just preprocessing-check") < output.index("just crap-report")
    assert "crap4py" not in output


def test_grid5000_protocol_is_in_the_pure_quality_scopes() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    protocol_source = "src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py"
    protocol_tests = "tests/grid5000/test_sentence_protocol.py"
    artifact_tests = "tests/grid5000/test_sentence_artifacts.py"
    assert protocol_source in mutation["source_paths"]
    assert protocol_tests in mutation["pytest_add_cli_args_test_selection"]
    assert artifact_tests in mutation["pytest_add_cli_args_test_selection"]


def test_file_boundary_refactors_stay_out_of_mutation_scope() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    for source_path, test_path in (
        (
            "src/osm_polygon_wikidata_only/v2/comparison.py",
            "tests/v2/test_comparison.py",
        ),
        (
            "src/osm_polygon_wikidata_only/v2/sentence_checkpoints.py",
            "tests/v2/test_sentence_checkpoints.py",
        ),
        (
            "src/osm_polygon_wikidata_only/hf/_geographic/parquet_inputs.py",
            "tests/hf/test_geographic_text_coverage.py",
        ),
        (
            "src/osm_polygon_wikidata_only/hf/_dataset_stats/aggregation.py",
            "tests/hf/test_dataset_stats.py",
        ),
        ("src/osm_polygon_wikidata_only/v2/sat.py", "tests/v2/test_sat.py"),
        # `io/atomic.py` is the shared temp-sibling + `os.replace` ritual. Its
        # remaining mutable surface is keyword values that the standard library
        # and pyarrow normalize themselves -- `"UTF-8"` for `"utf-8"`, an
        # omitted `compression` for pyarrow's own snappy default -- so those
        # mutants are equivalent and no test can kill them. It stays under the
        # `crap-atomic` scope instead.
        ("src/osm_polygon_wikidata_only/io/atomic.py", "tests/io/test_atomic.py"),
    ):
        assert source_path not in mutation["source_paths"]
        assert test_path not in mutation["pytest_add_cli_args_test_selection"]


def test_upload_retry_policy_is_in_mutation_scope() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    assert "src/osm_polygon_wikidata_only/hf/_upload_retry.py" in mutation["source_paths"]
    assert (
        "tests/hf/test_upload_operation_helpers.py"
        in mutation["pytest_add_cli_args_test_selection"]
    )


def test_diff_review_executes_unmerged_path_check() -> None:
    """The diff-review recipe must evaluate, not quote, its command substitution."""

    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    assert 'test -z "$(git diff --name-only --diff-filter=U)"' in justfile
    assert 'test -z "$$(git diff --name-only --diff-filter=U)"' not in justfile


def test_installed_artifact_smoke_is_in_root_and_nested_gates() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    assert "package-smoke:" in justfile
    assert "preprocessing-package-smoke:" in justfile
    assert "just package-smoke" in justfile
    assert "just preprocessing-package-smoke" in justfile
    assert "scripts/quality/package_smoke.py" in justfile


def test_package_smoke_helpers_are_quality_gated() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    assert "scripts/quality/package_smoke.py" in mutation["source_paths"]
    assert "tests/quality/test_package_smoke.py" in mutation["pytest_add_cli_args_test_selection"]
