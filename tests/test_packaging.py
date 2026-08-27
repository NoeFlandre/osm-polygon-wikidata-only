"""Public distribution metadata and typing marker tests."""

from __future__ import annotations

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
        "crap4py",
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
        "wtpsplit[onnx-gpu]==2.2.1"
    ]


def test_justfile_is_the_uv_managed_quality_command_catalog() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    for recipe in (
        "sync:",
        "test:",
        "coverage:",
        "baseline:",
        "ruff:",
        "lint:",
        "tests:",
        "acceptance-tests:",
        "architecture-checks:",
        "format:",
        "format-check:",
        "typecheck:",
        "ty:",
        "build:",
        "docs:",
        "trackio:",
        "mutation:",
        "crap:",
        "crap-all:",
        "crap-upload:",
        "quality-strength:",
        "smoke-test:",
        "diff-review:",
        "qa-gauntlet:",
        "quality-gauntlet:",
        "check:",
    ):
        assert recipe in justfile
    for command in (
        "uv sync --frozen",
        "uv run pytest",
        "uv run ruff check src tests scripts",
        "uv run ruff format --check src tests scripts",
        "uv run ty check src scripts",
        "uv build",
        "uv run mkdocs build --strict",
        "git diff --check",
    ):
        assert command in justfile
    assert "mypy" not in justfile
    assert "scripts/quality/qa_gauntlet.py" in justfile


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


def test_coverage_recipes_use_isolated_temporary_databases() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    assert "COVERAGE_FILE=/tmp/osm-polygon-wikidata-only-coverage-$$" in justfile


def test_crap_gate_keeps_the_existing_v2_fingerprint_scope() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    assert "src/osm_polygon_wikidata_only/v2/fingerprints.py" in justfile


def test_grid5000_protocol_is_in_the_pure_quality_scopes() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    protocol_source = "src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py"
    protocol_tests = "tests/grid5000/test_sentence_protocol.py"
    assert protocol_source in justfile
    assert "--cov=osm_polygon_wikidata_only.grid5000.sentence_protocol" in justfile
    assert protocol_source in mutation["source_paths"]
    assert protocol_tests in mutation["pytest_add_cli_args_test_selection"]


def test_diff_review_executes_unmerged_path_check() -> None:
    """The diff-review recipe must evaluate, not quote, its command substitution."""

    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    assert 'test -z "$(git diff --name-only --diff-filter=U)"' in justfile
    assert 'test -z "$$(git diff --name-only --diff-filter=U)"' not in justfile
