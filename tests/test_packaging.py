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
    assert "run: just check" in workflow
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


def test_justfile_is_the_uv_managed_quality_command_catalog() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "Justfile").read_text(encoding="utf-8")

    for recipe in (
        "sync:",
        "test:",
        "coverage:",
        "lint:",
        "format:",
        "format-check:",
        "typecheck:",
        "build:",
        "docs:",
        "trackio:",
        "mutation:",
        "crap:",
        "crap-upload:",
        "quality-strength:",
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


def test_github_actions_runs_the_test_strength_gate() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "run: just quality-strength" in workflow


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
    assert "run: just check" in workflow
    for recipe in ("coverage", "lint", "format-check", "typecheck", "build"):
        assert f"run: just {recipe}" not in workflow
    assert "uv sync --frozen" not in workflow
