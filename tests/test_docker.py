"""Public Docker reproducibility contracts.

These tests intentionally inspect the checked-in container contract instead of
starting a real data run.  The optional smoke test is opt-in because a Docker
daemon is not available in every contributor environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (REPOSITORY / name).read_text(encoding="utf-8")


def test_dockerfile_uses_locked_uv_install_and_safe_runtime() -> None:
    dockerfile = _read("Dockerfile")

    assert "ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.16-python3.12-bookworm-slim" in dockerfile
    assert "FROM ${UV_IMAGE} AS build" in dockerfile
    assert "FROM ${UV_IMAGE} AS runtime" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md LICENSE ./" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER app" in dockerfile
    assert 'ENTRYPOINT ["osm-polygon-wikidata-only"]' in dockerfile
    assert 'CMD ["--help"]' in dockerfile
    assert 'VOLUME ["/data"]' in dockerfile
    assert "OSM_POLYGON_DATA_ROOT=/data" in dockerfile
    assert "PYTHONHASHSEED=0" in dockerfile
    assert "LC_ALL=C.UTF-8" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "HF_TOKEN" not in dockerfile


def test_dockerignore_excludes_local_data_secrets_and_presentations() -> None:
    dockerignore = _read(".dockerignore")

    for pattern in (
        ".git/",
        ".venv/",
        "presentations/",
        "raw/",
        "processed/",
        "processed_v2/",
        "cache/",
        "logs/",
        "*.osm.pbf",
        "*.parquet",
        ".env",
        ".huggingface/",
        ".DS_Store",
    ):
        assert pattern in dockerignore


def test_justfile_documents_safe_docker_targets_and_opt_in_pipeline() -> None:
    justfile = _read("Justfile")

    for recipe in ("docker-build:", "docker-help:", "docker-test:", "docker-check:"):
        assert recipe in justfile
    assert "docker-run data_root: docker-build" in justfile
    assert '--user "$(id -u):$(id -g)"' in justfile
    assert "--mount" in justfile
    assert "dst=/data/raw,readonly" in justfile
    assert "--data-root /data" in justfile
    assert "sync-dir /data/raw" in justfile
    assert "--push" in justfile


def test_development_docs_explain_reproducible_docker_mounts_and_safety() -> None:
    development = _read("docs/development.md")

    for text in (
        "Docker reproducibility",
        "docker build",
        "uv.lock",
        "read-only",
        "--mount",
        "/data/raw",
        "Ctrl-C",
        "HF_TOKEN",
    ):
        assert text in development
    assert "never" in development.lower()


def test_architecture_docs_describe_the_container_data_boundary() -> None:
    architecture = _read("docs/architecture.md")

    for text in ("build", "development", "runtime", "/data", "non-root", "sync-dir /data/raw"):
        assert text in architecture


def test_ci_builds_and_smoke_tests_the_runtime_image() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "docker build" in workflow
    assert "--target runtime" in workflow
    assert "docker run --rm" in workflow
    assert "--help" in workflow


@pytest.mark.integration
def test_opt_in_docker_smoke() -> None:
    """Build and run the image only when explicitly requested.

    The repository's normal test suite remains hermetic.  CI exercises the
    same build and help smoke command in its dedicated container job.
    """

    if os.environ.get("RUN_DOCKER_SMOKE") != "1":
        pytest.skip("set RUN_DOCKER_SMOKE=1 to run the Docker smoke test")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is not installed")
    daemon = subprocess.run([docker, "info"], capture_output=True, text=True, check=False)
    if daemon.returncode != 0:
        pytest.skip(f"Docker daemon is unavailable: {daemon.stderr.strip()}")

    image = "osm-polygon-wikidata-only:test"
    build = subprocess.run(
        [docker, "build", "--target", "runtime", "--tag", image, str(REPOSITORY)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    help_run = subprocess.run(
        [docker, "run", "--rm", image, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_run.returncode == 0, help_run.stdout + help_run.stderr
    assert "Build a Hugging Face dataset" in help_run.stdout
