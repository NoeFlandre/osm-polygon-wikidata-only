"""Contracts that keep the Python toolchain pin in a single file."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _pinned_python() -> str:
    return (ROOT / ".python-version").read_text(encoding="utf-8").strip()


def test_workflows_install_the_python_named_by_the_version_file() -> None:
    install_lines: list[tuple[str, str]] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        assert "python-version" not in text, f"{workflow.name} hard-codes a Python version"
        install_lines.extend(
            (workflow.name, line.strip())
            for line in text.splitlines()
            if "uv python install" in line
        )

    assert install_lines, "no workflow installs Python"
    for name, line in install_lines:
        assert line == "run: uv python install", f"{name} passes an explicit version: {line}"


def test_runtime_image_python_tag_matches_the_version_file() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"ARG UV_IMAGE=\S*-python(\d+\.\d+)-", dockerfile)

    assert match is not None, "Dockerfile UV_IMAGE does not name a Python version"
    assert match.group(1) == _pinned_python()
