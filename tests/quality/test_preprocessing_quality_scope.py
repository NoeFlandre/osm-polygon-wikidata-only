import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_preprocessing_declares_its_quality_dependencies_and_gate() -> None:
    config = tomllib.loads((ROOT / "preprocessing/pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = set(config["dependency-groups"]["dev"])
    assert "pytest>=9.1.1" in dev_dependencies

    justfile = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "preprocessing-check:" in justfile
    assert "uv sync --frozen --directory preprocessing" in justfile
    assert "preprocessing/src" in justfile
    assert "preprocessing/tests" in justfile
    assert "uv run ruff check" in justfile
    assert (
        "uv run --frozen ty check --project preprocessing --python preprocessing/.venv preprocessing/src"
        in justfile
    )

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert "preprocessing" in workflow["jobs"]
    steps = workflow["jobs"]["preprocessing"]["steps"]
    assert any(step.get("run") == "just preprocessing-check" for step in steps)
