"""Contract tests for the public MkDocs site and Pages workflow."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_pages_workflow_builds_strictly_and_deploys_with_least_privilege() -> None:
    workflow = (REPOSITORY / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "pages: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "uv sync --frozen" in workflow
    assert "uv run mkdocs build --strict" in workflow
    assert "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa # v3" in workflow
    assert "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e # v4" in workflow
    assert "path: site" in workflow


def test_ci_runs_the_canonical_full_quality_gate() -> None:
    """CI must execute the same complete gate documented for contributors."""
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "run: just quality-gauntlet" in workflow
    assert "run: just check" not in workflow
    assert "run: just coverage" not in workflow
