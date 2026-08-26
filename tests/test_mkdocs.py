"""Contract tests for the public MkDocs site and Pages workflow."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_mkdocs_configuration_defines_public_navigation_and_exclusions() -> None:
    config = (REPOSITORY / "mkdocs.yml").read_text(encoding="utf-8")

    assert "site_name: OSM Polygon Wikidata Only" in config
    assert "site_url: https://noeflandre.github.io/osm-polygon-wikidata-only/" in config
    assert "name: material" in config
    for page in (
        "index.md",
        "api.md",
        "architecture.md",
        "development.md",
        "sentence-splitting.md",
    ):
        assert page in config
    assert "superpowers/**" in config
    assert ".DS_Store" in config


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
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "path: site" in workflow


def test_ci_runs_the_canonical_full_quality_gate() -> None:
    """CI must execute the same complete gate documented for contributors."""
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "run: just quality-gauntlet" in workflow
    assert "run: just check" not in workflow
    assert "run: just coverage" not in workflow


def test_docs_landing_page_is_public_and_points_to_project_sources() -> None:
    landing = (REPOSITORY / "docs/index.md").read_text(encoding="utf-8")

    assert "Hugging Face" in landing
    assert "sync-dir" in landing
    assert "docs/api.md" not in landing
    assert "/Volumes/" not in landing
    assert "external drive" not in landing.lower()
    assert "[API reference](api.md)" in landing
    assert "[Architecture](architecture.md)" in landing
