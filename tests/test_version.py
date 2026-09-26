"""The package version is single-sourced from ``pyproject.toml`` (issue #109)."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import osm_polygon_wikidata_only
from osm_polygon_wikidata_only.config.settings import DEFAULT_USER_AGENT

ROOT = Path(__file__).resolve().parents[1]


def test_dunder_version_matches_installed_metadata() -> None:
    assert osm_polygon_wikidata_only.__version__ == version("osm-polygon-wikidata-only")


def test_installed_metadata_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert osm_polygon_wikidata_only.__version__ == project["version"]


def test_user_agent_carries_the_real_version() -> None:
    assert f"osm-polygon-wikidata-only/{osm_polygon_wikidata_only.__version__} " in DEFAULT_USER_AGENT


def test_version_literal_lives_only_in_pyproject() -> None:
    literal = osm_polygon_wikidata_only.__version__
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if f'"{literal}"' in path.read_text(encoding="utf-8")
        or f"/{literal} " in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_changelog_documents_the_current_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Keep a Changelog" in changelog
    assert f"## [{osm_polygon_wikidata_only.__version__}]" in changelog


def test_release_workflow_pins_actions_by_sha() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "tags:" in workflow
    for line in workflow.splitlines():
        if "uses:" in line:
            ref = line.split("@", 1)[1].split()[0]
            assert len(ref) == 40, line
