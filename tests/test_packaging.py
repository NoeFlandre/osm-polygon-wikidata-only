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
    assert "uv run ty check src scripts" in workflow
    assert "uv run mypy" not in workflow
