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


def test_package_declares_inline_typing_support() -> None:
    marker = Path(__file__).parents[1] / "src/osm_polygon_wikidata_only/py.typed"
    assert marker.is_file()
