"""Contracts for the source and test scope of the mutation gate."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def test_language_split_release_facade_is_in_mutation_source_and_test_scopes() -> None:
    config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    assert "src/osm_polygon_wikidata_only/hf/language_split_release.py" in mutation["source_paths"]
    assert (
        "tests/hf/test_language_split_release.py" in mutation["pytest_add_cli_args_test_selection"]
    )
