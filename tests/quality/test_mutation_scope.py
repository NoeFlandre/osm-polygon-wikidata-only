"""Contracts for the source and test scope of the mutation gate."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def test_language_split_release_facade_is_in_mutation_source_and_test_scopes() -> None:
    config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    assert {
        "src/osm_polygon_wikidata_only/hf/language_split_release.py",
        "src/osm_polygon_wikidata_only/v2/language_splits.py",
    } <= set(mutation["source_paths"])
    assert {
        "tests/hf/test_language_split_release.py",
        "tests/v2/test_language_splits.py",
    } <= set(mutation["pytest_add_cli_args_test_selection"])


def test_language_split_mutation_tests_are_collection_safe() -> None:
    for relative_path in (
        "tests/hf/test_language_split_release.py",
        "tests/v2/test_language_splits.py",
    ):
        module = ast.parse((REPOSITORY / relative_path).read_text(encoding="utf-8"))

        top_level_modules = {
            node.module.split(".", maxsplit=1)[0]
            for node in module.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        top_level_modules.update(
            alias.name.split(".", maxsplit=1)[0]
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        assert "datasets" not in top_level_modules, relative_path
        assert "numpy" not in top_level_modules, relative_path
