"""The ``cli`` package exposes the legacy command API lazily."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only import cli
from osm_polygon_wikidata_only.cli import commands


@pytest.mark.parametrize("name", ["build_parser", "main"])
def test_legacy_command_api_resolves_to_the_commands_module(name: str) -> None:
    assert getattr(cli, name) is getattr(commands, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        getattr(cli, "missing")


def test_dir_lists_the_lazy_names() -> None:
    assert {"build_parser", "main"} <= set(dir(cli))
