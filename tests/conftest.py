"""Shared pytest configuration for the root test suite."""

from __future__ import annotations

import shutil

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``requires_just`` when the ``just`` runner is absent."""

    if shutil.which("just") is not None:
        return
    skip_just = pytest.mark.skip(reason="just executable is not installed")
    for item in items:
        if item.get_closest_marker("requires_just") is not None:
            item.add_marker(skip_just)
