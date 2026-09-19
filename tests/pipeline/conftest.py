"""Shared fixtures for pipeline reconciliation test modules."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.cli import commands


@pytest.fixture
def mock_hf_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic Hugging Face authentication for local tests."""
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "fake-token")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(
        commands,
        "verify_repo_authorization",
        lambda token, repo_id: "noeflandre",
    )
