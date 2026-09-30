"""Shared fixtures for pipeline reconciliation test modules."""

from __future__ import annotations

from typing import Any

import pytest

from osm_polygon_wikidata_only.cli import commands, run_sync
from osm_polygon_wikidata_only.config.settings import Settings


class _NoopScheduler:
    def __init__(self) -> None:
        self.snapshot: dict[str, object] = {}


class _NoopSession:
    def __init__(self) -> None:
        self.auth_snapshot: dict[str, object] = {}


class _NoopWikimediaRuntime:
    def __init__(self) -> None:
        self.settings: Settings = Settings(repo_id="test", user_agent="test")
        self.scheduler = _NoopScheduler()
        self.session = _NoopSession()
        self.wikidata: None = None
        self.wikipedia: None = None
        self.cache: None = None


def _build_noop_wikimedia_runtime(*_args: Any, **_kwargs: Any) -> _NoopWikimediaRuntime:
    return _NoopWikimediaRuntime()


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


@pytest.fixture
def no_op_wikimedia_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep sync tests offline and fail if a no-op runtime client is invoked."""
    monkeypatch.setattr(run_sync, "build_wikimedia_runtime", _build_noop_wikimedia_runtime)
