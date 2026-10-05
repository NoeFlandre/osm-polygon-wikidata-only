"""Characterisation tests for the shared path-safe stem helper."""

from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.domain.stems import is_safe_stem, require_safe_stem
from osm_polygon_wikidata_only.grid5000 import sentence_controller_policy, sentence_protocol

CASES = [("", False), (".", False), ("..", False), ("a/b", False), ("a\\b", False), ("ok", True)]


@pytest.mark.parametrize(("stem", "expected"), CASES)
def test_is_safe_stem(stem: str, expected: bool) -> None:
    assert is_safe_stem(stem) is expected


def test_is_safe_stem_rejects_non_strings() -> None:
    assert is_safe_stem(None) is False
    assert is_safe_stem(3) is False


@pytest.mark.parametrize(("stem", "expected"), CASES)
def test_require_safe_stem_uses_label_and_error(stem: str, expected: bool) -> None:
    if expected:
        assert require_safe_stem(stem, label="x") == stem
        return
    with pytest.raises(KeyError, match="Invalid thing"):
        require_safe_stem(stem, label="thing", error=KeyError)
    with pytest.raises(ValueError, match="Invalid thing"):
        require_safe_stem(stem, label="thing")


def test_run_id_helper_is_shared() -> None:
    assert sentence_controller_policy.is_safe_run_id is sentence_protocol.is_safe_run_id
    assert sentence_protocol.is_safe_run_id("run-1_a")
    assert not sentence_protocol.is_safe_run_id("Run/1")
