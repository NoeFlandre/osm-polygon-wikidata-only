"""Tests for osm_polygon_wikidata_only.domain.ids."""

from __future__ import annotations

from osm_polygon_wikidata_only.domain.ids import article_id, content_hash, polygon_id


def test_polygon_id_is_stable() -> None:
    assert polygon_id("monaco-latest", "way", 123) == "monaco-latest:way:123"


def test_polygon_id_distinguishes_type() -> None:
    assert polygon_id("monaco-latest", "way", 123) != polygon_id("monaco-latest", "relation", 123)


def test_article_id_includes_all_components() -> None:
    assert article_id("Q42", "en", 100, 555) == "Q42:en:100:555"


def test_article_id_distinguishes_revision() -> None:
    assert article_id("Q42", "en", 100, 555) != article_id("Q42", "en", 100, 556)


def test_content_hash_is_deterministic_hex() -> None:
    h1 = content_hash("hello world")
    h2 = content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex
    int(h1, 16)  # parses as hex


def test_content_hash_differs_for_different_text() -> None:
    assert content_hash("a") != content_hash("b")


def test_content_hash_preserves_unicode() -> None:
    h = content_hash("héllo wörld")
    # Different from ASCII-only
    assert h != content_hash("hello world")
