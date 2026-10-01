"""Shared synthetic rows for augmentation contract tests."""

from __future__ import annotations


def article_row() -> dict[str, object]:
    """Return the canonical article row used by augmentation tests."""
    return {
        "article_id": "Q1:en:10:20",
        "wikidata": "Q1",
        "language": "en",
        "site": "enwiki",
        "title": "Andorra",
        "url": "https://en.wikipedia.org/wiki/Andorra",
        "page_id": 10,
        "revision_id": 20,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-02T00:00:00Z",
        "full_text": "Lead. History text.",
        "full_text_format": "plain_text",
        "article_length_chars": 19,
        "article_length_words": 3,
        "article_length_tokens_estimate": 4,
        "license": "CC BY-SA 4.0",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "abc",
    }
