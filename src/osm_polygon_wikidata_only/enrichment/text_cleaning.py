"""Deterministic text cleaning for Wikipedia article content.

All functions are pure and operate on strings. The output is stable
for the same input across runs and platforms.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import ClassVar, Literal

_WHITESPACE_RE = re.compile(r"\s+")
_SENTINEL_RE = re.compile(r"\{\{[^}]*\}\}")  # simple {{...}} markers


class _RenderedTextParser(HTMLParser):
    """Collect visible text from MediaWiki parser HTML."""

    _IGNORED: ClassVar[frozenset[str]] = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORED:
            self._ignored_depth += 1
        elif tag in {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in {"div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "tr"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces, strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_template_markers(text: str) -> str:
    """Remove simple ``{{...}}`` template markers."""
    return _SENTINEL_RE.sub("", text)


def normalize_unicode(text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC") -> str:
    """Apply Unicode normalization ``form`` (default NFC)."""
    result: str = unicodedata.normalize(form, text)
    return result


def clean_article_text(text: str) -> str:
    """Apply the full cleaning pipeline to Wikipedia text.

    The order is fixed: normalize unicode, strip simple templates,
    collapse whitespace, strip.
    """
    out = normalize_unicode(text)
    out = strip_template_markers(out)
    out = normalize_whitespace(out)
    return out


def html_to_plain_text(html: str) -> str:
    """Convert rendered MediaWiki HTML to deterministic visible plain text."""
    parser = _RenderedTextParser()
    parser.feed(html)
    parser.close()
    return clean_article_text("".join(parser.parts))


def count_words(text: str) -> int:
    """Approximate whitespace-token word count."""
    if not text:
        return 0
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: characters / 4.

    This is a deliberate dependency-free approximation. For most
    English Wikipedia text the rule of thumb ``chars / 4`` is within
    30% of the true BPE token count, which is enough for an
    upper-bound estimate used for budgeting.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


__all__ = [
    "clean_article_text",
    "count_words",
    "estimate_tokens",
    "html_to_plain_text",
    "normalize_unicode",
    "normalize_whitespace",
    "strip_template_markers",
]
