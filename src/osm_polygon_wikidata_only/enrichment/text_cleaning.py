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


class BlockTextParser(HTMLParser):
    """Collect text from HTML while skipping ignored elements.

    Character data inside an element named in ``IGNORED_TAGS`` is dropped, and
    ignored elements may nest. Outside them, a start tag in ``BLOCK_START_TAGS``
    or an end tag in ``BLOCK_END_TAGS`` reports a block boundary through
    :meth:`on_block_boundary`, and other character data goes to :meth:`on_text`.
    Subclasses set the three tag sets and decide where each callback writes.
    """

    IGNORED_TAGS: ClassVar[frozenset[str]] = frozenset()
    BLOCK_START_TAGS: ClassVar[frozenset[str]] = frozenset()
    BLOCK_END_TAGS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0

    @property
    def ignoring(self) -> bool:
        """Whether the parser is inside an element named in ``IGNORED_TAGS``."""
        return self._ignored_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002 -- HTMLParser override signature
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag in self.BLOCK_START_TAGS and not self.ignoring:
            self.on_block_boundary()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
        elif tag in self.BLOCK_END_TAGS and not self.ignoring:
            self.on_block_boundary()

    def handle_data(self, data: str) -> None:
        if not self.ignoring:
            self.on_text(data)

    def on_block_boundary(self) -> None:
        """Record a separator where a block element starts or ends."""
        raise NotImplementedError

    def on_text(self, data: str) -> None:
        """Record visible character data."""
        raise NotImplementedError


# Elements that start a new visual unit. Each one gets a separator on both its
# start and end tag so that neighbouring cells, terms and blocks never fuse into
# a single token. Inline tags are left out on purpose: ``foo<b>bar</b>`` must
# stay ``foobar``.
_RENDERED_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "article",
        "blockquote",
        "br",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


class _RenderedTextParser(BlockTextParser):
    """Collect visible text from MediaWiki parser HTML."""

    IGNORED_TAGS = frozenset({"script", "style"})
    BLOCK_START_TAGS = _RENDERED_BLOCK_TAGS
    BLOCK_END_TAGS = _RENDERED_BLOCK_TAGS

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def on_block_boundary(self) -> None:
        self.parts.append(" ")

    def on_text(self, data: str) -> None:
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
    return normalize_whitespace(out)


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
