"""Convert exact-revision MediaWiki HTML into ordered plain-text sections."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser

from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
)
from osm_polygon_wikidata_only.utils.json import dumps

from .models import Document, Section, stable_id

_EXCLUDED = frozenset({"references", "external links", "bibliography", "notes", "further reading"})


class _SectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[str, str, int, str]] = [("", "", 0, "")]
        self._heading_level = 0
        self._heading_parts: list[str] = []
        self._text_parts: list[str] = []
        self._ignored = 0

    def _flush(self) -> None:
        heading, anchor, level, _ = self.sections[-1]
        self.sections[-1] = (heading, anchor, level, clean_article_text(" ".join(self._text_parts)))
        self._text_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "table", "sup"}:
            self._ignored += 1
        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._heading_level = int(tag[1])
            self._heading_parts = []
        elif not self._ignored and tag in {"p", "li", "br", "div"}:
            self._text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._close_ignored_tag(tag):
            return
        if self._close_heading(tag):
            return
        if not self._ignored and tag in {"p", "li", "div"}:
            self._text_parts.append(" ")

    def _close_ignored_tag(self, tag: str) -> bool:
        if tag not in {"script", "style", "table", "sup"}:
            return False
        if self._ignored:
            self._ignored -= 1
        return True

    def _close_heading(self, tag: str) -> bool:
        if not self._heading_level or tag != f"h{self._heading_level}":
            return False
        heading = clean_article_text(" ".join(self._heading_parts))
        self.sections.append((heading, heading.replace(" ", "_"), self._heading_level, ""))
        self._heading_level = 0
        return True

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        (self._heading_parts if self._heading_level else self._text_parts).append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def parse_sections(document: Document, html: str) -> list[Section]:
    parser = _SectionParser()
    parser.feed(html)
    parser.close()
    return _build_sections(document, parser.sections)


def _build_sections(
    document: Document,
    parsed_sections: list[tuple[str, str, int, str]],
) -> list[Section]:
    rows: list[Section] = []
    stack: list[Section] = []
    for index, (heading, anchor, level, text) in enumerate(parsed_sections):
        if heading.casefold() in _EXCLUDED or not text:
            continue
        _trim_section_stack(stack, level)
        row = _make_section(document, index, heading, anchor, level, text, stack)
        rows.append(row)
        stack.append(row)
    return rows


def _trim_section_stack(stack: list[Section], level: int) -> None:
    while stack and stack[-1].level >= level:
        stack.pop()


def _make_section(
    document: Document,
    index: int,
    heading: str,
    anchor: str,
    level: int,
    text: str,
    stack: list[Section],
) -> Section:
    path = [item.heading for item in stack if item.heading] + ([heading] if heading else [])
    return Section(
        stable_id(document.document_id, index, heading),
        document.document_id,
        document.article_id,
        document.wikidata,
        document.project,
        document.language,
        document.site,
        document.page_id,
        document.revision_id,
        index,
        heading,
        anchor,
        level,
        stack[-1].section_id if stack else "",
        dumps(path),
        text,
        len(text),
        count_words(text),
        estimate_tokens(text),
        hashlib.sha256(text.encode()).hexdigest(),
        document.license,
        document.attribution,
    )


__all__ = ["parse_sections"]
