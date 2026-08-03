"""Pure normalization of multilingual OSM Wikipedia tags."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$", re.IGNORECASE)
_WIKIPEDIA_KEY = "wikipedia"


@dataclass(frozen=True, slots=True)
class WikipediaTagRef:
    """One normalized direct Wikipedia page reference from an OSM tag."""

    language: str
    title: str
    raw_key: str
    raw_value: str


@dataclass(frozen=True, slots=True)
class WikipediaTagRejection:
    """One malformed or unusable OSM Wikipedia value."""

    raw_key: str
    raw_value: str
    reason: str


def _language(value: str) -> str | None:
    normalized = value.strip().replace("_", "-").lower()
    return normalized if _LANGUAGE_RE.fullmatch(normalized) else None


def _from_url(value: str) -> tuple[str, str] | None:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    if not parsed.scheme or not host.endswith(".wikipedia.org"):
        return None
    language = _language(host.removesuffix(".wikipedia.org").removesuffix(".m"))
    if language is None or not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ").strip()
    return language, title


def _normalize_value(
    key_language: str | None,
    value: str,
) -> tuple[tuple[str, str] | None, str]:
    value = value.strip()
    if not value:
        return None, "empty language or title"

    url_ref = _from_url(value)
    if url_ref is not None:
        if key_language is not None and key_language != url_ref[0]:
            return None, "language disagrees with URL"
        return url_ref, ""

    if key_language is None:
        language_text, separator, title = value.partition(":")
        if not separator:
            return None, "missing language prefix"
        language = _language(language_text)
        if language is None:
            return None, "invalid language code"
    else:
        language = key_language
        title = value

    title = unquote(title).replace("_", " ").strip()
    if not title or title.startswith(":"):
        return None, "empty language or title"
    return (language, title), ""


def parse_wikipedia_tags(
    tags: dict[str, str],
) -> tuple[tuple[WikipediaTagRef, ...], tuple[WikipediaTagRejection, ...]]:
    """Return normalized direct references and non-fatal rejection records.

    Both the conventional ``wikipedia=lang:Title`` form and dynamic
    ``wikipedia:<language>=Title`` keys are accepted. Language codes are
    validated structurally, not against a hardcoded list, so new and
    regional Wikimedia projects remain discoverable.
    """

    refs: list[WikipediaTagRef] = []
    rejected: list[WikipediaTagRejection] = []
    seen: set[tuple[str, str]] = set()
    for key, raw in tags.items():
        if key != _WIKIPEDIA_KEY and not key.startswith(f"{_WIKIPEDIA_KEY}:"):
            continue
        key_language: str | None = None
        if key != _WIKIPEDIA_KEY:
            key_language = _language(key.removeprefix(f"{_WIKIPEDIA_KEY}:"))
            if key_language is None:
                rejected.append(WikipediaTagRejection(key, raw, "invalid language code"))
                continue

        values = [part.strip() for part in raw.split(";")]
        for value in values:
            if not value:
                rejected.append(WikipediaTagRejection(key, raw, "empty value"))
                continue
            normalized, reason = _normalize_value(key_language, value)
            if normalized is None:
                rejected.append(WikipediaTagRejection(key, value, reason))
                continue
            language, title = normalized
            identity = (language, title)
            if identity in seen:
                continue
            seen.add(identity)
            refs.append(
                WikipediaTagRef(
                    language=language,
                    title=title,
                    raw_key=key,
                    raw_value=value,
                )
            )

    refs.sort(key=lambda ref: (ref.language, ref.title, ref.raw_key, ref.raw_value))
    rejected.sort(key=lambda item: (item.raw_key, item.raw_value, item.reason))
    return tuple(refs), tuple(rejected)


__all__ = ["WikipediaTagRef", "WikipediaTagRejection", "parse_wikipedia_tags"]
