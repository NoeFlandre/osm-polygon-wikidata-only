"""Pure validation and parsing helpers for Wikidata responses."""

from __future__ import annotations

import re
from typing import Any

from .models import Sitelinks, WikidataEntity

_QID_PATTERN = re.compile(r"^Q[1-9]\d*$")
# Wikimedia project prefixes that are NOT language Wikipedias. These
# sitelinks must be dropped early because their hosts do not live under
# ``<lang>.wikipedia.org`` (e.g. Wikifunctions lives at
# ``wikifunctions.org``). Without this filter the pipeline tries to
# fetch them, gets a DNS failure, and aborts the whole PBF.
_NON_LANGUAGE_PROJECTS = frozenset(
    {
        "commons",
        "foundation",
        "incubator",
        "mediawiki",
        "meta",
        "outreach",
        "sources",
        "species",
        "strategy",
        "test",
        "test2",
        "wikidata",
        "wikifunctions",
    }
)


def is_valid_qid(qid: str) -> bool:
    """Return whether *qid* is a positive Wikidata entity identifier."""
    return bool(_QID_PATTERN.fullmatch(qid))


def qids_from_osm_tag(value: str) -> tuple[str, ...]:
    """Return the distinct QIDs in one OSM ``wikidata=*`` value.

    OSM uses semicolons for the uncommon case where a tag contains
    multiple values. Preserve their source order while trimming the
    whitespace permitted around separators. An empty or malformed
    component makes the complete value invalid rather than silently
    dropping source data.
    """
    components = tuple(component.strip() for component in value.split(";"))
    if not components or any(not is_valid_qid(component) for component in components):
        return ()
    return tuple(dict.fromkeys(components))


def parse_wikidata_entity(qid: str, data: dict[str, Any]) -> WikidataEntity | None:
    """Parse one entity from a ``wbgetentities`` response."""
    raw = _entity_payload(qid, data)
    if raw is None:
        return None
    return WikidataEntity(
        qid,
        _language_sitelinks(raw),
        _localized_values(raw, "labels"),
        _localized_values(raw, "descriptions"),
        _localized_aliases(raw),
    )


def _entity_payload(qid: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Return a non-missing entity payload, if present."""
    entities = data.get("entities") or {}
    if qid not in entities:
        return None
    raw = entities[qid]
    return None if raw.get("missing") is not None else raw


def _language_sitelinks(raw: dict[str, Any]) -> Sitelinks:
    """Keep titled sitelinks that point to language Wikipedias."""
    sitelinks: Sitelinks = {}
    for site, info in (raw.get("sitelinks") or {}).items():
        title = _language_sitelink_title(site, info)
        if title:
            sitelinks[site] = title
    return sitelinks


def _language_sitelink_title(site: str, info: dict[str, Any]) -> str | None:
    """Return a valid language sitelink title, if present."""
    if not _is_language_wiki(site):
        return None
    title = info.get("title")
    return title if title else None


def _localized_values(raw: dict[str, Any], field: str) -> dict[str, str]:
    """Extract localized scalar values from one entity field."""
    return {key: value.get("value", "") for key, value in (raw.get(field) or {}).items()}


def _localized_aliases(raw: dict[str, Any]) -> dict[str, list[str]]:
    """Extract non-empty localized aliases from one entity."""
    return {
        key: [value["value"] for value in values if value.get("value")]
        for key, values in (raw.get("aliases") or {}).items()
    }


def language_from_site(site: str) -> str:
    """Convert a Wikidata Wikipedia site key to its language code."""
    if not site.endswith("wiki"):
        return site
    language = site[: -len("wiki")]
    return {"be_x_old": "be-tarask"}.get(language, language.replace("_", "-"))


def _is_language_wiki(site: str) -> bool:
    """Return whether *site* identifies a language Wikipedia."""
    if not site.endswith("wiki") or len(site) <= len("wiki"):
        return False
    language = site[: -len("wiki")]
    return _is_language_code(language)


def _is_language_code(language: str) -> bool:
    """Return whether a site prefix is a lowercase language code."""
    if language in _NON_LANGUAGE_PROJECTS:
        return False
    if language != language.lower():
        return False
    return all(_is_language_character(character) for character in language)


def _is_language_character(character: str) -> bool:
    """Return whether one site-key character is valid in a language code."""
    return character.isalnum() or character in ("_", "-")


__all__ = ["is_valid_qid", "language_from_site", "parse_wikidata_entity", "qids_from_osm_tag"]
