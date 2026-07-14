"""Human-readable column descriptions for the augmentation sidecars.

The dataset card renderer is the canonical place that lists every
augmentation column; the descriptions live here so the renderer and
the column list are guaranteed to evolve together.

The column SOURCE OF TRUTH is :mod:`osm_polygon_wikidata_only.augmentation.schema`
(DOCUMENT_COLUMNS, SECTION_COLUMNS, FACT_COLUMNS). This module owns
the descriptions only; column ordering, types, and identities stay
with the schema module.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
)

# Per-column descriptions. Keyed by column name. Multi-line
# descriptions are kept short and renderer-friendly.
DOCUMENT_DESCRIPTIONS: dict[str, str] = {
    "document_id": "Deterministic document identifier (`<wikidata>:<project>:<language>:<page_id>:<revision_id>`).",
    "article_id": "Stable article identifier that pairs this document with its `articles/<stem>.parquet` row.",
    "wikidata": "Wikidata QID this document is linked to.",
    "project": "Wiki project name: `wikipedia` or `wikivoyage`.",
    "language": "Wikipedia or Wikivoyage language code (e.g. `en`).",
    "site": "Wikidata sitelink host, e.g. `enwiki` or `enwikivoyage`.",
    "title": "Page title as returned by the MediaWiki API.",
    "url": "Canonical URL of the page.",
    "page_id": "MediaWiki page ID (integer).",
    "revision_id": "MediaWiki revision ID used to fetch the page text (integer).",
    "revision_timestamp": "ISO-8601 timestamp of the revision.",
    "retrieved_at": "ISO-8601 UTC timestamp when the pipeline fetched the page.",
    "full_text": "Cleaned plain-text document body (Wikipedia articles or Wikivoyage pages).",
    "full_text_format": "Encoding of `full_text`; always `plain_text`.",
    "article_length_chars": "Length of `full_text` in characters.",
    "article_length_words": "Approximate whitespace-token count of `full_text`.",
    "article_length_tokens_estimate": "Rough token estimate: `chars / 4`.",
    "license": "License string (`CC BY-SA` for Wikipedia/Wikivoyage text).",
    "attribution": "Attribution string for the page.",
    "source_api": "Which API was queried: `mediawiki_action_api` or `wikivoyage_rest_api`.",
    "fetch_status": "One of: `ok`, `page_not_found`, `http_error`, `rate_limited`, `parse_error`, `empty_text`.",
    "fetch_error": "Short diagnostic on failure, empty string on success.",
    "content_hash": "Stable SHA-256 of `full_text` for change tracking.",
}

SECTION_DESCRIPTIONS: dict[str, str] = {
    "section_id": "Deterministic SHA-256 over `(document_id, section_index, anchor)`.",
    "document_id": "FK back to `documents` (Wikipedia or Wikivoyage).",
    "article_id": "FK back to the corresponding `articles` row.",
    "wikidata": "Wikidata QID (denormalized for fast filtering).",
    "project": "Wiki project name: `wikipedia` or `wikivoyage`.",
    "language": "Wikipedia or Wikivoyage language code.",
    "site": "Wikidata sitelink host.",
    "page_id": "MediaWiki page ID (integer).",
    "revision_id": "MediaWiki revision ID (integer).",
    "section_index": "Sequential position of the section inside the document (integer).",
    "heading": "Section heading, or empty string when the section is the lead.",
    "anchor": "Section anchor after MediaWiki parsing.",
    "level": "Heading level (1..6), or 0 for the lead section (integer).",
    "parent_section_id": "Section ID of the enclosing section, or empty string.",
    "section_path": "JSON array of ancestor section IDs, in order.",
    "text": "Plain-text section body.",
    "text_length_chars": "Length of `text` in characters (integer).",
    "text_length_words": "Approximate whitespace-token count of `text` (integer).",
    "text_length_tokens_estimate": "Rough token estimate: `chars / 4` (integer).",
    "content_hash": "Stable SHA-256 of `section.text`.",
    "license": "License string for this section.",
    "attribution": "Attribution string for this section.",
}

FACT_DESCRIPTIONS: dict[str, str] = {
    "fact_id": "Deterministic SHA-256 over `(subject, property, value, ordinal)`.",
    "wikidata": "Wikidata QID the fact belongs to (the subject).",
    "property_id": "Property P-id (e.g. `P17`).",
    "property_label_en": "English label for the property, when available.",
    "property_labels": "Deterministic JSON object of property labels per language.",
    "value_type": "Wikidata value datatype: `wikibase-entityid`, `string`, `quantity`, `time`, ...",
    "value_entity_id": "Entity-valued object QID, when the value is a Wikidata entity.",
    "value_label_en": "English label for the value entity, when available.",
    "value_labels": "Deterministic JSON object of value labels per language.",
    "value_text": "Rendered text representation of the value.",
    "numeric_value": "Numeric amount for `quantity`-typed values, otherwise null.",
    "unit_entity_id": "Wikidata QID of the unit (e.g. `Q11573` for metre).",
    "rank": "Wikidata rank: `preferred`, `normal`, or `deprecated`.",
    "qualifiers": "Deterministic JSON object of qualifier snaks, or `{}` when absent.",
    "references": "Deterministic JSON array of reference groups, or `[]` when absent.",
    "retrieved_at": "ISO-8601 UTC timestamp when the pipeline fetched the entity.",
    "source_api": "Which API was queried (always `wikidata_action_api`).",
}


def _check_coverage(descriptions: dict[str, str], columns: tuple[str, ...], label: str) -> None:
    missing = [c for c in columns if c not in descriptions]
    extra = sorted(set(descriptions) - set(columns))
    if missing or extra:
        raise AssertionError(
            f"{label}: descriptions must cover exactly {columns}; missing={missing}, extra={extra}"
        )


_check_coverage(DOCUMENT_DESCRIPTIONS, DOCUMENT_COLUMNS, "DOCUMENT_DESCRIPTIONS")
_check_coverage(SECTION_DESCRIPTIONS, SECTION_COLUMNS, "SECTION_DESCRIPTIONS")
_check_coverage(FACT_DESCRIPTIONS, FACT_COLUMNS, "FACT_DESCRIPTIONS")


__all__ = [
    "DOCUMENT_DESCRIPTIONS",
    "FACT_DESCRIPTIONS",
    "SECTION_DESCRIPTIONS",
]
