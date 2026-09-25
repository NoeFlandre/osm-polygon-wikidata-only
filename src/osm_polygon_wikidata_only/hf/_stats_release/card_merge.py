"""Merging a regenerated dataset card with the remote card."""

from __future__ import annotations

import re
from collections.abc import Sequence

from osm_polygon_wikidata_only.hf._publication.language_card import LANGUAGE_CARD_HEADING

# Sections preserved verbatim from the remote card: author-owned prose, plus
# sections owned by another publication path. "## Language partitions" is
# written by the language-split release and describes artifacts this release
# knows nothing about, so regenerating the card must not drop it. Every other
# section is data-derived and is always regenerated, so a released card can
# never carry a stale statistic next to a fresh one.
_PRESERVED_SECTION_HEADINGS = frozenset(
    {
        "## Citation",
        LANGUAGE_CARD_HEADING,
        "## License",
        "## Licensing",
        "## Reproducibility",
        "## Data sources & licenses",
        "## How to load",
    }
)


def _split_h2_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(re.finditer(r"(?m)^## [^\n]*\n?", markdown))
    if not matches:
        return markdown, []
    prefix = markdown[: matches[0].start()]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        heading = match.group(0).rstrip("\r\n")
        sections.append((heading, markdown[match.start() : end]))
    return prefix, sections


def merge_release_card(existing: str, generated: str) -> str:
    """Return the released card: regenerated data, preserved prose and header.

    The generated body is authoritative. Every section it renders is
    data-derived and replaces whatever the remote card held, and any
    section the generated card no longer renders is dropped rather than
    carried forward -- that carry-forward is what previously let a stale
    statistic survive beside a freshly computed one.

    Two things are preserved from the remote card. Author-owned prose
    sections listed in :data:`_PRESERVED_SECTION_HEADINGS` are appended
    when the generated card omits them. The YAML front matter is kept
    verbatim, because the Dataset Viewer ``configs:`` block there is
    owned by the publication and language-split paths -- a statistics
    release must never drop the published language partitions from the
    Viewer.
    """
    if not existing:
        return generated
    existing_front_matter, existing_body = _split_front_matter(existing)
    generated_front_matter, generated_body = _split_front_matter(generated)
    prefix, generated_sections = _split_h2_sections(generated_body)
    if not generated_sections:
        return generated
    sections = _released_sections(generated_sections, existing_body)
    front_matter = existing_front_matter or generated_front_matter
    return front_matter + _join_card_sections(prefix, sections)


def _released_sections(
    generated_sections: Sequence[tuple[str, str]],
    existing_body: str,
) -> list[str]:
    """Return the generated sections followed by any preserved prose."""
    rendered = {heading for heading, _section in generated_sections}
    generated = [section for _heading, section in generated_sections]
    return generated + _preserved_sections(existing_body, rendered)


def _preserved_sections(existing_body: str, rendered: set[str]) -> list[str]:
    """Return remote sections this release must not drop or regenerate."""
    _prefix, existing_sections = _split_h2_sections(existing_body)
    return [
        section
        for heading, section in existing_sections
        if heading in _PRESERVED_SECTION_HEADINGS and heading not in rendered
    ]


def _split_front_matter(card: str) -> tuple[str, str]:
    """Split a card into its YAML front matter and the markdown body."""
    if not card.startswith("---\n"):
        return "", card
    end = card.find("\n---\n", 4)
    if end < 0:
        return "", card
    boundary = end + len("\n---\n")
    return card[:boundary], card[boundary:]


def _join_card_sections(prefix: str, sections: Sequence[str]) -> str:
    parts = [prefix.strip("\n")] if prefix.strip() else []
    parts.extend(section.rstrip("\n") for section in sections)
    return "\n\n".join(parts) + "\n"
