"""Managed language section of the Hub dataset card.

The language publication owns one Markdown section and one marked block of
YAML ``configs`` entries in the dataset card. These helpers replace exactly
those regions and leave every other byte of the card untouched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from osm_polygon_wikidata_only.hf._publication.language_errors import LanguagePublicationError
from osm_polygon_wikidata_only.hf.language_split_release import LanguageSplitVersion

__all__ = ["LANGUAGE_CARD_HEADING", "language_sort_key", "merge_language_card"]

LANGUAGE_CARD_HEADING = "## Language partitions"
_LANGUAGE_CONFIG_BEGIN = "  # BEGIN LANGUAGE SPLIT CONFIGS"
_LANGUAGE_CONFIG_END = "  # END LANGUAGE SPLIT CONFIGS"


def language_sort_key(language: str) -> tuple[bool, str]:
    """Sort languages alphabetically with ``unknown`` last."""
    return language == "unknown", language


def merge_language_card(
    existing: str,
    *,
    version: LanguageSplitVersion,
    configurations: Sequence[str],
    languages: Sequence[str],
    configuration_languages: Sequence[tuple[str, Sequence[str]]] = (),
) -> str:
    """Replace only the managed language section and preserve other card text."""
    if configuration_languages:
        existing = _merge_language_front_matter(existing, version, configuration_languages)
    section = _render_language_card_section(version, configurations, languages)
    pattern = re.compile(
        rf"^{re.escape(LANGUAGE_CARD_HEADING)}\n.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(existing)
    if match:
        matched = match.group(0)
        trailing_newlines = len(matched) - len(matched.rstrip("\n"))
        separator = "\n" * trailing_newlines
        replacement = section.rstrip("\n") + separator
        return existing[: match.start()] + replacement + existing[match.end() :]
    marker = re.search(r"^## Data sources & licenses\n", existing, re.MULTILINE)
    if marker:
        return existing[: marker.start()] + section + "\n" + existing[marker.start() :]
    return existing.rstrip() + "\n\n" + section


def _merge_language_front_matter(
    existing: str,
    version: LanguageSplitVersion,
    configuration_languages: Sequence[tuple[str, Sequence[str]]],
) -> str:
    """Replace the managed language config block without reserializing YAML."""
    if not existing.startswith("---\n"):
        raise LanguagePublicationError(
            "dataset card has no YAML front matter; refusing to add Viewer language configs"
        )
    closing = existing.find("\n---", 4)
    if closing < 0:
        raise LanguagePublicationError(
            "dataset card YAML front matter is unterminated; refusing to add Viewer language configs"
        )
    front_matter = existing[4:closing]
    block = _render_language_front_matter_block(version, configuration_languages)
    marker_pattern = re.compile(
        rf"^{re.escape(_LANGUAGE_CONFIG_BEGIN)}\n.*?^{re.escape(_LANGUAGE_CONFIG_END)}\n?",
        re.MULTILINE | re.DOTALL,
    )
    marked = marker_pattern.search(front_matter)
    if marked:
        updated_front_matter = front_matter[: marked.start()] + block + front_matter[marked.end() :]
    else:
        configs = re.search(r"^configs:\s*$", front_matter, re.MULTILINE)
        if configs is None:
            raise LanguagePublicationError(
                "dataset card YAML front matter has no configs field; "
                "refusing to add Viewer language configs"
            )
        insertion = configs.end()
        updated_front_matter = (
            front_matter[:insertion] + "\n" + block.rstrip("\n") + front_matter[insertion:]
        )
    return "---\n" + updated_front_matter + existing[closing:]


def _render_language_front_matter_block(
    version: LanguageSplitVersion,
    configuration_languages: Sequence[tuple[str, Sequence[str]]],
) -> str:
    lines = [_LANGUAGE_CONFIG_BEGIN]
    for configuration, languages in sorted(configuration_languages, key=lambda item: item[0]):
        sorted_languages = sorted(set(languages), key=language_sort_key)
        if version is LanguageSplitVersion.V1:
            lines.append(f"  - config_name: {configuration}")
            lines.append("    data_files:")
            for language in sorted_languages:
                storage_split = f"lang-{language}"
                path = f"data/{configuration}/{storage_split}-00000-of-00001.parquet"
                lines.append(f"      - split: {storage_split}")
                lines.append(f"        path: {path}")
            continue
        for language in sorted_languages:
            storage_split = f"lang-{language}"
            lines.append(f"  - config_name: {_v2_language_config_name(configuration, language)}")
            lines.append("    data_files:")
            path = f"language_splits/{configuration}/{storage_split}/part-*.parquet"
            lines.append("      - split: train")
            lines.append(f"        path: {path}")
    lines.append(_LANGUAGE_CONFIG_END)
    return "\n".join(lines) + "\n"


def _v2_language_config_name(configuration: str, language: str) -> str:
    """Return a Viewer subset name for one V2 table/language pair."""
    return f"{configuration}__lang_{language.replace('-', '_')}"


def _render_language_card_section(
    version: LanguageSplitVersion,
    configurations: Sequence[str],
    languages: Sequence[str],
) -> str:
    contract = "V1" if version is LanguageSplitVersion.V1 else "V2"
    lines = [
        LANGUAGE_CARD_HEADING,
        "",
        f"The {contract} language release is an additive, row-level partition of the published text tables.",
        "Each source row is routed by its normalized `language` value; multilingual rows are not collapsed to a polygon-level preferred language.",
        "Missing, blank, malformed, and legacy-unusable values are preserved in the explicit `lang-unknown` partition.",
        "",
        f"Validated languages: **{len(languages)}** (including `unknown`).",
        "",
        "| Configuration | Split names | Remote path |",
        "| --- | --- | --- |",
    ]
    for configuration in sorted(configurations):
        if version is LanguageSplitVersion.V1:
            split_label = "`lang-<language>`"
            unknown_label = "`lang-unknown`"
            path = f"data/{configuration}/lang-<language>-00000-of-00001.parquet"
        else:
            split_label = f"`{configuration}__lang_<language>` (split `train`)"
            unknown_label = f"`{configuration}__lang_unknown` (split `train`)"
            path = f"language_splits/{configuration}/lang-<language>/part-*.parquet"
        lines.append(f"| `{configuration}` | {split_label} and {unknown_label} | `{path}` |")
    lines.extend(
        [
            "",
            "The release manifest records the source fingerprint, schema, row counts, and SHA-256 hash for every generated file.",
            "",
        ]
    )
    return "\n".join(lines)
