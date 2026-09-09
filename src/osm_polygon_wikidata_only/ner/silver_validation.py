"""Pure, model-independent quality signals for geographic NER pilots."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_QID = re.compile(r"q\d+\Z")
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_NUMBER = re.compile(r"[\d\W_]+\Z", re.UNICODE)


def normalize_name(value: object) -> str:
    """Normalize a name conservatively for case and whitespace-independent matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def is_artifact(value: object) -> bool:
    """Return whether a model surface is an obvious non-name artifact."""
    normalized = normalize_name(value)
    return bool(normalized and (_QID.fullmatch(normalized) or _URL.search(normalized))) or bool(
        normalized and _NUMBER.fullmatch(normalized)
    )


def audit_entities(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    known_names: Sequence[str],
) -> dict[str, int]:
    """Count reference matches, artifacts, unmatched spans, and model agreement."""
    reference = _normalized_names(known_names)
    primary_surfaces = _normalized_surfaces(primary)
    usable = _usable_surfaces(primary_surfaces)
    secondary_surfaces = set(_normalized_surfaces(secondary))
    return {
        "primary_entities": len(primary),
        "primary_artifacts": len(primary_surfaces) - len(usable),
        "primary_known_name_matches": _matches(usable, reference),
        "primary_unmatched": _unmatched(usable, reference),
        "secondary_entities": len(secondary),
        "model_agreements": _matches(usable, secondary_surfaces),
    }


def _normalized_names(values: Sequence[str]) -> set[str]:
    return {normalized for value in values if (normalized := normalize_name(value))}


def _matches(values: Sequence[str], reference: set[str]) -> int:
    return sum(value in reference for value in values)


def _unmatched(values: Sequence[str], reference: set[str]) -> int:
    return sum(value not in reference for value in values)


def _normalized_surfaces(entities: Sequence[Mapping[str, Any]]) -> list[str]:
    return [normalize_name(_surface(entity)) for entity in entities]


def _usable_surfaces(surfaces: Sequence[str]) -> list[str]:
    return [surface for surface in surfaces if surface and not is_artifact(surface)]


def _surface(entity: Mapping[str, Any]) -> object:
    return entity.get("text", "")
