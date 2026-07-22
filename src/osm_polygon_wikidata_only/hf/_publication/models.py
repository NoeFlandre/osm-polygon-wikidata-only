"""Immutable models and validation errors used by publication assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PublicationValidationError(ValueError):
    """Raised when publication artifacts violate their local contract."""


@dataclass(frozen=True, slots=True)
class CorePublicationArtifacts:
    """Validated local artifacts for publishing an existing core region."""

    polygons_path: Path
    polygon_articles_path: Path
    wikipedia_documents_path: Path | None
    manifest_path: Path
    stem: str
    manifest_entry: dict[str, Any]


__all__ = ["CorePublicationArtifacts", "PublicationValidationError"]
