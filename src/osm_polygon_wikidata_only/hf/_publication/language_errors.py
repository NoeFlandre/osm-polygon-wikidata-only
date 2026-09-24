"""Error type shared by the language-partition publication units."""

from __future__ import annotations

__all__ = ["LanguagePublicationError"]


class LanguagePublicationError(RuntimeError):
    """Raised when a language publication cannot be completed safely."""
