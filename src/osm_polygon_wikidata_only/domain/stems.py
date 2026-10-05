"""Shared path-safety predicate for region stems and similar path components."""

from __future__ import annotations


def is_safe_stem(stem: object) -> bool:
    """Return whether *stem* is a non-empty string usable as one path component.

    A safe stem is not ``.`` or ``..`` and contains no ``/`` or ``\\``.
    """
    if not isinstance(stem, str):
        return False
    return bool(stem) and stem not in {".", ".."} and "/" not in stem and "\\" not in stem


def require_safe_stem(
    stem: str,
    *,
    label: str,
    error: type[Exception] = ValueError,
) -> str:
    """Return *stem* when safe; otherwise raise ``error("Invalid <label>: <stem!r>")``."""
    if not is_safe_stem(stem):
        raise error(f"Invalid {label}: {stem!r}")
    return stem


__all__ = ["is_safe_stem", "require_safe_stem"]
