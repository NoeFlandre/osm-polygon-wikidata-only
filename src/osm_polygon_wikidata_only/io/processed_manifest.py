"""Read the region mapping of a processed-PBF manifest.

The manifest is a JSON object from source key to region entry. The V2 layout
nests that mapping under ``regions``; the flat layout is the mapping itself.
Callers pass the errors to raise, so each package keeps its own error type
and message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast


def processed_manifest_entries(
    raw: object,
    *,
    on_not_object: Callable[[], Exception],
    on_regions_invalid: Callable[[], Exception],
    regions_required: bool = False,
    allow_empty: bool = True,
) -> dict[object, object]:
    """Return the region mapping of a parsed processed manifest.

    Without ``regions_required`` the ``regions`` key is optional: when it is
    absent the document is the mapping, and when it is present its value must
    be an object. With ``regions_required`` the mapping must sit under
    ``regions``. With ``allow_empty`` false an empty mapping is refused too.
    """
    if not isinstance(raw, dict):
        raise on_not_object()
    entries = _regions_of(cast(dict[object, object], raw), regions_required)
    if _regions_refused(entries, allow_empty):
        raise on_regions_invalid()
    return cast(dict[object, object], entries)


def _regions_of(document: dict[object, object], regions_required: bool) -> object:
    if regions_required:
        return document.get("regions")
    return document.get("regions", document)


def _regions_refused(entries: object, allow_empty: bool) -> bool:
    return not isinstance(entries, dict) or (not allow_empty and not entries)


__all__ = ["processed_manifest_entries"]
