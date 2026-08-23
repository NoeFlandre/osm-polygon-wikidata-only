"""Deterministic, lossless deduplication for V2 document artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

_Identity = str | tuple[str, str, str]


def _document_identity(raw: dict[str, Any]) -> str:
    identity = str(raw.get("document_id", ""))
    if not identity:
        raise ValueError("Document row is missing document_id")
    return identity


def _link_identity(raw: dict[str, Any]) -> tuple[str, str, str]:
    identity = (
        str(raw.get("polygon_id", "")),
        str(raw.get("project", "")),
        str(raw.get("document_id", "")),
    )
    if not all(identity):
        raise ValueError(f"Link row is missing identity fields: {identity!r}")
    return identity


def _deduplicate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    identity_of: Callable[[dict[str, Any]], _Identity],
    kind: str,
) -> list[dict[str, Any]]:
    by_identity: dict[_Identity, dict[str, Any]] = {}
    for raw in rows:
        identity = identity_of(raw)
        row = dict(raw)
        previous = by_identity.get(identity)
        if previous is not None and previous != row:
            raise ValueError(f"Conflicting duplicate {kind} identity {identity!r}")
        by_identity[identity] = row
    return [by_identity[identity] for identity in sorted(by_identity)]


def deduplicate_documents(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse identical document rows and reject conflicting identities.

    A document identity is its ``document_id``.  Identical duplicates are
    harmless repeated observations and collapse to one row.  Different rows
    with the same identity are data conflicts and fail closed.
    """
    return _deduplicate_rows(rows, identity_of=_document_identity, kind="document")


def deduplicate_links(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse identical links and reject conflicting link identities.

    A link identity is ``(polygon_id, project, document_id)``.  Source
    provenance remains part of the row, so conflicting provenance is treated
    as a conflict rather than silently discarded.
    """
    return _deduplicate_rows(rows, identity_of=_link_identity, kind="link")


__all__ = ["deduplicate_documents", "deduplicate_links"]
