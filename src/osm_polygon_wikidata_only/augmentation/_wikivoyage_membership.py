"""Shared Wikivoyage document membership policy for polygon QIDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osm_polygon_wikidata_only.augmentation.schema import DOCUMENT_COLUMNS


@dataclass(frozen=True, slots=True)
class RejectedWikivoyageDocument:
    """Identity data needed by callers to build their own rejection records."""

    document_id: str
    wikidata: str


@dataclass(frozen=True, slots=True)
class WikivoyageDocumentPartition:
    """Membership result without imposing a caller's apply or audit contract."""

    retained_rows: list[dict[str, Any]]
    rejected_documents: list[RejectedWikivoyageDocument]


def partition_wikivoyage_documents(
    rows: list[dict[str, Any]],
    valid_qids: set[str],
) -> WikivoyageDocumentPartition:
    """Project valid documents and collect identities absent from polygon QIDs.

    Both integrity application and read-only normalization planning use this
    policy. Each caller remains responsible for its rejection record type,
    reason, cascade counts, and transaction behavior.
    """
    retained_rows: list[dict[str, Any]] = []
    rejected_documents: list[RejectedWikivoyageDocument] = []
    for row in rows:
        document_id = str(row.get("document_id", ""))
        wikidata = str(row.get("wikidata", ""))
        if wikidata in valid_qids:
            retained_rows.append({column: row.get(column) for column in DOCUMENT_COLUMNS})
        else:
            rejected_documents.append(
                RejectedWikivoyageDocument(document_id=document_id, wikidata=wikidata)
            )
    return WikivoyageDocumentPartition(retained_rows, rejected_documents)


__all__ = [
    "RejectedWikivoyageDocument",
    "WikivoyageDocumentPartition",
    "partition_wikivoyage_documents",
]
