"""Completeness failures for fail-closed PBF publication."""

NON_FATAL_FETCH_STATUSES = frozenset({"article_not_found", "empty_text"})


class IncompleteEnrichmentError(RuntimeError):
    """Raised when expected Wikimedia work did not finish successfully."""


__all__ = ["NON_FATAL_FETCH_STATUSES", "IncompleteEnrichmentError"]
