"""Completeness failures for fail-closed PBF publication."""


class IncompleteEnrichmentError(RuntimeError):
    """Raised when expected Wikimedia work did not finish successfully."""


__all__ = ["IncompleteEnrichmentError"]
