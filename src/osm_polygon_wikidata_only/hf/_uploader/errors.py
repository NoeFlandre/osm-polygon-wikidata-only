"""UploadError exception.

Re-exported by the :mod:`osm_polygon_wikidata_only.hf.uploader`
facade. The error identity is part of the documented contract; do not
rename.
"""

from __future__ import annotations


class UploadError(RuntimeError):
    """Raised when an upload request fails.

    ``transient`` marks failures worth retrying (rate limits, server errors,
    network outages). Every other failure is permanent and must not be retried.
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


__all__ = ["UploadError"]
