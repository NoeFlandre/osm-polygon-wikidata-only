"""UploadError exception.

Re-exported by the :mod:`osm_polygon_wikidata_only.hf.uploader`
facade. The error identity is part of the documented contract; do not
rename.
"""

from __future__ import annotations


class UploadError(RuntimeError):
    """Raised when an upload request fails."""


__all__ = ["UploadError"]
