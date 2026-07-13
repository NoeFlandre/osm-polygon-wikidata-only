"""Per-file upload to a Hugging Face Hub repository.

Uploads are deliberately *per file* (not a single
``Dataset.push_to_hub`` blob), so users can update/replace just one
PBF's parquet without re-uploading the whole dataset. Public callers
go through :func:`upload_parquet`, :func:`upload_manifest`, and
:func:`upload_card`.

This module is a thin compatibility facade. The implementation lives
in :mod:`osm_polygon_wikidata_only.hf._uploader` and every public
name below is re-exported unchanged.
"""

from __future__ import annotations

from ._uploader.authorization import verify_repo_authorization
from ._uploader.errors import UploadError
from ._uploader.operations import (
    default_commit_message,
    upload_card,
    upload_files,
    upload_manifest,
    upload_parquet,
)
from ._uploader.protocol import HfHub
from ._uploader.stub import StubHfHub
from ._uploader.token import resolve_hf_token, verify_hf_token

__all__ = [
    "HfHub",
    "StubHfHub",
    "UploadError",
    "default_commit_message",
    "resolve_hf_token",
    "upload_card",
    "upload_files",
    "upload_manifest",
    "upload_parquet",
    "verify_hf_token",
    "verify_repo_authorization",
]
