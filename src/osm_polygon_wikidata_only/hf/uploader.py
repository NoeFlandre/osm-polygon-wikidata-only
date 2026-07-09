"""Per-file upload to a Hugging Face Hub repository.

Uploads are deliberately *per file* (not a single
``Dataset.push_to_hub`` blob), so users can update/replace just one
PBF's parquet without re-uploading the whole dataset. Public callers
go through :func:`upload_parquet`, :func:`upload_manifest`, and
:func:`upload_card`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


class UploadError(RuntimeError):
    """Raised when an upload request fails."""


class HfHub(Protocol):
    """Minimal contract for an HF Hub client.

    Only methods actually used by this module are listed. Real
    Hugging Face clients (``huggingface_hub.HfApi``) implement all
    of these. Tests can substitute a stub that records calls.
    """

    def upload_file(
        self,
        *,
        path_or_fileobj: str | bytes | Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> str: ...

    def create_commit(
        self,
        *,
        repo_id: str,
        operations: Iterable[Any],
        commit_message: str,
        repo_type: str,
        num_threads: int,
    ) -> Any: ...


class StubHfHub:
    """In-memory HF Hub used by tests.

    Records every uploaded file in ``uploads``. Never touches the
    network.
    """

    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []

    def upload_file(
        self,
        *,
        path_or_fileobj: str | os.PathLike[str] | bytes | Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        data: bytes
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            data = bytes(path_or_fileobj)
        elif isinstance(path_or_fileobj, (str, os.PathLike)):
            with open(path_or_fileobj, "rb") as f:
                data = f.read()
        elif hasattr(path_or_fileobj, "read"):
            raw = path_or_fileobj.read()
            data = raw if isinstance(raw, bytes) else bytes(raw)
        else:
            data = bytes(path_or_fileobj)
        self.uploads.append(
            {
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "commit_message": commit_message,
                "size_bytes": len(data),
                "commit_id": str(uuid4()),
            }
        )
        return path_in_repo

    def create_commit(
        self,
        *,
        repo_id: str,
        operations: Iterable[Any],
        commit_message: str,
        repo_type: str,
        num_threads: int,
    ) -> str:
        paths = [str(operation.path_in_repo) for operation in operations]
        self.commits.append(
            {
                "repo_id": repo_id,
                "paths": paths,
                "commit_message": commit_message,
                "repo_type": repo_type,
                "num_threads": num_threads,
            }
        )
        return str(uuid4())


def _build_hf_api(token: str | None) -> Any:
    """Build a real HF Hub client lazily.

    Imported only when actually needed so the module is importable
    in environments without ``huggingface_hub``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise UploadError(
            "huggingface_hub is required for real uploads. Install with `uv add huggingface_hub`."
        ) from e
    return HfApi(token=token)


def _commit_message(commit_message: str | None, *, default_prefix: str) -> str:
    return commit_message or f"Upload {default_prefix}"


def upload_parquet(
    repo_id: str,
    local_path: Path,
    *,
    path_in_repo: str,
    hub: HfHub | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload one parquet file. Returns the resolved ``path_in_repo``."""
    if not local_path.exists():
        raise UploadError(f"Local file does not exist: {local_path}")
    client = hub or _build_hf_api(token=token)
    msg = _commit_message(commit_message, default_prefix=f"parquet {Path(path_in_repo).name}")
    LOGGER.info("Uploading %s -> %s@%s", local_path, repo_id, path_in_repo)
    return client.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )


def upload_files(
    repo_id: str,
    files: Iterable[tuple[Path, str]],
    *,
    hub: HfHub | None = None,
    token: str | None = None,
    commit_message: str,
    num_threads: int = 5,
) -> str:
    """Publish multiple files in one atomic, concurrent Hub commit."""
    entries = list(files)
    if not entries:
        raise UploadError("Cannot create an empty upload commit")
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    remote_paths = [remote for _, remote in entries]
    if len(remote_paths) != len(set(remote_paths)):
        raise UploadError("Upload commit contains duplicate remote paths")
    for local_path, _ in entries:
        if not local_path.exists():
            raise UploadError(f"Local file does not exist: {local_path}")
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as e:  # pragma: no cover
        raise UploadError("huggingface_hub is required for batch uploads.") from e
    operations = [
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path))
        for local_path, remote_path in entries
    ]
    client = hub or _build_hf_api(token=token)
    LOGGER.info("Uploading %d files atomically to %s", len(operations), repo_id)
    result = client.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=commit_message,
        repo_type="dataset",
        num_threads=num_threads,
    )
    return str(result)


def upload_manifest(
    repo_id: str,
    local_manifest_path: Path,
    *,
    path_in_repo: str,
    hub: HfHub | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload the manifest JSON."""
    return upload_parquet(
        repo_id,
        local_manifest_path,
        path_in_repo=path_in_repo,
        hub=hub,
        token=token,
        commit_message=commit_message or "Update manifest",
    )


def upload_card(
    repo_id: str,
    card_markdown: str,
    *,
    path_in_repo: str = "README.md",
    hub: HfHub | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload the dataset card."""
    if not card_markdown.strip():
        raise UploadError("Cannot upload empty dataset card")
    client = hub or _build_hf_api(token=token)
    # Use the lighter-weight ``upload_file`` with a buffered string.
    import io

    buffer = io.BytesIO(card_markdown.encode("utf-8"))
    LOGGER.info("Uploading dataset card (%d chars) -> %s", len(card_markdown), repo_id)
    return client.upload_file(
        path_or_fileobj=buffer,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Update dataset card",
    )


def default_commit_message(stem: str) -> str:
    return f"Update PBF {stem}"


__all__ = [
    "HfHub",
    "StubHfHub",
    "UploadError",
    "default_commit_message",
    "upload_card",
    "upload_files",
    "upload_manifest",
    "upload_parquet",
]
