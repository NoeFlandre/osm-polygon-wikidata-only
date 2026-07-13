"""Upload operations and shared helpers.

This module owns:

* :func:`upload_parquet`: per-file upload.
* :func:`upload_files`: atomic, concurrent commit of multiple files.
* :func:`upload_manifest`: thin wrapper over :func:`upload_parquet`
  with a manifest-default commit message.
* :func:`upload_card`: in-memory README upload via a buffered
  ``BytesIO``.
* :func:`default_commit_message`: PBF stem commit message.

It also owns the shared helpers :func:`_build_hf_api`,
:func:`_ensure_repo_exists`, :func:`_translate_hf_error`,
:func:`_commit_message` and the private ``_resolve_hf_token`` alias
(re-exported from :mod:`.token`).
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import UploadError
from .protocol import HfHub
from .token import _resolve_hf_token

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.uploader")

__all__ = [
    "default_commit_message",
    "upload_card",
    "upload_files",
    "upload_manifest",
    "upload_parquet",
]


def _build_hf_api(
    token: str | None,
    *,
    api_factory: Any = None,
) -> Any:
    """Build a real HF Hub client lazily.

    Imported only when actually needed so the module is importable
    in environments without ``huggingface_hub``. The resolved token
    must be non-empty; otherwise we raise :class:`UploadError` with a
    actionable hint instead of letting the request fail later with a
    confusing ``401 Unauthorized``.
    """
    factory = api_factory
    if factory is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:  # pragma: no cover
            raise UploadError(
                "huggingface_hub is required for real uploads. Install with `uv add huggingface_hub`."
            ) from e
        factory = HfApi
    api = factory(token=token)
    if not getattr(api, "token", None):
        raise UploadError(
            "No Hugging Face token available. Set the HF_TOKEN environment variable, "
            "run `huggingface-cli login`, or pass --hf-token explicitly."
        )
    return api


def _ensure_repo_exists(hub: HfHub, repo_id: str, *, repo_type: str = "dataset") -> None:
    """Create the target repo on the Hub if it does not exist yet.

    ``create_commit`` assumes the destination repository already
    exists; without this call the first upload to a brand new repo
    fails with ``RepositoryNotFoundError``/401 Unauthorized. Passing
    ``exist_ok=True`` makes the call a no-op when the repo is
    already present, so it is safe to invoke before every upload.
    """
    LOGGER.info("Ensuring %s repo exists: %s", repo_type, repo_id)
    try:
        hub.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    except Exception as error:
        raise _translate_hf_error(error, repo_id=repo_id) from error


def _translate_hf_error(error: Exception, *, repo_id: str) -> UploadError:
    """Translate Hugging Face HTTP errors into actionable :class:`UploadError`.

    ``huggingface_hub`` returns ``RepositoryNotFoundError`` for both
    404 (genuine "repo does not exist") and 401 ("Invalid username or
    password"). The 401 case is almost always an auth issue, not a
    missing repo, but the wrapper's default message hides that. We
    inspect the response body to surface the real cause and tell the
    user what to do.
    """
    status_code: int | None = None
    server_message: str | None = None
    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        try:
            server_message = response.text
        except Exception:  # pragma: no cover - defensive
            server_message = None
    if server_message is None:
        server_message = getattr(error, "server_message", None) or str(error)
    server_message = (server_message or "").strip()
    lowered = server_message.lower()
    auth_markers = (
        "invalid username",
        "invalid user token",
        "invalid token",
        "bad credentials",
        "401 unauthorized",
        "token is required",
    )
    is_auth = status_code in (401, 403) or any(marker in lowered for marker in auth_markers)
    if is_auth:
        return UploadError(
            f"Hugging Face rejected the upload to {repo_id}: {server_message}. "
            "Verify your HF_TOKEN is a write token from the account that owns "
            f"{repo_id}, or pass --hf-token (or --repo-id) explicitly."
        )
    return UploadError(f"Hugging Face upload to {repo_id} failed: {server_message}")


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
    _resolve_token: Any = _resolve_hf_token,
    _api_factory: Any = None,
) -> str:
    """Upload one parquet file. Returns the resolved ``path_in_repo``."""
    if not local_path.exists():
        raise UploadError(f"Local file does not exist: {local_path}")
    client = hub or _build_hf_api(_resolve_token(token), api_factory=_api_factory)
    _ensure_repo_exists(client, repo_id)
    msg = _commit_message(commit_message, default_prefix=f"parquet {Path(path_in_repo).name}")
    LOGGER.info("Uploading %s -> %s@%s", local_path, repo_id, path_in_repo)
    try:
        return client.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=msg,
        )
    except Exception as error:
        raise _translate_hf_error(error, repo_id=repo_id) from error


def upload_files(
    repo_id: str,
    files: Iterable[tuple[Path, str]],
    *,
    hub: HfHub | None = None,
    token: str | None = None,
    commit_message: str,
    num_threads: int = 5,
    _resolve_token: Any = _resolve_hf_token,
    _api_factory: Any = None,
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
    client = hub or _build_hf_api(_resolve_token(token), api_factory=_api_factory)
    _ensure_repo_exists(client, repo_id)
    LOGGER.info("Uploading %d files atomically to %s", len(operations), repo_id)
    try:
        result = client.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            repo_type="dataset",
            num_threads=num_threads,
        )
    except Exception as error:
        raise _translate_hf_error(error, repo_id=repo_id) from error
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
    _resolve_token: Any = _resolve_hf_token,
    _api_factory: Any = None,
) -> str:
    """Upload the dataset card."""
    if not card_markdown.strip():
        raise UploadError("Cannot upload empty dataset card")
    client = hub or _build_hf_api(_resolve_token(token), api_factory=_api_factory)
    _ensure_repo_exists(client, repo_id)
    # Use the lighter-weight ``upload_file`` with a buffered string.
    buffer = io.BytesIO(card_markdown.encode("utf-8"))
    LOGGER.info("Uploading dataset card (%d chars) -> %s", len(card_markdown), repo_id)
    try:
        return client.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or "Update dataset card",
        )
    except Exception as error:
        raise _translate_hf_error(error, repo_id=repo_id) from error


def default_commit_message(stem: str) -> str:
    return f"Update PBF {stem}"
