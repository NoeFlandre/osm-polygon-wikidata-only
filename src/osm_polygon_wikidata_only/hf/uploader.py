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

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        exist_ok: bool,
    ) -> Any: ...


class StubHfHub:
    """In-memory HF Hub used by tests.

    Records every uploaded file in ``uploads``. Never touches the
    network.
    """

    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []
        self.created_repos: list[dict[str, Any]] = []

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

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        exist_ok: bool,
    ) -> str:
        self.created_repos.append(
            {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": exist_ok}
        )
        return repo_id


def resolve_hf_token(explicit: str | None) -> str | None:
    """Return the effective HF token, honouring ``HF_TOKEN`` env and saved logins.

    ``HfApi(token=explicit).token`` only stores the explicit value and
    never reads the environment, so naively probing ``HfApi().token``
    is not enough. We delegate to ``huggingface_hub.get_token`` when
    no explicit value is supplied, which honours ``HF_TOKEN``,
    ``HUGGING_FACE_HUB_TOKEN`` and the saved login cache.
    """
    if explicit:
        return explicit
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover
        return None
    try:
        token = get_token()
    except Exception:  # pragma: no cover - get_token can raise if backend misbehaves
        return None
    if isinstance(token, str) and token:
        return token
    return None


def verify_hf_token(explicit: str | None, *, _whoami: Any = None) -> str | None:
    """Verify the effective HF token by calling ``whoami``.

    Raises :class:`UploadError` with the upstream message when the
    token is rejected (expired, revoked, or wrong account). Returns
    the verified username on success. Returns ``None`` if no token is
    configured (the caller decides whether that is fatal).
    """
    token = resolve_hf_token(explicit)
    if not token:
        return None
    if _whoami is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:  # pragma: no cover
            raise UploadError(
                "huggingface_hub is required to verify a token. "
                "Install with `uv add huggingface_hub`."
            ) from e

        def _whoami(tok: str) -> dict[str, Any]:
            return HfApi(token=tok).whoami()

    try:
        info = _whoami(token)
    except Exception as error:
        raise UploadError(
            f"Hugging Face rejected HF_TOKEN: {error}. "
            "Generate a fresh write token at https://huggingface.co/settings/tokens."
        ) from error
    name = info.get("name") if isinstance(info, dict) else None
    return str(name) if name else "unknown"


def verify_repo_authorization(
    explicit: str | None,
    repo_id: str,
    *,
    _verify: Any = None,
) -> str:
    """Verify the token's owner matches ``repo_id``'s namespace.

    Hugging Face rejects ``create_repo``/``create_commit`` for repos
    in a namespace the token does not own with a misleading 401
    ``Invalid username or password`` body. Catching that mismatch
    here turns a 25-minute run that dies at first upload into an
    immediate, actionable error at process start.
    """
    if "/" not in repo_id:
        raise UploadError(f"repo_id must be of the form 'namespace/name', got {repo_id!r}")
    namespace = repo_id.split("/", 1)[0]
    if _verify is None:
        _verify = verify_hf_token
    try:
        username = _verify(explicit)
    except UploadError:
        raise
    if username is None:
        raise UploadError(
            "No Hugging Face token available. Set HF_TOKEN, run `huggingface-cli login`, "
            "or pass --hf-token."
        )
    if username != namespace:
        raise UploadError(
            f"HF_TOKEN authenticates as '{username}', but --repo-id '{repo_id}' lives in the "
            f"'{namespace}' namespace. Either use a write token issued by '{namespace}' "
            f"or pass --repo-id '{username}/osm-polygon-wikidata-only' to push under your own namespace."
        )
    return str(username)


# Internal alias kept so the test-only ``_resolve_token`` keyword in
# :func:`upload_files`, :func:`upload_parquet`, :func:`upload_card` and the
# preflight helper can refer to it without importing the public name into
# every signature.
_resolve_hf_token = resolve_hf_token


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
    import io

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
