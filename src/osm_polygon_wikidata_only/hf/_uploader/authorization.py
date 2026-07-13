"""Repository-namespace authorization.

Verifies that the token's verified owner matches the namespace in
``repo_id``. Hugging Face rejects ``create_repo``/``create_commit``
for repos in a namespace the token does not own with a misleading
401 ``Invalid username or password`` body. Catching that mismatch
here turns a 25-minute run that dies at first upload into an
immediate, actionable error at process start.
"""

from __future__ import annotations

from typing import Any

from .errors import UploadError
from .token import verify_hf_token

__all__ = ["verify_repo_authorization"]


def verify_repo_authorization(
    explicit: str | None,
    repo_id: str,
    *,
    _verify: Any = None,
) -> str:
    """Verify the token's owner matches ``repo_id``'s namespace."""
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
