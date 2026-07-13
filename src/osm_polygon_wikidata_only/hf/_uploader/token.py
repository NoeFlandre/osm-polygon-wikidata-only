"""HF token resolution and verification.

Honors explicit token > ``HF_TOKEN`` env / saved login, and verifies
the resolved token by calling ``whoami`` on the live HF API. All
Hugging Face API calls are lazy and guarded so the module is
importable without ``huggingface_hub`` installed.
"""

from __future__ import annotations

from typing import Any

from .errors import UploadError

__all__ = ["_resolve_hf_token", "resolve_hf_token", "verify_hf_token"]


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


# Internal alias kept so the test-only ``_resolve_token`` keyword in
# :func:`upload_files`, :func:`upload_parquet`, :func:`upload_card` and the
# preflight helper can refer to it without importing the public name into
# every signature.
_resolve_hf_token = resolve_hf_token


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
