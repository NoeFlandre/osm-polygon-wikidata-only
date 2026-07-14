"""HfHub Protocol: the minimal contract a real (or stub) HF client must satisfy.

Only methods actually used by the uploader are listed. Real
Hugging Face clients (``huggingface_hub.HfApi``) implement all of
these. Tests substitute a stub that records calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

__all__ = ["HfHub"]


class HfHub(Protocol):
    """Minimal contract for an HF Hub client."""

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

    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
    ) -> bool: ...
