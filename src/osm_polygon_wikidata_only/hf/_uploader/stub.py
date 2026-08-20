"""StubHfHub: in-memory HF Hub used by tests.

Records every uploaded file in ``uploads``, every commit in
``commits``, and every ``create_repo`` call in ``created_repos``.
Never touches the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from .plan import PublicationOp

__all__ = ["StubHfHub"]


class StubHfHub:
    """In-memory HF Hub used by tests.

    Records every uploaded file in ``uploads``. Never touches the
    network.
    """

    def __init__(self, *, remote_files: set[str] | None = None) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []
        self.created_repos: list[dict[str, Any]] = []
        # ``None`` preserves the historical permissive stub behavior.
        # Supplying a set enables explicit remote-state simulation.
        self.remote_files = remote_files

    def file_exists(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
    ) -> bool:
        del repo_id, repo_type
        return self.remote_files is None or filename in self.remote_files

    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: str,
    ) -> list[str]:
        del repo_id, repo_type
        if self.remote_files is not None:
            return sorted(list(self.remote_files))
        return []

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
        ops = [_serialize_operation(operation) for operation in operations]
        self.commits.append(
            {
                "repo_id": repo_id,
                "paths": [op["path_in_repo"] for op in ops],
                "operations": ops,
                "commit_message": commit_message,
                "repo_type": repo_type,
                "num_threads": num_threads,
            }
        )
        _apply_remote_operations(self.remote_files, ops)
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


def _serialize_operation(operation: Any) -> dict[str, Any]:
    if isinstance(operation, PublicationOp):
        return {"action": operation.action, "path_in_repo": operation.path_in_repo}
    cls_name = type(operation).__name__
    action = "delete" if "Delete" in cls_name else "add"
    return {"action": action, "path_in_repo": getattr(operation, "path_in_repo", None)}


def _apply_remote_operations(
    remote_files: set[str] | None,
    operations: list[dict[str, Any]],
) -> None:
    if remote_files is None:
        return
    for operation in operations:
        if operation["action"] == "add":
            remote_files.add(operation["path_in_repo"])
        else:
            remote_files.discard(operation["path_in_repo"])
