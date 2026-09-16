"""StubHfHub: in-memory HF Hub used by tests.

Records every uploaded file in ``uploads``, every commit in
``commits``, and every ``create_repo`` call in ``created_repos``.
Never touches the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from .plan import PublicationOp

__all__ = ["StubHfHub"]


class StubHfHub:
    """In-memory HF Hub used by tests.

    Records every uploaded file in ``uploads``. Never touches the
    network.
    """

    def __init__(
        self,
        *,
        remote_files: set[str] | None = None,
        remote_content: Mapping[str, bytes] | None = None,
    ) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []
        self.created_repos: list[dict[str, Any]] = []
        # ``None`` preserves the historical permissive stub behavior.
        # Supplying a set enables explicit remote-state simulation.
        self.remote_files = remote_files
        self.remote_content: dict[str, bytes] = dict(remote_content or {})
        self._revision: str | None = str(uuid4()) if self.remote_content or remote_files else None

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

    def get_paths_info(
        self,
        repo_id: str,
        paths: list[str],
        *,
        revision: str | None = None,
        repo_type: str,
    ) -> list[Any]:
        del repo_id, revision, repo_type
        return [
            _remote_path_info(path, self.remote_content)
            for path in paths
            if _remote_path_exists(path, self.remote_files, self.remote_content)
        ]

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any:
        del repo_id, repo_type
        return SimpleNamespace(sha=self._revision or "")

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str,
        repo_type: str,
        cache_dir: str | None = None,
    ) -> str:
        del repo_id, revision, repo_type
        if filename not in self.remote_content:
            raise FileNotFoundError(filename)
        root = Path(cache_dir) if cache_dir is not None else Path.cwd() / ".stub-hf-cache"
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.remote_content[filename])
        return str(path)

    def upload_file(
        self,
        *,
        path_or_fileobj: str | os.PathLike[str] | bytes | Any,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        data = _upload_bytes(path_or_fileobj)
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
        self.remote_content[path_in_repo] = data
        self._revision = self.uploads[-1]["commit_id"]
        if self.remote_files is not None:
            self.remote_files.add(path_in_repo)
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
        original_operations = list(operations)
        ops = [_serialize_operation(operation) for operation in original_operations]
        commit_id = str(uuid4())
        self.commits.append(
            {
                "commit_id": commit_id,
                "repo_id": repo_id,
                "paths": [op["path_in_repo"] for op in ops],
                "operations": ops,
                "commit_message": commit_message,
                "repo_type": repo_type,
                "num_threads": num_threads,
            }
        )
        _apply_remote_operations(
            self.remote_files,
            self.remote_content,
            original_operations,
            ops,
        )
        self._revision = commit_id
        return commit_id

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


def _remote_path_exists(
    path: str,
    remote_files: set[str] | None,
    remote_content: Mapping[str, bytes],
) -> bool:
    return path in remote_content or (remote_files is not None and path in remote_files)


def _remote_path_info(path: str, remote_content: Mapping[str, bytes]) -> Any:
    return SimpleNamespace(
        path=path,
        size=(len(remote_content[path]) if path in remote_content else 0),
        lfs=None,
    )


def _upload_bytes(source: Any) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if hasattr(source, "read"):
        raw = source.read()
        return raw if isinstance(raw, bytes) else bytes(raw)
    return bytes(source)


def _apply_remote_operations(
    remote_files: set[str] | None,
    remote_content: dict[str, bytes],
    operations: Iterable[Any],
    serialized_operations: list[dict[str, Any]],
) -> None:
    for original, operation in zip(operations, serialized_operations, strict=True):
        if operation["action"] == "add":
            data = _operation_bytes(original)
            remote_content[operation["path_in_repo"]] = data
            if remote_files is not None:
                remote_files.add(operation["path_in_repo"])
        else:
            remote_content.pop(operation["path_in_repo"], None)
            if remote_files is not None:
                remote_files.discard(operation["path_in_repo"])


def _operation_bytes(operation: Any) -> bytes:
    source = getattr(operation, "path_or_fileobj", None)
    return _operation_source_bytes(source)


def _operation_source_bytes(source: Any) -> bytes:
    if isinstance(source, bytes):
        return source
    return _operation_non_bytes(source)


def _operation_non_bytes(source: Any) -> bytes:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    return _operation_stream_bytes(source)


def _operation_stream_bytes(source: Any) -> bytes:
    if hasattr(source, "read"):
        value = source.read()
        return value if isinstance(value, bytes) else bytes(value)
    return b""
