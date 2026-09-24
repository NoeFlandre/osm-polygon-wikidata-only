"""Hub snapshot helpers shared by the dataset release paths."""

from __future__ import annotations

from typing import Any

__all__ = ["read_repo_sha"]


def read_repo_sha(hub: Any, repo_id: str) -> object:
    """Return the raw ``repo_info().sha`` of a dataset repository.

    Callers validate the value themselves because they differ in whether a
    missing revision is fatal.
    """
    info = hub.repo_info(repo_id, repo_type="dataset")
    return getattr(info, "sha", None)
