"""Shared release value types and errors."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HF_DATASET_COMMIT_URL = re.compile(
    r"https://huggingface\.co/datasets/[^/]+/[^/]+/commit/(?P<revision>[0-9a-f]{40})"
)


class StatsReleaseError(RuntimeError):
    """Raised when a statistics release cannot be planned or verified."""


@dataclass(frozen=True)
class ReleasedFile:
    """One published metadata file and its exact identity."""

    path_in_repo: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RemoteState:
    revision: str | None
    contents: dict[str, bytes]


def hub_revision(revision: str) -> str:
    match = _HF_DATASET_COMMIT_URL.fullmatch(revision)
    return match.group("revision") if match else revision


__all__ = ["ReleasedFile", "StatsReleaseError"]
