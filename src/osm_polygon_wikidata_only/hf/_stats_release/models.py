"""Shared release value types and errors."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub

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


class RemoteVerifier(Protocol):
    """Confirm the released files exist at the uploaded revision."""

    def __call__(
        self,
        repo_id: str,
        files: tuple[ReleasedFile, ...],
        *,
        revision: str,
    ) -> str: ...


@dataclass(frozen=True)
class RemoteState:
    revision: str | None
    contents: dict[str, bytes]


@dataclass(frozen=True)
class ReleaseOptions:
    """Publication mode, Hub access and revision pins for one statistics release.

    ``apply=False`` plans the release and never touches the Hub.
    """

    apply: bool = False
    hub: HfHub | None = None
    verifier: RemoteVerifier | None = None
    token: str | None = None
    source_revision: str | None = None
    data_revision: str | None = None


@dataclass(frozen=True)
class StatsReleaseReport:
    """Evidence for one statistics release run."""

    repo_id: str
    processed_dir: str
    files: tuple[ReleasedFile, ...]
    polygon_files: int
    polygon_rows: int
    published: bool
    revision: str | None
    provenance: dict[str, Any] = field(default_factory=dict)
    committed: bool = False
    no_op: bool = False
    changed_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "committed": self.committed,
            "files": [
                {
                    "path_in_repo": item.path_in_repo,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
            "no_op": self.no_op,
            "polygon_files": self.polygon_files,
            "polygon_rows": self.polygon_rows,
            "processed_dir": self.processed_dir,
            "provenance": self.provenance,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }


def hub_revision(revision: str) -> str:
    match = _HF_DATASET_COMMIT_URL.fullmatch(revision)
    return match.group("revision") if match else revision


__all__ = ["ReleaseOptions", "ReleasedFile", "StatsReleaseError", "StatsReleaseReport"]
