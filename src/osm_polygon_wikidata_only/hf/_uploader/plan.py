"""Publication-op model for atomic HF commits.

The dataset publication layer used to assemble
``list[tuple[Path, str]]`` tuples. The first publication that
unified the augmentation manifest under ``manifests/`` needs to
DELETE the legacy path atomically in the SAME commit as the new
upload. A richer operation model is required at the
publication / uploader boundary:

* ``PublicationOp`` is a dataclass with ``action`` ∈
  ``{"add", "delete"}`` and ``path_in_repo``.
* For ``action == "add"`` the local file path is
  ``PublicationOp.local_path``.
* For ``action == "delete"`` there is no local file path.

The assemblers in :mod:`hf.publication` return a ``list[PublicationOp]``.
The uploader (:func:`upload_files`) accepts ``ops=[...]`` and turns
the list into the corresponding ``huggingface_hub`` commit operations
inside ONE atomic commit. CLI callers do not need to know about the
op model -- they pass the assembled list straight through.

The model is intentionally minimal so the publication layer stays
pure: there is no network ownership here, no schema validation,
and no side effects on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

Action = str  # "add" | "delete"


@dataclass(frozen=True, slots=True)
class PublicationOp:
    """A single atomic-commit operation: add a file or delete a path."""

    action: Action
    path_in_repo: str
    local_path: Path | None = None

    def __post_init__(self) -> None:
        if self.action not in {"add", "delete"}:
            raise ValueError(f"PublicationOp.action must be 'add' or 'delete', got {self.action!r}")
        if self.action == "add" and self.local_path is None:
            raise ValueError(
                f"PublicationOp(action='add', path_in_repo={self.path_in_repo!r}) "
                "requires a local_path"
            )
        if self.action == "delete" and self.local_path is not None:
            raise ValueError(
                f"PublicationOp(action='delete', path_in_repo={self.path_in_repo!r}) "
                "must not carry a local_path"
            )


def add_op(local_path: str | Path, *, path_in_repo: str) -> PublicationOp:
    """Build an ``add`` op from a local file path and a remote path."""
    return PublicationOp(
        action="add",
        path_in_repo=path_in_repo,
        local_path=Path(local_path),
    )


def delete_op(path_in_repo: str) -> PublicationOp:
    """Build a ``delete`` op for a remote path that is no longer canonical."""
    return PublicationOp(action="delete", path_in_repo=path_in_repo)


__all__ = ["PublicationOp", "add_op", "delete_op"]
