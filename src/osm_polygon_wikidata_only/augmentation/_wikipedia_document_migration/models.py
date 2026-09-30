"""Immutable plans and results for Wikipedia document migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MigrationOperation(StrEnum):
    """Classification of what action a stem requires."""

    CREATE_MISSING = "create_missing"
    UPGRADE_LEGACY = "upgrade_legacy"
    ALREADY_CANONICAL = "already_canonical"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class StemPlan:
    """Per-stem migration plan entry.

    Attributes
    ----------
    stem:
        Article stem name (filename without ``.parquet``).
    operation:
        What action this stem requires.
    reason:
        Empty for non-blocked operations. Descriptive error message for
        blocked stems, naming the problem without leaking absolute paths.
    article_hash:
        SHA-256 content hash of the article file at planning time.
        Empty string when no article file was found.
    document_hash:
        SHA-256 content hash of the document file at planning time.
        ``None`` when no document file existed.
    row_count:
        Number of canonical document rows, or zero for blocked stems.
    canonical_digest:
        Deterministic digest of the canonical schema and record batches.
        ``None`` for blocked stems. No dataset rows are retained in the plan.
    """

    stem: str
    operation: MigrationOperation
    reason: str
    article_hash: str
    document_hash: str | None
    row_count: int
    canonical_digest: str | None


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Immutable, validated migration plan.

    Built by :func:`plan_migration`. Passed to :func:`apply_migration`.
    """

    processed_dir: Path
    stems: tuple[StemPlan, ...]

    @property
    def is_safe_to_apply(self) -> bool:
        """True when no stems are blocked."""
        return all(s.operation != MigrationOperation.BLOCKED for s in self.stems)

    @property
    def blocked_stems(self) -> tuple[str, ...]:
        """Stems classified as blocked."""
        return tuple(s.stem for s in self.stems if s.operation == MigrationOperation.BLOCKED)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Deterministic result of applying a migration plan."""

    planned: int
    created: int
    upgraded: int
    skipped: int
    blocked: int
    created_stems: tuple[str, ...]
    upgraded_stems: tuple[str, ...]
    skipped_stems: tuple[str, ...]
    blocked_stems: tuple[str, ...]


class MigrationError(Exception):
    """Raised when migration planning or application fails."""
