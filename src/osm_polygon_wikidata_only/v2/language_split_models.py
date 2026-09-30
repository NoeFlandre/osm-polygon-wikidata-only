"""Shared data types and tuning constants for V2 language split generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.language_splits import LanguageInventory, LanguageTable

DEFAULT_BATCH_SIZE = 65_536


DEFAULT_MAX_ROWS_PER_SHARD = 100_000


LANGUAGE_SPLITS_DIRNAME = "language_splits"


LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH = Path("manifests/language_splits.json")


V2_LANGUAGE_SPLIT_CONTRACT_VERSION = "v2-language-splits-v2"


SHARD_FLUSH_BYTES = 16 * 1024 * 1024


TOTAL_FLUSH_BYTES = 192 * 1024 * 1024


class V2LanguageSplitError(ValueError):
    """Raised when a V2 partition cannot conserve or validate source rows."""


@dataclass(frozen=True, slots=True)
class V2LanguageSplitFile:
    """One deterministic output shard and its source-row accounting."""

    table: LanguageTable
    configuration: str
    language: str
    split: str
    source_files: tuple[str, ...]
    path: str
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable manifest representation of this shard."""
        return {
            "table": self.table.value,
            "configuration": self.configuration,
            "language": self.language,
            "split": self.split,
            "source_files": list(self.source_files),
            "path": self.path,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class V2LanguageSplitResult:
    """Published V2 language partitions and their deterministic manifest."""

    processed_root: Path
    output_root: Path
    manifest_path: Path
    inventory: LanguageInventory
    files: tuple[V2LanguageSplitFile, ...]


@dataclass(slots=True)
class LanguageShardWriteState:
    """Mutable state for one staged language shard."""

    language: str
    shard_index: int
    writer: pq.ParquetWriter
    final_path: Path
    staged_path: Path
    row_count: int
    source_files: list[str]
    pending: list[pa.RecordBatch] = field(default_factory=list)
    pending_bytes: int = 0


@dataclass(slots=True)
class LanguageTableWriteState:
    """Persistent per-language writers for one language-bearing table."""

    current: dict[str, LanguageShardWriteState]
    next_shard_index: dict[str, int]
    shards: list[LanguageShardWriteState]
    staged_paths: dict[Path, Path]
    pending_bytes: int = 0
