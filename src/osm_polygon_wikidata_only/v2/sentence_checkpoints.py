"""Atomic, resumable Parquet batches for V2 sentence segmentation."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.sentence_logic import sentence_schema

SENTENCE_CHECKPOINT_CONTRACT_VERSION = "v2-sentence-checkpoints-v1"
_CHECKPOINT_PAYLOAD_PATTERNS = ("batch-*.parquet", "*.tmp", ".*.tmp")


class SentenceCheckpoint:
    """Append-only batches for one region/project sentence pass."""

    def __init__(
        self,
        root: Path,
        stem: str,
        project: str,
        *,
        input_fingerprint: str,
        model_id: str,
        model_revision: str,
        batch_size: int,
    ) -> None:
        _validate_component(stem, "stem")
        if project not in {"wikipedia", "wikivoyage"}:
            raise ValueError(f"Unsupported sentence project: {project!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.root = Path(root) / stem / project
        self.metadata_path = self.root / "metadata.json"
        self.identity = {
            "contract_version": SENTENCE_CHECKPOINT_CONTRACT_VERSION,
            "stem": stem,
            "project": project,
            "input_fingerprint": input_fingerprint,
            "model_id": model_id,
            "model_revision": model_revision,
            "batch_size": batch_size,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata = self._prepare()

    def _prepare(self) -> dict[str, Any]:
        raw = self._read_metadata()
        if not isinstance(raw, dict) or raw.get("identity") != self.identity:
            self._reset()
            return self._initial_metadata()
        return cast(dict[str, Any], raw)

    def _read_metadata(self) -> object:
        try:
            return json_loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, UnicodeError, ValueError):
            return None

    def _initial_metadata(self) -> dict[str, Any]:
        metadata = {
            "identity": self.identity,
            "complete": False,
            "batch_count": 0,
            "row_count": 0,
        }
        _atomic_write_json(self.metadata_path, metadata)
        return metadata

    def _reset(self) -> None:
        self._remove_payloads()

    def _remove_payloads(self) -> None:
        for pattern in _CHECKPOINT_PAYLOAD_PATTERNS:
            for path in self.root.glob(pattern):
                path.unlink(missing_ok=True)

    @property
    def complete(self) -> bool:
        """Return whether all source batches were finalized."""
        return bool(self._metadata.get("complete", False))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the persisted checkpoint metadata."""
        return dict(self._metadata)

    def _batch_path(self, index: int) -> Path:
        if index < 0:
            raise ValueError("batch index must be non-negative")
        return self.root / f"batch-{index:08d}.parquet"

    @property
    def completed_batches(self) -> tuple[int, ...]:
        """Return indexes of valid, already-written batches."""
        completed: list[int] = []
        for path in sorted(self.root.glob("batch-*.parquet")):
            index = _batch_index(path)
            if index is not None and self.load_batch_table(index) is not None:
                completed.append(index)
        return tuple(completed)

    def load_batch_table(self, index: int) -> pa.Table | None:
        """Read one batch as a schema-validated Arrow table."""
        path = self._batch_path(index)
        try:
            with pq.ParquetFile(path) as parquet_file:
                table = parquet_file.read()
        except (OSError, pa.ArrowException):
            return None
        if not table.schema.equals(sentence_schema(), check_metadata=True):
            return None
        return table

    def load_batch(self, index: int) -> list[dict[str, Any]] | None:
        """Read one batch, returning ``None`` for a missing or invalid file."""
        table = self.load_batch_table(index)
        if table is None:
            return None
        return table.to_pylist()

    def load_rows(self, *, batch_count: int | None = None) -> list[dict[str, Any]]:
        """Read contiguous checkpoint batches in deterministic order."""
        expected = batch_count
        if expected is None and self.complete:
            expected = self._metadata.get("batch_count")
        indexes = self.completed_batches
        if expected is None:
            expected = max(indexes, default=-1) + 1
        if not isinstance(expected, int) or expected < 0:
            raise ValueError("Invalid sentence checkpoint batch count")
        if indexes != tuple(range(expected)):
            raise ValueError("Sentence checkpoint batches are not contiguous")
        rows: list[dict[str, Any]] = []
        for index in indexes:
            batch = self.load_batch(index)
            if batch is None:
                raise ValueError(f"Invalid sentence checkpoint batch: {index}")
            rows.extend(batch)
        return rows

    def write_batch(self, index: int, rows: list[dict[str, Any]]) -> None:
        """Atomically write one completed batch, including an empty batch."""
        path = self._batch_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary = Path(raw)
        try:
            normalized = [
                {field.name: row.get(field.name) for field in sentence_schema()} for row in rows
            ]
            pq.write_table(
                pa.Table.from_pylist(normalized, schema=sentence_schema()),
                temporary,
                compression="snappy",
            )  # type: ignore[no-untyped-call]
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._metadata = {
            **self._metadata,
            "complete": False,
        }
        _atomic_write_json(self.metadata_path, self._metadata)

    def mark_complete(self, *, batch_count: int, row_count: int) -> None:
        """Record complete accounting only after every batch is present."""
        indexes = self.completed_batches
        if indexes != tuple(range(batch_count)):
            raise ValueError("Sentence checkpoint batches are not contiguous")
        actual_row_count = sum(
            int(pq.read_metadata(self._batch_path(index)).num_rows) for index in indexes
        )
        if actual_row_count != row_count:
            raise ValueError("Sentence checkpoint row accounting is invalid")
        self._metadata = {
            **self._metadata,
            "complete": True,
            "batch_count": batch_count,
            "row_count": row_count,
        }
        _atomic_write_json(self.metadata_path, self._metadata)

    def finalize(
        self,
        output_path: Path,
        *,
        output_hash: str,
        summary: dict[str, Any],
    ) -> None:
        """Record the final output and remove duplicate batch payloads."""
        if not self.complete:
            raise ValueError("Cannot finalize an incomplete sentence checkpoint")
        self._metadata = {
            **self._metadata,
            "output_path": str(output_path),
            "output_hash": output_hash,
            "summary": dict(summary),
        }
        _atomic_write_json(self.metadata_path, self._metadata)
        self._remove_payloads()

    def output_matches(self, output_path: Path, *, output_hash: str) -> bool:
        """Return whether the recorded final output still matches on disk."""
        stored_path = self._metadata.get("output_path")
        stored_hash = self._metadata.get("output_hash")
        return bool(
            self.complete
            and stored_path == str(output_path)
            and stored_hash == output_hash
            and output_path.is_file()
            and sha256_file(output_path) == stored_hash
        )

    def reset(self) -> None:
        """Discard this contract's partial state and start a fresh pass."""
        self._reset()
        self._metadata = self._initial_metadata()

    def clear(self) -> None:
        """Remove only this checkpoint's files after final publication."""
        self._remove_payloads()
        self.metadata_path.unlink(missing_ok=True)
        with suppress(OSError):
            self.root.rmdir()
        with suppress(OSError):
            self.root.parent.rmdir()
        with suppress(OSError):
            self.root.parent.parent.rmdir()


def _batch_index(path: Path) -> int | None:
    try:
        return int(path.stem.removeprefix("batch-"))
    except ValueError:
        return None


def _validate_component(value: str, name: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid sentence checkpoint {name}: {value!r}")


def _atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_dumps(value) + "\n")


__all__ = ["SENTENCE_CHECKPOINT_CONTRACT_VERSION", "SentenceCheckpoint"]
