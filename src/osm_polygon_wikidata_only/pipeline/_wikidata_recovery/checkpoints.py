"""Durable, content-addressed checkpoints for long Wikidata recovery work."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import (
    FACT_COLUMNS,
    SECTION_COLUMNS,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps, loads

CHECKPOINT_CONTRACT_VERSION = "wikidata-recovery-batches-v1"
RECOVERY_QID_BATCH_SIZE = 25


@dataclass(frozen=True, slots=True)
class RecoveryBatchArtifacts:
    qids: tuple[str, ...]
    documents: tuple[dict[str, Any], ...]
    sections: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]


def recovery_plan_key(
    *,
    fingerprints: tuple[tuple[str, str], ...],
    affected_qids: tuple[str, ...],
    sections_hash: str,
    settings_identity: tuple[object, ...],
) -> str:
    """Return a stable key that invalidates checkpoints when any input changes."""
    payload = dumps(
        {
            "contract_version": CHECKPOINT_CONTRACT_VERSION,
            "fingerprints": list(fingerprints),
            "affected_qids": list(affected_qids),
            "sections_hash": sections_hash,
            "settings": list(settings_identity),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RecoveryCheckpointStore:
    """Persist complete batch artifacts; incomplete directories are never reusable."""

    def __init__(self, root: Path, stem: str, plan_key: str) -> None:
        if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
            raise ValueError(f"Invalid recovery checkpoint stem: {stem!r}")
        self._region_root = root / stem
        self._plan_root = self._region_root / plan_key

    def load(self, index: int, expected_qids: tuple[str, ...]) -> RecoveryBatchArtifacts | None:
        directory = self._batch_path(index)
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            raw = loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            if raw.get("contract_version") != CHECKPOINT_CONTRACT_VERSION:
                return None
            qids = tuple(str(value) for value in raw.get("qids", ()))
            if qids != expected_qids:
                return None
            documents = self._read(directory / "documents.parquet", wikipedia_document_schema())
            sections = self._read(directory / "sections.parquet", section_schema())
            facts = self._read(directory / "facts.parquet", fact_schema())
        except (OSError, ValueError, TypeError, pa.ArrowException):
            return None
        return RecoveryBatchArtifacts(qids, tuple(documents), tuple(sections), tuple(facts))

    def save(self, index: int, artifacts: RecoveryBatchArtifacts) -> Path:
        target = self._batch_path(index)
        existing = self.load(index, artifacts.qids)
        if existing is not None:
            if existing != artifacts:
                raise RuntimeError(f"Recovery checkpoint conflicts with completed batch {index}")
            return target
        self._plan_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".batch-{index:06d}-", dir=self._plan_root))
        try:
            self._write(
                temporary / "documents.parquet",
                artifacts.documents,
                WIKIPEDIA_DOCUMENT_COLUMNS,
                wikipedia_document_schema(),
            )
            self._write(
                temporary / "sections.parquet",
                artifacts.sections,
                SECTION_COLUMNS,
                section_schema(),
            )
            self._write(temporary / "facts.parquet", artifacts.facts, FACT_COLUMNS, fact_schema())
            atomic_write_text(
                temporary / "metadata.json",
                dumps(
                    {
                        "contract_version": CHECKPOINT_CONTRACT_VERSION,
                        "qids": list(artifacts.qids),
                    }
                )
                + "\n",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def clear(self) -> None:
        shutil.rmtree(self._region_root, ignore_errors=True)

    def _batch_path(self, index: int) -> Path:
        if index < 0:
            raise ValueError("Recovery checkpoint index must be non-negative")
        return self._plan_root / f"batch-{index:06d}"

    @staticmethod
    def _read(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
        actual: pa.Schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        if not actual.equals(schema, check_metadata=True):
            raise ValueError(f"Recovery checkpoint schema mismatch: {path}")
        table: pa.Table = pq.read_table(path)  # type: ignore[no-untyped-call]
        rows: list[dict[str, Any]] = table.to_pylist()
        return rows

    @staticmethod
    def _write(
        path: Path,
        rows: tuple[dict[str, Any], ...],
        columns: tuple[str, ...],
        schema: pa.Schema,
    ) -> None:
        normalized = [{column: row.get(column) for column in columns} for row in rows]
        pq.write_table(pa.Table.from_pylist(normalized, schema=schema), path, compression="snappy")  # type: ignore[no-untyped-call]


__all__: list[str] = []
