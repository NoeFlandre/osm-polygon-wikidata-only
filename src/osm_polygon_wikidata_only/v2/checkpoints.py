"""Durable, restart-safe checkpoints for the V2 workflow.

The published V2 files remain the commit boundary.  These checkpoints live
under the selected external data root and only contain work that can be
reconstructed from the source PBF and Wikimedia responses.  A stopped run
can therefore reuse completed extraction batches, direct page results, and
parsed section batches without treating a partial region as publishable.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.schema import polygon_v2_schema

CHECKPOINT_CONTRACT_VERSION = "wikipedia-tags-v2-checkpoints-v2"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def region_input_fingerprint(polygons: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Return a stable identity for one extracted region's fetch inputs.

    The identity includes the polygon context used to build link rows, not
    only the Wikipedia references.  This prevents a checkpoint from reusing
    a direct result with stale source-region or OSM identity fields after a
    source PBF changes while retaining the same tag reference.
    """
    context_fields = (
        "polygon_id",
        "wikipedia_tag_refs",
        "wikidata",
        "source_pbf",
        "region",
        "osm_type",
        "osm_id",
    )
    values = sorted(tuple(str(row.get(field, "")) for field in context_fields) for row in polygons)
    return _sha256_text(json_dumps(values))


def _atomic_write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(raw_tmp)
    try:
        pq.write_table(
            pa.Table.from_pylist(
                [{field.name: row.get(field.name) for field in schema} for row in rows],
                schema=schema,
            ),
            temporary,
            compression="snappy",
        )  # type: ignore[no-untyped-call]
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_dumps(value) + "\n")


class ExtractionCheckpoint:
    """Append-only Parquet batches for one source PBF extraction."""

    def __init__(
        self,
        root: Path,
        source_path: Path,
        *,
        limit: int | None = None,
    ) -> None:
        stem = source_path.name.removesuffix(".osm.pbf")
        self.root = Path(root) / "extraction" / stem
        self.metadata_path = self.root / "metadata.json"
        self.identity = {
            "contract_version": CHECKPOINT_CONTRACT_VERSION,
            "source": _file_fingerprint(source_path),
            "limit": limit,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata = self._prepare()
        self._next_chunk = (
            max(
                (int(path.stem.removeprefix("chunk-")) for path in self._chunks()),
                default=-1,
            )
            + 1
        )

    def _prepare(self) -> dict[str, Any]:
        try:
            raw = json_loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            raw = None
        if not isinstance(raw, dict) or raw.get("identity") != self.identity:
            self._clear_files()
            raw = {
                "identity": self.identity,
                "complete": False,
                "next_chunk": 0,
                "chunk_count": 0,
                "row_count": 0,
            }
            _atomic_write_json(self.metadata_path, raw)
        return raw

    def _clear_files(self) -> None:
        for path in self.root.glob("chunk-*.parquet"):
            path.unlink(missing_ok=True)
        for path in (*self.root.glob("*.tmp"), *self.root.glob(".*.tmp")):
            path.unlink(missing_ok=True)

    @property
    def complete(self) -> bool:
        return bool(self._metadata.get("complete", False))

    def _chunks(self) -> list[Path]:
        return sorted(self.root.glob("chunk-*.parquet"))

    def load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        schema = polygon_v2_schema()
        chunks = self._chunks()
        try:
            indices = [int(path.stem.removeprefix("chunk-")) for path in chunks]
            if indices != list(range(len(indices))):
                raise ValueError("Extraction checkpoint chunks are not contiguous")
            for path in chunks:
                with pq.ParquetFile(path) as parquet_file:
                    table = parquet_file.read()
                if not table.schema.equals(schema, check_metadata=True):
                    raise ValueError(f"Invalid extraction checkpoint schema: {path}")
                rows.extend(table.to_pylist())
            if self.complete and (
                not isinstance(self._metadata.get("chunk_count"), int)
                or not isinstance(self._metadata.get("row_count"), int)
                or self._metadata["chunk_count"] != len(chunks)
                or self._metadata["row_count"] != len(rows)
            ):
                raise ValueError("Complete extraction checkpoint accounting is invalid")
        except (OSError, ValueError, pa.ArrowException):
            self._clear_files()
            self._metadata = {
                "identity": self.identity,
                "complete": False,
                "next_chunk": 0,
                "chunk_count": 0,
                "row_count": 0,
            }
            self._next_chunk = 0
            _atomic_write_json(self.metadata_path, self._metadata)
            return []
        if not self.complete:
            self._metadata = {
                **self._metadata,
                "next_chunk": len(chunks),
                "chunk_count": len(chunks),
                "row_count": len(rows),
            }
            _atomic_write_json(self.metadata_path, self._metadata)
        return rows

    def append(self, rows: list[dict[str, Any]]) -> None:
        """Durably append one extraction batch before the source scan continues."""
        if not rows:
            return
        next_chunk = self._next_chunk
        path = self.root / f"chunk-{next_chunk:08d}.parquet"
        _atomic_write_table(path, rows, polygon_v2_schema())
        self._next_chunk += 1
        previous_rows = self._metadata.get("row_count", 0)
        if not isinstance(previous_rows, int):
            previous_rows = 0
        self._metadata = {
            **self._metadata,
            "complete": False,
            "next_chunk": next_chunk + 1,
            "chunk_count": next_chunk + 1,
            "row_count": previous_rows + len(rows),
        }
        _atomic_write_json(self.metadata_path, self._metadata)

    def mark_complete(self) -> None:
        chunks = self._chunks()
        self._metadata = {
            **self._metadata,
            "complete": True,
            "chunk_count": len(chunks),
        }
        _atomic_write_json(self.metadata_path, self._metadata)

    def clear(self) -> None:
        """Remove only this checkpoint's owned files after final publication."""
        self._clear_files()
        self.metadata_path.unlink(missing_ok=True)
        with suppress(OSError):
            self.root.rmdir()
        with suppress(OSError):
            self.root.parent.rmdir()


class RegionFetchCheckpoint:
    """Per-region direct-result and section checkpoints."""

    def __init__(
        self,
        root: Path,
        stem: str,
        *,
        input_fingerprint: str | None = None,
        fetch_full_text: bool = True,
    ) -> None:
        self.root = Path(root) / "fetch" / stem
        self.metadata_path = self.root / "metadata.json"
        self.direct_root = self.root / "direct"
        self.sections_root = self.root / "sections"
        self.root.mkdir(parents=True, exist_ok=True)
        self.direct_root.mkdir(exist_ok=True)
        self.sections_root.mkdir(exist_ok=True)
        self.input_fingerprint = input_fingerprint
        self.fetch_full_text = fetch_full_text
        self._prepare()

    def _prepare(self) -> None:
        try:
            raw = json_loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            raw = None
        invalid = (
            not isinstance(raw, dict)
            or raw.get("contract_version") != CHECKPOINT_CONTRACT_VERSION
            or raw.get("v2_contract_version") != V2_CONTRACT_VERSION
            or raw.get("fetch_full_text") != self.fetch_full_text
        )
        if invalid:
            self.clear()
            self.root.mkdir(parents=True, exist_ok=True)
            self.direct_root.mkdir(exist_ok=True)
            self.sections_root.mkdir(exist_ok=True)
            raw = {
                "contract_version": CHECKPOINT_CONTRACT_VERSION,
                "v2_contract_version": V2_CONTRACT_VERSION,
                "input_fingerprint": self.input_fingerprint,
                "fetch_full_text": self.fetch_full_text,
            }
            _atomic_write_json(self.metadata_path, raw)
        elif (
            self.input_fingerprint is not None
            and raw.get("input_fingerprint") != self.input_fingerprint
        ):
            self.clear()
            self.root.mkdir(parents=True, exist_ok=True)
            self.direct_root.mkdir(exist_ok=True)
            self.sections_root.mkdir(exist_ok=True)
            _atomic_write_json(
                self.metadata_path,
                {
                    "contract_version": CHECKPOINT_CONTRACT_VERSION,
                    "v2_contract_version": V2_CONTRACT_VERSION,
                    "input_fingerprint": self.input_fingerprint,
                    "fetch_full_text": self.fetch_full_text,
                },
            )

    @property
    def has_work(self) -> bool:
        return any(self.direct_root.glob("*.json")) or any(self.sections_root.glob("*.json"))

    @staticmethod
    def _key(value: str) -> str:
        return _sha256_text(value)

    def save_direct(self, polygon_id: str, refs: Any, result: Any) -> None:
        """Persist one complete polygon result; failed/deferred refs are retried."""
        if result.deferred_errors:
            return
        statuses = []
        for status in result.statuses:
            ref = status.ref
            statuses.append(
                {
                    "ref": {
                        "language": ref.language,
                        "title": ref.title,
                        "raw_key": ref.raw_key,
                        "raw_value": ref.raw_value,
                    },
                    "status": status.status,
                    "error": status.error,
                    "reused_v1": status.reused_v1,
                }
            )
        payload = {
            "polygon_id": polygon_id,
            "refs": [
                {
                    "language": ref.language,
                    "title": ref.title,
                    "raw_key": ref.raw_key,
                    "raw_value": ref.raw_value,
                }
                for ref in refs
            ],
            "documents": list(result.documents),
            "links": list(result.links),
            "statuses": statuses,
        }
        _atomic_write_json(self.direct_root / f"{self._key(polygon_id)}.json", payload)

    def load_direct(self, polygon_id: str, refs: Any) -> Any | None:
        path = self.direct_root / f"{self._key(polygon_id)}.json"
        try:
            payload = json_loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None
        expected_refs = [
            {
                "language": ref.language,
                "title": ref.title,
                "raw_key": ref.raw_key,
                "raw_value": ref.raw_value,
            }
            for ref in refs
        ]
        if not isinstance(payload, dict) or payload.get("polygon_id") != polygon_id:
            return None
        if payload.get("refs") != expected_refs:
            return None
        statuses_raw = payload.get("statuses")
        if not isinstance(statuses_raw, list) or len(statuses_raw) != len(refs):
            return None
        from osm_polygon_wikidata_only.v2.direct_enrichment import (
            DirectEnrichmentResult,
            DirectWikipediaStatus,
        )
        from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef

        statuses = []
        for raw, ref in zip(statuses_raw, refs, strict=True):
            if not isinstance(raw, dict):
                return None
            statuses.append(
                DirectWikipediaStatus(
                    WikipediaTagRef(ref.language, ref.title, ref.raw_key, ref.raw_value),
                    str(raw.get("status", "")),
                    str(raw.get("error", "")),
                    bool(raw.get("reused_v1", False)),
                )
            )
        documents = payload.get("documents", [])
        links = payload.get("links", [])
        if not isinstance(documents, list) or not isinstance(links, list):
            return None
        if not all(isinstance(row, dict) for row in (*documents, *links)):
            return None
        return DirectEnrichmentResult(
            documents=tuple(documents),
            links=tuple(links),
            statuses=tuple(statuses),
        )

    def save_sections(self, document_id: str, rows: list[dict[str, Any]]) -> None:
        if any(str(row.get("document_id", "")) != document_id for row in rows):
            raise ValueError(f"Section checkpoint rows do not match document {document_id!r}")
        _atomic_write_json(
            self.sections_root / f"{self._key(document_id)}.json",
            {"document_id": document_id, "rows": rows},
        )

    def load_section_state(self) -> tuple[list[dict[str, Any]], set[str]]:
        """Load section rows and completed document IDs in one directory pass.

        Empty section results are represented by a checkpoint file with no
        rows, so callers need both the rows and the set of completed document
        IDs.  Returning both together avoids parsing every checkpoint file
        twice during a resumed region.
        """
        rows: list[dict[str, Any]] = []
        completed: set[str] = set()
        for path in sorted(self.sections_root.glob("*.json")):
            payload = self._load_section_payload(path)
            if payload is None:
                continue
            completed.add(payload["document_id"])
            rows.extend(row for row in payload["rows"] if isinstance(row, dict))
        return rows, completed

    def load_sections(self) -> list[dict[str, Any]]:
        """Load all valid section rows from the checkpoint directory."""
        return self.load_section_state()[0]

    def section_document_ids(self) -> set[str]:
        """Return documents whose section fetch completed, including empty results."""
        return self.load_section_state()[1]

    @staticmethod
    def _load_section_payload(path: Path) -> dict[str, Any] | None:
        try:
            payload = json_loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        document_id = payload.get("document_id")
        rows = payload.get("rows")
        if not isinstance(document_id, str) or not document_id or not isinstance(rows, list):
            return None
        if any(
            not isinstance(row, dict) or str(row.get("document_id", "")) != document_id
            for row in rows
        ):
            return None
        return {"document_id": document_id, "rows": rows}

    def clear(self) -> None:
        for directory in (self.direct_root, self.sections_root):
            if not directory.exists():
                continue
            for path in (
                *directory.glob("*.json"),
                *directory.glob("*.tmp"),
                *directory.glob(".*.tmp"),
            ):
                path.unlink(missing_ok=True)
            with suppress(OSError):
                directory.rmdir()
        for path in (*self.root.glob("*.tmp"), *self.root.glob(".*.tmp")):
            path.unlink(missing_ok=True)
        self.metadata_path.unlink(missing_ok=True)
        with suppress(OSError):
            self.root.rmdir()
        with suppress(OSError):
            self.root.parent.rmdir()


def clear_v2_checkpoints(root: Path, stem: str) -> None:
    """Remove one region's extraction and fetch checkpoints if present."""
    extraction_root = Path(root) / "extraction" / stem
    for path in extraction_root.glob("chunk-*.parquet"):
        path.unlink(missing_ok=True)
    for path in (*extraction_root.glob("*.tmp"), *extraction_root.glob(".*.tmp")):
        path.unlink(missing_ok=True)
    (extraction_root / "metadata.json").unlink(missing_ok=True)
    with suppress(OSError):
        extraction_root.rmdir()
    with suppress(OSError):
        extraction_root.parent.rmdir()
    fetch_root = Path(root) / "fetch" / stem
    for directory in (fetch_root / "direct", fetch_root / "sections"):
        for path in (
            *directory.glob("*.json"),
            *directory.glob("*.tmp"),
            *directory.glob(".*.tmp"),
        ):
            path.unlink(missing_ok=True)
        with suppress(OSError):
            directory.rmdir()
    for path in (*fetch_root.glob("*.tmp"), *fetch_root.glob(".*.tmp")):
        path.unlink(missing_ok=True)
    (fetch_root / "metadata.json").unlink(missing_ok=True)
    with suppress(OSError):
        fetch_root.rmdir()
    with suppress(OSError):
        fetch_root.parent.rmdir()


__all__ = [
    "CHECKPOINT_CONTRACT_VERSION",
    "ExtractionCheckpoint",
    "RegionFetchCheckpoint",
    "clear_v2_checkpoints",
    "region_input_fingerprint",
]
