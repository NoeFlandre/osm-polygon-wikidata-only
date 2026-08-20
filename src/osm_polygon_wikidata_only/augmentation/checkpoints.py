"""Durable checkpoints for long, otherwise atomic augmentation runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import Document, Section, WikidataFact
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps, loads

CHECKPOINT_CONTRACT_VERSION = "augmentation-checkpoints-v1"
SECTION_CHECKPOINT_BATCH_SIZE = 50


def augmentation_plan_key(
    *,
    core_hashes: dict[str, str],
    qids: tuple[str, ...],
    document_identities: tuple[tuple[str, int, str], ...],
) -> str:
    """Return the content key for every derived augmentation phase."""
    payload = dumps(
        {
            "contract_version": CHECKPOINT_CONTRACT_VERSION,
            "core_hashes": core_hashes,
            "qids": list(qids),
            "wikipedia_documents": [list(identity) for identity in document_identities],
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def entities_digest(entities: dict[str, dict[str, Any]]) -> str:
    """Return a deterministic digest of the exact upstream entity payload."""
    return hashlib.sha256(dumps(entities).encode("utf-8")).hexdigest()


def document_identities(
    documents: list[Document] | tuple[Document, ...],
) -> tuple[tuple[str, int, str], ...]:
    """Return ordered identities that invalidate section checkpoints."""
    return tuple(
        (document.document_id, document.revision_id, document.content_hash)
        for document in documents
    )


def _section_batch_expected_documents(
    metadata: dict[str, Any] | None,
) -> tuple[tuple[str, int, str], ...] | None:
    """Normalize section-batch document identities from checkpoint metadata."""
    if metadata is None:
        return None
    try:
        return _section_document_identities(metadata.get("documents", ()))
    except (TypeError, ValueError):
        return None


def _section_document_identities(values: Any) -> tuple[tuple[str, int, str], ...]:
    """Normalize the valid list-shaped identities from a metadata value."""
    return tuple(
        identity for value in values if (identity := _section_document_identity(value)) is not None
    )


def _section_document_identity(value: object) -> tuple[str, int, str] | None:
    """Normalize one section document identity or ignore its shape."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    return str(value[0]), int(cast(str | int, value[1])), str(value[2])


def _section_batch_rows(
    rows: list[dict[str, Any]] | None,
) -> list[Section] | None:
    """Construct typed sections from a checkpoint table or reject it."""
    if rows is None:
        return None
    try:
        return [Section(**row) for row in rows]
    except (TypeError, ValueError):
        return None


def _validated_section_batch(
    sections: list[Section],
    expected_documents: tuple[tuple[str, int, str], ...],
) -> list[Section] | None:
    """Validate section identities and return deterministic ordering."""
    if not _section_ids_are_known(sections, expected_documents):
        return None
    if not _section_ids_are_unique(sections):
        return None
    sections.sort(key=lambda row: (row.document_id, row.section_index))
    return sections


def _section_ids_are_known(
    sections: list[Section],
    expected_documents: tuple[tuple[str, int, str], ...],
) -> bool:
    """Return whether every section belongs to an expected document."""
    expected_ids = {document_id for document_id, _, _ in expected_documents}
    return all(section.document_id in expected_ids for section in sections)


def _section_ids_are_unique(sections: list[Section]) -> bool:
    """Return whether a section batch has no repeated section identity."""
    return len({section.section_id for section in sections}) == len(sections)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read one checkpoint JSON object, returning ``None`` on corruption."""
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def _validated_entity_payload(
    raw: object,
    expected_qids: tuple[str, ...],
) -> dict[str, dict[str, Any]] | None:
    """Validate an entity checkpoint's keys and object-valued payloads."""
    if not _entity_keys_are_expected(raw, expected_qids):
        return None
    if not isinstance(raw, dict) or not _entity_values_are_objects(cast(dict[object, object], raw)):
        return None
    return cast(dict[str, dict[str, Any]], raw)


def _entity_keys_are_expected(raw: object, expected_qids: tuple[str, ...]) -> bool:
    """Return whether an entity payload has only requested QID keys."""
    return isinstance(raw, dict) and set(raw).issubset(expected_qids)


def _entity_values_are_objects(raw: dict[object, object]) -> bool:
    """Return whether every entity payload value is a JSON object."""
    return all(isinstance(key, str) and isinstance(value, dict) for key, value in raw.items())


def _validated_voyage_documents(rows: list[dict[str, Any]] | None) -> list[Document] | None:
    """Construct and validate a deterministic Wikivoyage document checkpoint."""
    if rows is None:
        return None
    documents = _documents_from_rows(rows)
    if documents is None:
        return None
    if not _all_wikivoyage_documents(documents):
        return None
    if not _document_ids_are_unique(documents):
        return None
    return sorted(documents, key=lambda row: row.document_id)


def _documents_from_rows(rows: list[dict[str, Any]]) -> list[Document] | None:
    """Construct typed documents, rejecting malformed parquet rows."""
    try:
        return [Document(**row) for row in rows]
    except (TypeError, ValueError):
        return None


def _all_wikivoyage_documents(documents: list[Document]) -> bool:
    """Return whether every document belongs to the Wikivoyage project."""
    return all(document.project == "wikivoyage" for document in documents)


def _document_ids_are_unique(documents: list[Document]) -> bool:
    """Return whether a document collection has unique stable identities."""
    return len({document.document_id for document in documents}) == len(documents)


def _validate_augmentation_stem(stem: str) -> str:
    """Validate and return a single-region checkpoint stem."""
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid augmentation checkpoint stem: {stem!r}")
    return stem


def _validate_plan_key(plan_key: str) -> str:
    """Validate and return a SHA-256 augmentation plan key."""
    if len(plan_key) != 64 or any(character not in "0123456789abcdef" for character in plan_key):
        raise ValueError("Invalid augmentation checkpoint plan key")
    return plan_key


class AugmentationCheckpointStore:
    """Persist complete phase artifacts below one operator data-root cache."""

    def __init__(self, root: Path, stem: str, plan_key: str) -> None:
        self.region_root = root / _validate_augmentation_stem(stem)
        self.plan_root = self.region_root / _validate_plan_key(plan_key)
        try:
            self.plan_root.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("Augmentation checkpoint path escapes its cache root") from error

    def load_entities(self, expected_qids: tuple[str, ...]) -> dict[str, dict[str, Any]] | None:
        directory = self.plan_root / "entities"
        metadata = self._metadata(directory)
        if metadata is None or tuple(metadata.get("qids", ())) != expected_qids:
            return None
        return _validated_entity_payload(
            _read_json_object(directory / "entities.json"), expected_qids
        )

    def save_entities(
        self,
        qids: tuple[str, ...],
        entities: dict[str, dict[str, Any]],
    ) -> Path:
        if not set(entities).issubset(qids):
            raise ValueError("Entity checkpoint contains an unexpected QID")

        def write(directory: Path) -> None:
            atomic_write_text(directory / "entities.json", dumps(entities) + "\n")
            self._write_metadata(directory, {"qids": list(qids)})

        return self._save_directory("entities", write)

    def load_voyage_documents(self, expected_entities_digest: str) -> list[Document] | None:
        directory = self.plan_root / "wikivoyage_documents"
        if not self._matches(directory, "entities_digest", expected_entities_digest):
            return None
        return _validated_voyage_documents(
            self._read_table(directory / "documents.parquet", document_schema())
        )

    def save_voyage_documents(
        self,
        entity_digest: str,
        documents: list[Document],
    ) -> Path:
        def write(directory: Path) -> None:
            self._write_table(
                directory / "documents.parquet",
                [document.to_dict() for document in documents],
                DOCUMENT_COLUMNS,
                document_schema(),
            )
            self._write_metadata(directory, {"entities_digest": entity_digest})

        return self._save_directory("wikivoyage_documents", write)

    def load_section_batch(
        self,
        index: int,
        expected_documents: tuple[tuple[str, int, str], ...],
    ) -> list[Section] | None:
        directory = self._section_batch_path(index)
        expected = _section_batch_expected_documents(self._metadata(directory))
        if expected is None or expected != expected_documents:
            return None
        sections = _section_batch_rows(
            self._read_table(directory / "sections.parquet", section_schema())
        )
        return None if sections is None else _validated_section_batch(sections, expected_documents)

    def save_section_batch(
        self,
        index: int,
        documents: tuple[tuple[str, int, str], ...],
        sections: list[Section],
    ) -> Path:
        relative = f"sections/batch-{self._validate_index(index):06d}"

        def write(directory: Path) -> None:
            self._write_table(
                directory / "sections.parquet",
                [section.to_dict() for section in sections],
                SECTION_COLUMNS,
                section_schema(),
            )
            self._write_metadata(
                directory,
                {"documents": [list(identity) for identity in documents]},
            )

        return self._save_directory(relative, write)

    def load_facts(self, expected_entities_digest: str) -> list[WikidataFact] | None:
        directory = self.plan_root / "wikidata_facts"
        if not self._matches(directory, "entities_digest", expected_entities_digest):
            return None
        rows = self._read_table(directory / "facts.parquet", fact_schema())
        if rows is None:
            return None
        try:
            facts = [WikidataFact(**row) for row in rows]
        except (TypeError, ValueError):
            return None
        return sorted(facts, key=lambda row: row.fact_id)

    def save_facts(self, entity_digest: str, facts: list[WikidataFact]) -> Path:
        def write(directory: Path) -> None:
            self._write_table(
                directory / "facts.parquet",
                [fact.to_dict() for fact in facts],
                FACT_COLUMNS,
                fact_schema(),
            )
            self._write_metadata(directory, {"entities_digest": entity_digest})

        return self._save_directory("wikidata_facts", write)

    def clear(self) -> None:
        """Remove checkpoints for this region only after durable completion."""
        shutil.rmtree(self.region_root, ignore_errors=True)

    def _save_directory(self, relative: str, write: Callable[[Path], None]) -> Path:
        target = self.plan_root / relative
        self.plan_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}-",
                dir=target.parent if target.parent.exists() else self.plan_root,
            )
        )
        try:
            write(temporary)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def _metadata(self, directory: Path) -> dict[str, Any] | None:
        try:
            raw = loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(raw, dict) or raw.get("contract_version") != CHECKPOINT_CONTRACT_VERSION:
            return None
        return raw

    def _matches(self, directory: Path, key: str, expected: str) -> bool:
        metadata = self._metadata(directory)
        return metadata is not None and metadata.get(key) == expected

    @staticmethod
    def _write_metadata(directory: Path, values: dict[str, Any]) -> None:
        atomic_write_text(
            directory / "metadata.json",
            dumps({"contract_version": CHECKPOINT_CONTRACT_VERSION, **values}) + "\n",
        )

    @staticmethod
    def _read_table(path: Path, schema: pa.Schema) -> list[dict[str, Any]] | None:
        try:
            actual: pa.Schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
            if not actual.equals(schema, check_metadata=True):
                return None
            table: pa.Table = pq.read_table(path)  # type: ignore[no-untyped-call]
            rows: list[dict[str, Any]] = table.to_pylist()
            return rows
        except (OSError, ValueError, TypeError, pa.ArrowException):
            return None

    @staticmethod
    def _write_table(
        path: Path,
        rows: list[dict[str, Any]],
        columns: tuple[str, ...],
        schema: pa.Schema,
    ) -> None:
        normalized = [{column: row.get(column) for column in columns} for row in rows]
        pq.write_table(pa.Table.from_pylist(normalized, schema=schema), path, compression="snappy")  # type: ignore[no-untyped-call]

    def _section_batch_path(self, index: int) -> Path:
        return self.plan_root / "sections" / f"batch-{self._validate_index(index):06d}"

    @staticmethod
    def _validate_index(index: int) -> int:
        if index < 0:
            raise ValueError("Section checkpoint index must be non-negative")
        return index


__all__: list[str] = []
