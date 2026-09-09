"""Bounded geographic NER with verified checkpoints and persistent inference reuse."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_json, atomic_write_parquet
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.run_lock import exclusive_run_lock

MODEL_ID = "whoisjones/otter-cross-mmbert"
MODEL_REVISION = "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d"
LABEL = "named geographic location"
INPUT_COLUMNS = ("sentence_id", "document_id", "project", "language", "text", "segmentation_status")
_IDENTITY_COLUMNS = INPUT_COLUMNS[:4]
_ENTITY = pa.struct(
    [
        ("text", pa.string()),
        ("label", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("score", pa.float64()),
    ]
)
OUTPUT_SCHEMA = pa.schema(
    [
        *((name, pa.string()) for name in _IDENTITY_COLUMNS),
        ("status", pa.string()),
        ("validation_status", pa.string()),
        ("contract_id", pa.string()),
        ("entities", pa.list_(_ENTITY)),
    ]
)


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Contract:
    """An immutable language, label, model and threshold selection."""

    languages: tuple[str, ...]
    threshold: float = 0.5
    validation_status: str = "pilot_unvalidated"
    implementation_revision: str = "geographic-ner-v1"
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    label: str = LABEL

    def __post_init__(self) -> None:
        _validate_languages(self.languages)
        _validate_threshold(self.threshold)
        if (self.model_id, self.model_revision, self.label) != (MODEL_ID, MODEL_REVISION, LABEL):
            raise ValueError("Only the pinned geographic NER model and label are supported")
        _validate_status(self.validation_status)

    @property
    def identity(self) -> str:
        return _digest(asdict(self))


def _validate_threshold(threshold: object) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("Threshold must be finite and between zero and one")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("Threshold must be finite and between zero and one")
    if not 0 <= threshold_value <= 1:
        raise ValueError("Threshold must be finite and between zero and one")


def _validate_status(status: str) -> None:
    if status not in ("pilot_unvalidated", "validated"):
        raise ValueError("Unknown language validation status")


def _valid_language(code: object) -> bool:
    return isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9-]{0,15}", code) is not None


def _validate_languages(languages: tuple[str, ...]) -> None:
    values = _language_values(languages)
    seen: set[str] = set()
    for code in values:
        _validate_language_code(code)
        if code in seen:
            raise ValueError("Select unique, explicit language codes")
        seen.add(code)


def _language_values(languages: object) -> tuple[str, ...]:
    if type(languages) is not tuple:
        raise ValueError("Select unique, explicit language codes")
    values = cast(tuple[str, ...], languages)
    if not values:
        raise ValueError("Select unique, explicit language codes")
    return values


def _validate_language_code(code: str) -> None:
    if not _valid_language(code):
        raise ValueError("Invalid language code")


class Extractor(Protocol):
    def predict(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Return original-string spans for each input, including empty results."""


def _routing(row: dict[str, Any], contract: Contract) -> str:
    if row["segmentation_status"] != "split":
        return "skipped_unsplit"
    if row["language"] not in contract.languages:
        return "skipped_language"
    return "ok" if row["text"].strip() else "empty_text"


def _validate_row(row: dict[str, Any]) -> None:
    _validate_row_fields(row)
    _validate_row_identity(row)
    _validate_row_project(row)


def _validate_row_fields(row: dict[str, Any]) -> None:
    for name in INPUT_COLUMNS:
        if not isinstance(row.get(name), str):
            raise ValueError("Sentence input fields must be strings")


def _validate_row_identity(row: dict[str, Any]) -> None:
    if not row["sentence_id"] or not row["document_id"]:
        raise ValueError("Sentence and document identities cannot be empty")


def _validate_row_project(row: dict[str, Any]) -> None:
    if row["project"] not in ("wikipedia", "wikivoyage"):
        raise ValueError("Unknown source project")


def _entity_mapping(entity: object) -> Mapping[str, Any]:
    if not isinstance(entity, Mapping):
        raise ValueError("Entity output is missing a required field")
    return cast(Mapping[str, Any], entity)


def _require_entity_fields(entity: Mapping[str, Any]) -> None:
    for field in ("text", "label", "start", "end", "score"):
        if field not in entity:
            raise ValueError("Entity output is missing a required field")


def _validate_entity_offsets(text: str, entity: Mapping[str, Any]) -> tuple[int, int]:
    start, end = entity["start"], entity["end"]
    if type(start) is not int or type(end) is not int:
        raise ValueError("Entity offsets must be integers")
    if not 0 <= start < end <= len(text):
        raise ValueError("Entity offsets are outside the original sentence")
    return start, end


def _validate_entity_surface(text: str, entity: Mapping[str, Any], start: int, end: int) -> None:
    if entity["label"] != LABEL:
        raise ValueError("Entity label or surface does not match the original sentence")
    if not isinstance(entity["text"], str):
        raise ValueError("Entity label or surface does not match the original sentence")
    if text[start:end] != entity["text"]:
        raise ValueError("Entity label or surface does not match the original sentence")


def _validate_entity(text: str, entity: object) -> dict[str, Any]:
    entity = _entity_mapping(entity)
    _require_entity_fields(entity)
    start, end = _validate_entity_offsets(text, entity)
    _validate_entity_surface(text, entity, start, end)
    return _entity_score(entity)


def _numeric_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Invalid entity score")
    return float(value)


def _entity_score(entity: Mapping[str, Any]) -> dict[str, Any]:
    score = _numeric_score(entity["score"])
    if not math.isfinite(score):
        raise ValueError("Invalid entity score")
    if not 0 <= score <= 1:
        raise ValueError("Invalid entity score")
    return {**{k: entity[k] for k in ("text", "label", "start", "end")}, "score": score}


def _cache_key(row: dict[str, Any], contract: Contract) -> str:
    return _digest((contract.identity, row["language"], row["text"]))


def _cached(db: sqlite3.Connection, key: str) -> list[dict[str, Any]] | None:
    found = db.execute("SELECT entities FROM predictions WHERE identity=?", (key,)).fetchone()
    if found is None:
        return None
    entities = json.loads(found[0])
    if not isinstance(entities, list):
        raise ValueError("Inference cache entry is not an entity list")
    return entities


def _missing_inputs(
    rows: list[dict[str, Any]], db: sqlite3.Connection, contract: Contract
) -> dict[str, str]:
    missing = {}
    for row in rows:
        _validate_row(row)
        key = _cache_key(row, contract)
        if _routing(row, contract) == "ok" and _cached(db, key) is None:
            missing[key] = row["text"]
    return missing


def _infer(missing: dict[str, str], db: sqlite3.Connection, extractor: Extractor) -> None:
    if not missing:
        return
    predictions = _predict_missing(missing, extractor)
    for (key, text), entities in zip(missing.items(), predictions, strict=True):
        verified = _verified_entities(text, entities)
        db.execute("INSERT INTO predictions VALUES (?, ?)", (key, _json(verified)))
    db.commit()


def _predict_missing(missing: dict[str, str], extractor: Extractor) -> list[list[dict[str, Any]]]:
    predictions = extractor.predict(list(missing.values()))
    if not isinstance(predictions, list) or len(predictions) != len(missing):
        raise ValueError("Model output count does not match input count")
    return predictions


def _verified_entities(text: str, entities: object) -> list[dict[str, Any]]:
    if not isinstance(entities, list):
        raise ValueError("Model output entities must be lists")
    return [_validate_entity(text, entity) for entity in entities]


def _output_row(row: dict[str, Any], db: sqlite3.Connection, contract: Contract) -> dict[str, Any]:
    status = _routing(row, contract)
    entities = _cached(db, _cache_key(row, contract)) if status == "ok" else []
    if entities is None:
        raise ValueError("Missing inference cache entry")
    entities = [_validate_entity(row["text"], entity) for entity in entities]
    return {
        **{k: row[k] for k in _IDENTITY_COLUMNS},
        "status": status,
        "validation_status": contract.validation_status,
        "contract_id": contract.identity,
        "entities": entities,
    }


def _receipt(source: Path, contract: Contract, batch_size: int) -> dict[str, Any]:
    return {
        "contract": json.loads(_json(asdict(contract))),
        "contract_id": contract.identity,
        "source_sha256": sha256_file(source),
        "source_rows": pq.ParquetFile(source).metadata.num_rows,
        "batch_size": batch_size,
        "processed_rows": 0,
        "status": "paused",
        "artifacts": [],
    }


def _resume(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = directory / "receipt.json"
    if not path.exists():
        return expected
    previous = json.loads(path.read_text())
    if not isinstance(previous, dict):
        raise ValueError("Checkpoint must be an object")
    for key in ("contract_id", "source_sha256", "source_rows", "batch_size", "contract"):
        if previous.get(key) != expected[key]:
            raise ValueError(f"Checkpoint identity changed: {key}")
    _verify_artifacts(directory, previous)
    return previous


def _checkpoint_count(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Checkpoint {name} must be an integer")
    if value < 0:
        raise ValueError(f"Checkpoint {name} cannot be negative")
    return value


def _checkpoint_state(receipt: dict[str, Any]) -> tuple[list[Any], int, int, int]:
    status = _checkpoint_status(receipt)
    processed_rows, source_rows, batch_size = _checkpoint_counts(receipt)
    _validate_checkpoint_completion(status, processed_rows, source_rows)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Checkpoint artifacts must be a list")
    return artifacts, processed_rows, source_rows, batch_size


def _checkpoint_status(receipt: dict[str, Any]) -> str:
    status = receipt.get("status")
    if status not in ("paused", "completed"):
        raise ValueError("Checkpoint status is invalid")
    return status


def _checkpoint_counts(receipt: dict[str, Any]) -> tuple[int, int, int]:
    processed_rows = _checkpoint_count(receipt.get("processed_rows"), "processed rows")
    source_rows = _checkpoint_count(receipt.get("source_rows"), "source rows")
    batch_size = _checkpoint_count(receipt.get("batch_size"), "batch size")
    if batch_size == 0:
        raise ValueError("Checkpoint row counts are invalid")
    if processed_rows > source_rows:
        raise ValueError("Checkpoint row counts are invalid")
    return processed_rows, source_rows, batch_size


def _validate_checkpoint_completion(status: str, processed_rows: int, source_rows: int) -> None:
    if status == "completed" and processed_rows != source_rows:
        raise ValueError("Completed checkpoint row accounting is incomplete")


def _artifact_name(artifact: object, index: int) -> str:
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint artifact is invalid")
    path_name = artifact.get("path")
    if not isinstance(path_name, str) or path_name != f"batch-{index:06d}.parquet":
        raise ValueError("Checkpoint artifact order is invalid")
    return path_name


def _artifact_rows(artifact: object) -> int:
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint artifact is invalid")
    rows = artifact.get("rows")
    if type(rows) is not int:
        raise ValueError("Checkpoint artifact row count is invalid")
    if rows <= 0:
        raise ValueError("Checkpoint artifact row count is invalid")
    return rows


def _expected_batch_rows(source_rows: int, batch_size: int, index: int) -> int:
    remaining = source_rows - index * batch_size
    if remaining <= 0:
        raise ValueError("Checkpoint contains too many artifacts")
    return min(batch_size, remaining)


def _verify_artifact_file(path: Path, artifact: object, artifact_rows: int) -> None:
    if not path.is_file():
        raise ValueError("Checkpoint artifact is missing")
    artifact_mapping = _artifact_mapping(artifact)
    _verify_artifact_hash(path, artifact_mapping)
    _verify_artifact_rows(path, artifact_rows)
    _verify_artifact_schema(path)


def _artifact_mapping(artifact: object) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint artifact is invalid")
    return cast(dict[str, Any], artifact)


def _verify_artifact_hash(path: Path, artifact: Mapping[str, Any]) -> None:
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError("Checkpoint artifact hash mismatch")


def _verify_artifact_rows(path: Path, artifact_rows: int) -> None:
    if pq.ParquetFile(path).metadata.num_rows != artifact_rows:
        raise ValueError("Checkpoint artifact row count mismatch")


def _verify_artifact_schema(path: Path) -> None:
    if pq.read_schema(path) != OUTPUT_SCHEMA:
        raise ValueError("Checkpoint artifact schema mismatch")


def _verify_artifacts(directory: Path, receipt: dict[str, Any]) -> None:
    artifacts, processed_rows, source_rows, batch_size = _checkpoint_state(receipt)
    rows = 0
    for index, artifact in enumerate(artifacts):
        path = directory / _artifact_name(artifact, index)
        artifact_rows = _artifact_rows(artifact)
        if artifact_rows != _expected_batch_rows(source_rows, batch_size, index):
            raise ValueError("Checkpoint artifact batch row count mismatch")
        _verify_artifact_file(path, artifact, artifact_rows)
        rows += artifact_rows
    if rows != processed_rows:
        raise ValueError("Checkpoint row accounting mismatch")


def _assert_source_unchanged(source: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise ValueError("Source changed during inference")


def _source_fingerprint(source: Path) -> tuple[int, int, int, int, int]:
    stat = os.stat(source)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _assert_source_fingerprint(source: Path, expected: tuple[int, int, int, int, int]) -> None:
    if _source_fingerprint(source) != expected:
        raise ValueError("Source changed during inference")


def _write_batch(
    directory: Path, index: int, rows: list[dict[str, Any]], receipt: dict[str, Any]
) -> None:
    path = directory / f"batch-{index:06d}.parquet"
    atomic_write_parquet(path, pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
    receipt["artifacts"].append({"path": path.name, "sha256": sha256_file(path), "rows": len(rows)})
    receipt["processed_rows"] += len(rows)
    atomic_write_json(directory / "receipt.json", receipt)


def _process_batch(
    source: Path,
    directory: Path,
    index: int,
    batch: Any,
    extractor: Extractor,
    contract: Contract,
    receipt: dict[str, Any],
    db: sqlite3.Connection,
    source_fingerprint: tuple[int, int, int, int, int],
) -> None:
    rows = batch.to_pylist()
    _infer(_missing_inputs(rows, db, contract), db, extractor)
    _assert_source_fingerprint(source, source_fingerprint)
    output_rows = [_output_row(row, db, contract) for row in rows]
    _assert_source_fingerprint(source, source_fingerprint)
    _write_batch(directory, index, output_rows, receipt)


def _execute(
    source: Path,
    directory: Path,
    extractor: Extractor,
    contract: Contract,
    receipt: dict[str, Any],
    db: sqlite3.Connection,
    deadline: float,
    clock: Callable[[], float],
) -> dict[str, Any]:
    completed = len(receipt["artifacts"])
    source_fingerprint = _source_fingerprint(source)
    batches = pq.ParquetFile(source).iter_batches(
        batch_size=receipt["batch_size"], columns=INPUT_COLUMNS
    )
    for index, batch in enumerate(batches):
        if index < completed:
            continue
        if clock() >= deadline:
            break
        _process_batch(
            source,
            directory,
            index,
            batch,
            extractor,
            contract,
            receipt,
            db,
            source_fingerprint,
        )
    _assert_source_unchanged(source, receipt["source_sha256"])
    receipt["status"] = (
        "completed" if receipt["processed_rows"] == receipt["source_rows"] else "paused"
    )
    atomic_write_json(directory / "receipt.json", receipt)
    return receipt


def run_shard(
    source: Path,
    output_dir: Path,
    extractor: Extractor,
    contract: Contract,
    *,
    batch_size: int = 256,
    deadline: float = math.inf,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Process or resume one immutable source shard; commit only complete batches."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    with exclusive_run_lock(output_dir / "run.lock"):
        receipt = _resume(output_dir, _receipt(source, contract, batch_size))
        atomic_write_json(output_dir / "receipt.json", receipt)
        db = sqlite3.connect(output_dir / "predictions.sqlite3")
        try:
            db.execute(
                "CREATE TABLE IF NOT EXISTS predictions (identity TEXT PRIMARY KEY, entities TEXT NOT NULL)"
            )
            return _execute(source, output_dir, extractor, contract, receipt, db, deadline, clock)
        finally:
            db.close()
