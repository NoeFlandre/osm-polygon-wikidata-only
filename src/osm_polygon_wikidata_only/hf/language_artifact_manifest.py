"""Processed-manifest validation for language-split inventories."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

from osm_polygon_wikidata_only.hf.language_inventory_models import (
    DatasetContract,
    LanguageInventoryError,
)
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION

_V1_MANIFEST_FIELDS = (
    ("polygons_path", "polygons"),
    ("polygon_articles_path", "polygon_articles"),
)
_V2_MANIFEST_FIELDS = (
    ("polygons_path", "polygons"),
    ("documents_path", "wikipedia/documents"),
    ("sections_path", "wikipedia/sections"),
    ("links_path", "polygon_document_links"),
)


def _manifest_fields_for(dataset: DatasetContract) -> tuple[tuple[str, str], ...]:
    """Return the manifest paths required by one dataset contract."""
    return _V1_MANIFEST_FIELDS if dataset is DatasetContract.V1 else _V2_MANIFEST_FIELDS


def _read_manifest_payload(manifest: Path) -> dict[str, object]:
    """Read and validate the JSON object stored in a processed manifest."""
    if not manifest.is_file():
        raise LanguageInventoryError(f"Processed manifest is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LanguageInventoryError(
            f"Could not read processed manifest {manifest}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise LanguageInventoryError(f"Processed manifest must be a JSON object: {manifest}")
    return cast(dict[str, object], payload)


def load_manifest_references(
    root: Path,
    manifest: Path,
    dataset: DatasetContract,
) -> set[Path]:
    payload = _read_manifest_payload(manifest)
    entries = _manifest_entries(payload, dataset, manifest)
    if not entries:
        raise LanguageInventoryError(f"Processed manifest has no entries: {manifest}")
    return _manifest_references(root, entries, _manifest_fields_for(dataset), manifest)


def _manifest_references(
    root: Path,
    entries: dict[object, object],
    fields: tuple[tuple[str, str], ...],
    manifest: Path,
) -> set[Path]:
    referenced: set[Path] = set()
    for key, raw_entry in sorted(entries.items()):
        entry = _manifest_entry(key, raw_entry, manifest)
        manifest_key = cast(str, key)
        for field, directory in fields:
            referenced.add(_manifest_field_path(root, entry, manifest_key, field, directory))
    return referenced


def _manifest_entries(
    payload: dict[str, object],
    dataset: DatasetContract,
    manifest: Path,
) -> dict[object, object]:
    if dataset is DatasetContract.V1:
        return cast(dict[object, object], payload)
    if payload.get("contract_version") != V2_CONTRACT_VERSION:
        raise LanguageInventoryError(
            f"V2 manifest contract version mismatch in {manifest}: expected {V2_CONTRACT_VERSION!r}"
        )
    entries = payload.get("regions")
    if not isinstance(entries, dict):
        raise LanguageInventoryError(f"V2 manifest regions must be an object: {manifest}")
    return cast(dict[object, object], entries)


def _manifest_entry(
    key: object,
    raw_entry: object,
    manifest: Path,
) -> dict[str, object]:
    """Validate one manifest key/value pair and return its object value."""
    if not isinstance(key, str) or not isinstance(raw_entry, dict):
        raise LanguageInventoryError(f"Malformed manifest entry {key!r} in {manifest}")
    return cast(dict[str, object], raw_entry)


def _manifest_field_path(
    root: Path,
    entry: dict[str, object],
    key: str,
    field: str,
    directory: str,
) -> Path:
    """Validate and resolve one required artifact field from a manifest entry."""
    raw_path = entry.get(field)
    if not isinstance(raw_path, str):
        raise LanguageInventoryError(f"Manifest entry {key!r} is missing string field {field!r}")
    return _manifest_path(root, raw_path, directory, key, field)


def _safe_manifest_relative_path(raw_path: str, key: object, field: str) -> Path:
    """Validate the lexical shape of a manifest-relative Parquet path."""
    relative = Path(raw_path)
    if not _manifest_path_shape_is_safe(relative, raw_path):
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} has unsafe path {raw_path!r}"
        )
    return relative


def _manifest_path_shape_is_safe(relative: Path, raw_path: str) -> bool:
    if relative.is_absolute():
        return False
    if not raw_path:
        return False
    if relative.suffix != ".parquet":
        return False
    return all(part not in {"", ".", ".."} for part in relative.parts)


def _validate_manifest_location(
    candidate: Path,
    root: Path,
    expected: Path,
    key: object,
    field: str,
) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} escapes processed root"
        ) from error
    if candidate.parent != expected:
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} must be under "
            f"{expected.relative_to(root).as_posix()!r}"
        )


def _manifest_path(
    root: Path,
    raw_path: str,
    expected_directory: str,
    key: object,
    field: str,
) -> Path:
    relative = _safe_manifest_relative_path(raw_path, key, field)
    candidate = (root / relative).resolve()
    expected = (root / expected_directory).resolve()
    _validate_manifest_location(candidate, root, expected, key, field)
    if not candidate.is_file():
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} points to missing artifact {raw_path!r}"
        )
    return candidate


def require_manifest_references(
    paths: tuple[Path, ...],
    referenced: set[Path],
    table: str,
) -> None:
    missing = [path for path in paths if path not in referenced]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise LanguageInventoryError(f"{table} artifact is not referenced by the manifest: {names}")


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise LanguageInventoryError(f"Artifact is outside processed root: {path}") from error


def file_sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as error:
        raise LanguageInventoryError(f"Could not hash artifact {path}: {error}") from error


def artifact_fingerprint(paths: list[Path], root: Path) -> str:
    digest = sha256()
    for path in sorted(
        {path.resolve() for path in paths}, key=lambda item: relative_path(item, root)
    ):
        digest.update(relative_path(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
