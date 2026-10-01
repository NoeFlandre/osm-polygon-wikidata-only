"""Currency checks and validation of previously written augmentation results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.steps import CONTRACT_VERSION
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.hashing import sha256_file


def sidecar_paths(data_root: DataRoot, stem: str) -> tuple[Path, Path, Path, Path, Path]:
    root = data_root.processed
    return (
        root / "wikipedia" / "documents" / f"{stem}.parquet",
        root / "wikipedia" / "sections" / f"{stem}.parquet",
        root / "wikivoyage" / "documents" / f"{stem}.parquet",
        root / "wikivoyage" / "sections" / f"{stem}.parquet",
        root / "wikidata" / "facts" / f"{stem}.parquet",
    )


def completed_region_stems(data_root: DataRoot) -> list[str]:
    """Return sorted core stems that have both polygon and article shards."""
    return _completed_region_stems(data_root)


def _completed_region_stems(data_root: DataRoot) -> list[str]:
    """Collect the deterministic intersection of core shard stems."""
    article_paths = (
        data_root.processed_articles.glob("*.parquet"),
        (data_root.processed / "wikipedia" / "documents").glob("*.parquet"),
    )
    article_stems = _stems_from_paths(path for paths in article_paths for path in paths)
    polygon_stems = _stems_from_paths(data_root.processed_polygons.glob("*.parquet"))
    return sorted(article_stems & polygon_stems)


def _stems_from_paths(paths: Iterable[Path]) -> set[str]:
    """Return unique file stems from a path iterable."""
    return {path.stem for path in paths}


def augmentation_is_current(data_root: DataRoot, stem: str) -> bool:
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not _augmentation_inputs_exist(data_root, stem):
        return False
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get(stem, {})
    if entry.get("contract_version") != CONTRACT_VERSION:
        return False
    if not _link_artifacts_are_current(entry, data_root, stem):
        return False
    expected_hashes = entry.get("core_hashes")
    return _core_hashes_are_current(expected_hashes, data_root, stem)


def _augmentation_inputs_exist(data_root: DataRoot, stem: str) -> bool:
    """Return whether the manifest and all five sidecars are present."""
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    return manifest_path.exists() and all(path.exists() for path in sidecar_paths(data_root, stem))


def _link_artifact_file_is_current(entry: dict[str, Any], links_path: Path) -> bool:
    """Validate the canonical link file and its recorded content hash."""
    return (
        entry.get("link_schema_version") == "polygon-document-links-v1"
        and links_path.is_file()
        and entry.get("link_artifact_sha256") == sha256_file(links_path)
    )


def _processed_link_manifest_is_current(data_root: DataRoot, stem: str, links_path: Path) -> bool:
    """Validate the processed-PBF link metadata against the link file."""
    processed_manifest_path = data_root.processed_manifests / "processed_pbfs.json"
    try:
        processed_entry = json.loads(processed_manifest_path.read_text(encoding="utf-8"))[
            f"{stem}.osm.pbf"
        ]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        processed_entry.get("link_schema_version") == "polygon-document-links-v1"
        and processed_entry.get("link_count") == pq.read_metadata(links_path).num_rows
    )


def _link_artifacts_are_current(entry: dict[str, Any], data_root: DataRoot, stem: str) -> bool:
    """Validate link metadata when the manifest carries the link contract."""
    if "link_schema_version" not in entry and "link_artifact_sha256" not in entry:
        return True
    links_path = data_root.processed_links / f"{stem}.parquet"
    return _link_artifact_file_is_current(
        entry, links_path
    ) and _processed_link_manifest_is_current(data_root, stem, links_path)


def _core_hashes_are_current(value: object, data_root: DataRoot, stem: str) -> bool:
    """Validate recorded core hashes against the current files."""
    if not _is_valid_core_hashes(value, data_root, stem):
        return False
    expected_hashes = cast(dict[str, str], value)
    expected_paths = [Path(key) for key in expected_hashes]
    if not all(path.is_file() for path in expected_paths):
        return False
    return _hashes_match_files(expected_hashes, expected_paths)


def _hashes_match_files(expected_hashes: dict[str, str], paths: list[Path]) -> bool:
    """Compare recorded hashes with freshly computed file hashes."""
    current = {str(path): sha256_file(path) for path in paths}
    return bool(expected_hashes == current)


_SHA256_HEX_LENGTH = 64


def _is_valid_core_hashes(value: object, data_root: DataRoot, stem: str) -> bool:
    """Return True iff *value* is the exact two-entry hash dict we accept.

    Accepts only:

    * ``processed/polygons/<stem>.parquet`` (the polygon table), and
    * exactly one of:
      - ``processed/articles/<stem>.parquet``, or
      - ``processed/wikipedia/documents/<stem>.parquet``.

    Rejects missing entries, extra entries, duplicate keys, malformed
    paths (wrong stem, directory traversal, non-string keys), malformed
    hashes (non-string values, wrong length, non-hex), and paths outside
    the resolved ``data_root.processed`` subtree. The polygon path is
    never used to select between legacy and canonical; both layout
    variants share it.
    """
    polygon_path = data_root.processed_polygons / f"{stem}.parquet"
    legacy_path = data_root.processed_articles / f"{stem}.parquet"
    canonical_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    processed_root = data_root.processed.resolve()

    polygon_key = str(polygon_path)
    legacy_key = str(legacy_path)
    canonical_key = str(canonical_path)
    allowed_keys = {polygon_key, legacy_key, canonical_key}

    if not isinstance(value, dict):
        return False
    hash_entries = cast(dict[object, object], value)
    if not _has_expected_core_hash_keys(hash_entries, polygon_key, legacy_key, canonical_key):
        return False

    for key, hash_value in hash_entries.items():
        if not _is_valid_core_hash_entry(key, hash_value, allowed_keys, processed_root):
            return False

    return True


def _has_expected_core_hash_keys(
    value: dict[object, object], polygon_key: str, legacy_key: str, canonical_key: str
) -> bool:
    """Require exactly the polygon path and one document-source path."""
    if not value or len(value) != 2:
        return False
    if polygon_key not in value:
        return False
    return sum(key in value for key in (legacy_key, canonical_key)) == 1


def _is_valid_sha256_hash(value: object) -> bool:
    """Return whether a value is a lowercase 64-character SHA-256 hash."""
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_valid_core_hash_path(key: str, allowed_keys: set[str], processed_root: Path) -> bool:
    """Return whether a manifest path is allowed and inside ``processed_root``."""
    if key not in allowed_keys:
        return False
    try:
        resolved = Path(key).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        resolved.relative_to(processed_root)
    except ValueError:
        return False
    return True


def _is_valid_core_hash_entry(
    key: object, hash_value: object, allowed_keys: set[str], processed_root: Path
) -> bool:
    """Validate one path/hash pair from the core hash map."""
    if not isinstance(key, str):
        return False
    return _is_valid_sha256_hash(hash_value) and _is_valid_core_hash_path(
        key, allowed_keys, processed_root
    )


def read_augmentation_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read one augmentation manifest and validate its top-level shape."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Augmentation manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Augmentation manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("Augmentation manifest must be a JSON object")
    return manifest


def augmentation_manifest_entry(manifest: dict[str, Any], stem: str) -> dict[str, Any]:
    """Return one manifest entry after validating its object shape."""
    if stem not in manifest:
        raise KeyError(f"Stem {stem!r} not found in augmentation manifest")
    entry = manifest[stem]
    if not isinstance(entry, dict):
        raise TypeError(f"Manifest entry for {stem!r} must be a JSON object")
    return entry


def _validate_augmentation_counts(counts: Any, stem: str) -> dict[str, Any]:
    """Validate the required non-negative sidecar counts."""
    if not isinstance(counts, dict):
        raise TypeError(f"Manifest entry counts for {stem!r} must be a JSON object")
    expected_keys = {
        "wikipedia_documents",
        "wikipedia_sections",
        "wikivoyage_documents",
        "wikivoyage_sections",
        "wikidata_facts",
    }
    if not expected_keys.issubset(counts.keys()):
        raise ValueError(
            f"Manifest entry counts for {stem!r} is missing required fields: expected {expected_keys}, got {set(counts.keys())}"
        )
    for key in expected_keys:
        _validate_count_value(counts[key], key)
    return counts


def _validate_count_value(value: object, key: str) -> None:
    """Require one manifest count to be a non-negative integer."""
    if not isinstance(value, int):
        raise TypeError(f"Manifest count for {key!r} must be a non-negative integer")
    if value < 0:
        raise TypeError(f"Manifest count for {key!r} must be a non-negative integer")


def validate_augmentation_entry(entry: dict[str, Any], stem: str) -> dict[str, Any]:
    """Validate one augmentation manifest entry and return its counts."""
    if entry.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"Invalid contract version for {stem!r} in manifest: expected {CONTRACT_VERSION!r}, got {entry.get('contract_version')!r}"
        )
    return _validate_augmentation_counts(entry.get("counts"), stem)


def validate_sidecar_file(path: Path, expected_schema: Any) -> None:
    """Require one sidecar file to exist, be readable, and match its schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Sidecar file is missing: {path}")
    try:
        actual_schema = pq.read_schema(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Sidecar file is unreadable: {path} ({exc})") from exc
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise ValueError(f"Sidecar schema mismatch for {path}")


def validate_sidecars(paths: tuple[Path, Path, Path, Path, Path]) -> None:
    """Validate all five canonical sidecar files."""
    expected_schemas = (
        wikipedia_document_schema(),
        section_schema(),
        document_schema(),
        section_schema(),
        fact_schema(),
    )
    for path, expected_schema in zip(paths, expected_schemas, strict=True):
        validate_sidecar_file(path, expected_schema)
