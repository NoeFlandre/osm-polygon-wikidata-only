"""Durable publication intent for locally migrated Wikipedia documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text

CONTRACT_VERSION = "pending-publications-v1"
FILENAME = "pending_migration_publications.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_stem(stem: str) -> str:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid pending publication stem: {stem!r}")
    return stem


def _validate_marker_stem(stem: str) -> str:
    if not isinstance(stem, str):
        raise ValueError(f"Marker stem must be a string; got {type(stem).__name__}")
    return _validate_marker_stem_shape(stem)


def _validate_marker_stem_shape(stem: str) -> str:
    """Validate path-safe marker stem content after type checking."""
    if not stem:
        raise ValueError("Marker stem must be non-empty")
    if "/" in stem or "\\" in stem:
        raise ValueError(f"Marker stem must not contain path separators: {stem!r}")
    if stem in {".", ".."}:
        raise ValueError(f"Marker stem must not be a path component: {stem!r}")
    return stem


def _validate_marker_hash(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Marker fingerprint hash must be a string; got {type(value).__name__}")
    if not _SHA256_RE.match(value):
        raise ValueError(f"Marker fingerprint hash must be 64 lowercase hex chars; got {value!r}")
    return value


def _manifest_path(data_root: DataRoot) -> Path:
    return data_root.processed_manifests / FILENAME


def _load_envelope(data_root: DataRoot) -> dict[str, Any]:
    """Read and validate the pending-publications envelope shape.

    Returns an empty envelope when the file does not exist. Raises
    on malformed shape (the marker layer must not silently accept
    corruption of the stems field).
    """
    path = _manifest_path(data_root)
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed pending publication manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError("Pending publication manifest must be a JSON object")
    version = data.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"Invalid contract version: expected {CONTRACT_VERSION!r}, got {version!r}"
        )
    return data


def _save_envelope(data_root: DataRoot, envelope: dict[str, Any]) -> None:
    """Persist the envelope atomically with deterministic key ordering."""
    path = _manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def load_pending_publications(data_root: DataRoot) -> set[str]:
    """Load pending publication stems from the durable manifest in processed manifests."""
    path = _manifest_path(data_root)
    if not path.exists():
        return set()

    data = _load_envelope(data_root)

    return _validated_pending_stems(data)


def _validated_pending_stems(data: dict[str, Any]) -> set[str]:
    """Validate and return the durable pending stem list."""
    return set(_validate_pending_stem_values(_pending_stem_values(data)))


def _pending_stem_values(data: dict[str, Any]) -> list[object]:
    """Read the pending stem list with its envelope-shape checks."""
    stems = data.get("stems")
    if stems is None:
        raise TypeError("Pending publication manifest is missing 'stems' field")
    if not isinstance(stems, list):
        raise TypeError("Pending publication manifest 'stems' field must be a list")
    return stems


def _validate_pending_stem_values(stems: list[object]) -> list[str]:
    """Validate stem types, path safety, and uniqueness."""
    validated_stems: list[str] = []
    for idx, stem in enumerate(stems):
        if not isinstance(stem, str):
            raise TypeError(f"Pending publication stem at index {idx} is not a string: {stem!r}")
        _validate_stem(stem)
        validated_stems.append(stem)
    if len(validated_stems) != len(set(validated_stems)):
        raise ValueError("Pending publication manifest contains duplicate stems")
    return validated_stems


def save_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Save pending publication stems atomically to the durable manifest.

    Merges with the existing envelope -- other fields (especially
    ``metadata_refresh``) are preserved.
    """
    path = _manifest_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = {_validate_stem(stem) for stem in stems}
    envelope = _load_envelope(data_root) if path.exists() else {}
    envelope["contract_version"] = CONTRACT_VERSION
    envelope["stems"] = sorted(validated)
    atomic_write_text(path, json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def add_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Add stems to the pending publications list."""
    if not stems:
        return
    current = load_pending_publications(data_root)
    save_pending_publications(data_root, current | stems)


def remove_pending_publications(data_root: DataRoot, stems: set[str]) -> None:
    """Remove stems from the pending publications list."""
    if not stems:
        return
    current = load_pending_publications(data_root)
    save_pending_publications(data_root, current - stems)


# ---------------------------------------------------------------------------
# Metadata-refresh marker
# ---------------------------------------------------------------------------


def _envelope_marker_payload(
    stems: list[str], fingerprint_hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "stems": sorted(stems),
        "fingerprint_hashes": {
            stem: fingerprint_hashes[stem] for stem in sorted(fingerprint_hashes)
        },
    }


def set_metadata_refresh_marker(
    data_root: DataRoot,
    stems: list[str],
    fingerprint_hashes: dict[str, str],
) -> None:
    """Persist the metadata-refresh marker.

    The marker indicates that a local regional transaction has
    committed and global metadata may now be stale. The marker is
    serialized inside the existing
    ``pending_migration_publications.json`` envelope under the
    ``metadata_refresh`` key, alongside the durable pending
    publication stems.

    Strict validation:

    * ``stems`` and ``fingerprint_hashes`` must share the same key set.
    * Every fingerprint hash must be a 64-char lowercase SHA-256.
    * Every stem must be safe (non-empty, no path separators, no
      ``.``/``..``).
    * Duplicate stems are rejected.

    The marker is persisted atomically. If the persisted payload is
    byte-for-byte identical to the existing content, the file is
    left untouched (no mtime bump).
    """
    validated_stems, validated_hashes = _validated_marker_inputs(stems, fingerprint_hashes)
    marker_payload = _envelope_marker_payload(validated_stems, validated_hashes)

    envelope = _load_envelope(data_root)
    if envelope.get("metadata_refresh") == marker_payload and _manifest_path(data_root).exists():
        # No-op rewrite: do not touch the file (preserve mtime + hash).
        return
    envelope = _with_marker_payload(envelope, marker_payload)
    _save_envelope(data_root, envelope)


def _validated_marker_inputs(
    stems: list[str],
    fingerprint_hashes: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Validate marker collections and return normalized values."""
    _validate_marker_collections(stems, fingerprint_hashes)
    validated_stems = [_validate_marker_stem(stem) for stem in stems]
    validated_hashes = {
        stem: _validate_marker_hash(fingerprint_hashes[stem]) for stem in validated_stems
    }
    return validated_stems, validated_hashes


def _validate_marker_collections(stems: list[str], fingerprint_hashes: dict[str, str]) -> None:
    """Validate marker container types and matching keys."""
    _validate_marker_types(stems, fingerprint_hashes)
    if not stems:
        raise ValueError("Marker must include at least one stem")
    if len(stems) != len(set(stems)):
        raise ValueError("Marker stems must not contain duplicates")
    if set(fingerprint_hashes.keys()) != set(stems):
        raise ValueError("Marker fingerprint_hashes keys must match the stems list exactly")


def _validate_marker_types(stems: list[str], fingerprint_hashes: dict[str, str]) -> None:
    """Validate marker collection container types."""
    if not isinstance(stems, list):
        raise ValueError(f"Marker stems must be a list; got {type(stems).__name__}")
    if not isinstance(fingerprint_hashes, dict):
        raise ValueError(
            f"Marker fingerprint_hashes must be a dict; got {type(fingerprint_hashes).__name__}"
        )


def _with_marker_payload(
    envelope: dict[str, Any], marker_payload: dict[str, Any]
) -> dict[str, Any]:
    """Add marker data while preserving a valid envelope shape."""
    envelope["metadata_refresh"] = marker_payload
    envelope.setdefault("stems", [])
    envelope.setdefault("contract_version", CONTRACT_VERSION)
    return envelope


def load_metadata_refresh_marker(data_root: DataRoot) -> dict[str, Any] | None:
    """Return the metadata-refresh marker payload, or ``None`` if absent.

    The marker is read from the same envelope as the pending
    publication stems. If the marker field is missing, ``None`` is
    returned. If the marker field is malformed, a ``ValueError`` is
    raised (the marker's strict validation is part of the contract).
    """
    if not _manifest_path(data_root).exists():
        return None
    envelope = _load_envelope(data_root)
    if "metadata_refresh" not in envelope:
        return None
    return _validated_metadata_marker(envelope["metadata_refresh"])


def _validated_metadata_marker(marker: object) -> dict[str, Any]:
    """Validate and return a metadata-refresh marker payload."""
    if not isinstance(marker, dict):
        raise TypeError("metadata_refresh field must be a JSON object")
    typed_marker = cast(dict[str, Any], marker)
    _validate_metadata_marker_keys(typed_marker)
    _validate_metadata_marker_stems(typed_marker["stems"])
    _validate_metadata_marker_hashes(typed_marker["fingerprint_hashes"], typed_marker["stems"])
    return typed_marker


def _validate_metadata_marker_stems(raw_stems: object) -> None:
    """Validate marker stem values."""
    stems = raw_stems
    if not isinstance(stems, list):
        raise TypeError("metadata_refresh.stems must be a list")
    if len(stems) != len(set(stems)):
        raise ValueError("metadata_refresh.stems must not contain duplicates")
    for stem in stems:
        if not isinstance(stem, str):
            raise TypeError("metadata_refresh.stems entries must be strings")
        _validate_marker_stem(stem)


def _validate_metadata_marker_hashes(raw_hashes: object, raw_stems: object) -> None:
    """Validate marker fingerprint hashes and their key set."""
    hashes = raw_hashes
    if not isinstance(hashes, dict):
        raise TypeError("metadata_refresh.fingerprint_hashes must be a dict")
    stems = cast(list[object], raw_stems)
    if set(hashes.keys()) != set(stems):
        raise ValueError(
            "metadata_refresh.fingerprint_hashes keys must match metadata_refresh.stems"
        )
    for _stem, sha in hashes.items():
        _validate_marker_hash(sha)


def _validate_metadata_marker_keys(marker: dict[str, Any]) -> None:
    """Validate the exact marker object shape."""
    if set(marker.keys()) != {"stems", "fingerprint_hashes"}:
        raise ValueError(
            "metadata_refresh field must have exactly 'stems' and 'fingerprint_hashes' keys"
        )


def clear_metadata_refresh_marker(data_root: DataRoot) -> None:
    """Remove only the metadata-refresh marker from the envelope.

    The pending-publication stems field is preserved untouched.
    Does nothing if no marker is present (no-op rewrite).
    """
    if not _manifest_path(data_root).exists():
        return
    envelope = _load_envelope(data_root)
    if "metadata_refresh" not in envelope:
        return
    del envelope["metadata_refresh"]
    _save_envelope(data_root, envelope)
