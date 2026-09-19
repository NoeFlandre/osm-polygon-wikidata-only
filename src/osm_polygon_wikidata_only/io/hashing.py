"""Streaming content-hash helpers for local artifacts.

A publication hashes the same artifacts repeatedly: the language inventory
fingerprints every source file, the manifest hashes every generated file,
and the remote verification hashes them again. Each pass re-reads the
whole corpus, which on a spinning external disk dominates the run.

Digests are therefore memoised against a cheap fingerprint of the file --
its size, inode, and modification time. A file whose fingerprint is
unchanged cannot have different bytes than the digest recorded for it, and
any write changes the fingerprint, so a stale digest is never returned.

The cache lives for the process. Callers that want it to survive across
runs pass a directory to :func:`enable_hash_cache`; the index is then
written back atomically at the end of the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from osm_polygon_wikidata_only.io.atomic import atomic_write_text

LOGGER = logging.getLogger(__name__)

_SHA256_CHUNK_SIZE = 1024 * 1024
_CACHE_CONTRACT_VERSION = "sha256-v1"
_CACHE_FILENAME = "hash_cache.json"

_Fingerprint = tuple[int, int, int]
_CACHE: dict[str, tuple[_Fingerprint, str]] = {}
_CACHE_DIR: Path | None = None


def _fingerprint(path: Path) -> _Fingerprint:
    stat = path.stat()
    return (stat.st_size, stat.st_ino, stat.st_mtime_ns)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* using bounded memory."""
    key = str(path.resolve())
    try:
        fingerprint = _fingerprint(path)
    except OSError:
        return _digest(path)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    digest = _digest(path)
    _CACHE[key] = (fingerprint, digest)
    return digest


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enable_hash_cache(cache_dir: Path) -> None:
    """Load a persisted digest index and keep writing to ``cache_dir``."""
    global _CACHE_DIR
    _CACHE_DIR = cache_dir
    for key, entry in _persisted_entries(cache_dir / _CACHE_FILENAME).items():
        restored = _restored_entry(entry)
        if restored is not None:
            _CACHE[key] = restored


def _persisted_entries(index_path: Path) -> dict[str, object]:
    """Return the stored entries, or nothing when the index is unusable."""
    payload = _read_index(index_path)
    if payload.get("contract") != _CACHE_CONTRACT_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {str(key): value for key, value in entries.items()}


def _read_index(index_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _restored_entry(value: object) -> tuple[_Fingerprint, str] | None:
    """Return one validated (fingerprint, digest) pair from stored JSON."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    size, inode, mtime, digest = value
    if not isinstance(digest, str):
        return None
    fingerprint = _restored_fingerprint(size, inode, mtime)
    return None if fingerprint is None else (fingerprint, digest)


def _restored_fingerprint(size: object, inode: object, mtime: object) -> _Fingerprint | None:
    if not isinstance(size, int) or not isinstance(inode, int) or not isinstance(mtime, int):
        return None
    return (size, inode, mtime)


def flush_hash_cache() -> None:
    """Persist the digest index when a cache directory was enabled."""
    if _CACHE_DIR is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": _CACHE_CONTRACT_VERSION,
        "entries": {key: [*fingerprint, digest] for key, (fingerprint, digest) in _CACHE.items()},
    }
    try:
        atomic_write_text(_CACHE_DIR / _CACHE_FILENAME, json.dumps(payload, separators=(",", ":")))
    except OSError as error:
        LOGGER.warning("Could not persist the hash cache: %s", error)


__all__ = ["enable_hash_cache", "flush_hash_cache", "sha256_file"]
