"""Persistent per-file summary cache for the augmentation scanner.

The README is regenerated automatically before every publication path,
but the dataset sidecars are largely stable from run to run: only the
PBF files touched by the most recent pipeline pass actually change.

The first README refresh walks every sidecar and computes per-file
summaries. Every subsequent refresh is fed the same ``AugmentationStats``
algorithm but consults the cache first: unchanged sidecars are
restored from the cache (zero Parquet table reads), sidecars that
disappeared from disk are dropped from aggregates, and sidecars that
gained a new fingerprint are rescanned once.

Cache key
---------
A per-file cache key is the pair
``(relative_path, fingerprint)`` where:

* ``relative_path`` is the canonical relative path inside ``<processed>/``
  (e.g. ``wikipedia/documents/monaco-latest.parquet``),
* ``fingerprint`` is computed by
  :func:`_file_fingerprint`. It encodes the size in bytes, the inode
  number ``st_ino``, the ctime ``st_ctime_ns``, the mtime
  ``st_mtime_ns``, and the cache contract version. Re-extracting a
  sidecar with the same inode, ctime, mtime, and size is a no-op.
  Replacing a sidecar (different bytes, different inode, different
  ctime, or different mtime) triggers a rescan.

Storage layout
--------------
The cache index lives at
``<data_root_cache>/stats_cache/index.json``. The index is a single
JSON object: the special ``"__contract_version__"`` key holds the
``CACHE_CONTRACT_VERSION`` constant, and every other key is a
``relative_path`` whose value is a JSON blob describing the cached
summary for that file. The index is rewritten on every refresh; the
keyset is rebuilt from the live filesystem so removed files drop out.

Atomicity
---------
The cache index is written via
:func:`osm_polygon_wikidata_only.io.atomic.atomic_write_text`. A crash
mid-write leaves the previous index intact; sibling ``.tmp`` files
are cleaned up by ``atomic_write_text`` itself.

Compatibility
-------------
If the on-disk ``__contract_version__`` is missing or differs from
``CACHE_CONTRACT_VERSION``, the loader treats the index as empty so
the next scanner pass rebuilds a compatible index from scratch.
Detecting this rebuild requires no extra IO: the fingerprint check
itself catches the version drift because the contract version is part
of the fingerprint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from osm_polygon_wikidata_only.io.atomic import atomic_write_text

LOGGER = logging.getLogger("osm_polygon_wikidata_only.hf.dataset_stats")

CACHE_SUBDIR = "stats_cache"
# The cache index is invalid if this version changes. Bump whenever
# the on-disk summary shape, the merge rules, or the empty-data
# classification changes.
CACHE_CONTRACT_VERSION = "v1"


def cache_dir(data_root_cache: Path) -> Path:
    """Return the cache directory used to store per-file summaries."""
    return data_root_cache / CACHE_SUBDIR


def index_path(data_root_cache: Path) -> Path:
    """Return the index file used to discover cached summaries."""
    return cache_dir(data_root_cache) / "index.json"


def _file_fingerprint(parquet_path: Path) -> str:
    """Return a stable fingerprint over multiple replacement-sensitive
    stat fields plus the cache contract version.

    Encoding :attr:`st_ino`, :attr:`st_ctime_ns`, :attr:`st_mtime_ns`,
    :attr:`st_size`, and the contract version catches:

    * Same-size same-mtime content swaps (ctime and/or inode change).
    * Same-content re-extracts that preserve mtime but bumped inode.
    * Code changes that redefine the summary shape or counting rules.
    """
    stat = parquet_path.stat()
    return (
        f"{CACHE_CONTRACT_VERSION}:{stat.st_ino}:{stat.st_ctime_ns}:"
        f"{stat.st_mtime_ns}:{stat.st_size}"
    )


def _relative_path(processed_dir: Path, parquet_path: Path) -> str:
    """Return the canonical relative path under ``<processed>/``."""
    return parquet_path.relative_to(processed_dir).as_posix()


def _index_is_compatible(decoded: Mapping[str, object]) -> bool:
    """Return ``True`` iff the decoded index carries the expected contract."""
    version = decoded.get("__contract_version__")
    return isinstance(version, str) and version == CACHE_CONTRACT_VERSION


def load_cache_index(data_root_cache: Path) -> dict[str, dict[str, object]]:
    """Load the cache index.

    Returns a dict keyed by ``relative_path`` mapping to a JSON
    blob. An empty index is returned when no cache exists, when the
    on-disk JSON cannot be parsed, when the contract version is
    missing, or when the contract version differs from
    :data:`CACHE_CONTRACT_VERSION`. In every incompatible case the
    scanner rebuilds a compatible index on the next refresh.
    """
    path = index_path(data_root_cache)
    if not path.exists():
        return {}
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        LOGGER.warning("Stats cache index unreadable; treating as empty: %s", path)
        return {}
    if not isinstance(decoded, dict):
        return {}
    if not _index_is_compatible(decoded):
        LOGGER.warning(
            "Stats cache index contract version mismatch: rebuilding from scratch: %s", path
        )
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in decoded.items():
        if key == "__contract_version__":
            continue
        if isinstance(value, dict):
            out[key] = value
    return out


def write_cache_index(data_root_cache: Path, entries: Mapping[str, Mapping[str, object]]) -> None:
    """Persist the cache index atomically.

    The index file is rewritten via
    :func:`osm_polygon_wikidata_only.io.atomic.atomic_write_text`. A
    crash mid-write preserves the previous index; sibling ``.tmp``
    files are cleaned up via the helper's ``except`` branch.
    """
    target = index_path(data_root_cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {**dict(entries), "__contract_version__": CACHE_CONTRACT_VERSION},
        sort_keys=True,
        indent=2,
    )
    atomic_write_text(target, payload + "\n")


def _scan_paths(processed_dir: Path, augment_subdirs: Iterable[str]) -> list[Path]:
    """Return every sidecar path under ``processed_dir`` sorted.

    Deterministic enumeration that the cache layer walks to detect new
    or removed files.
    """
    out: list[Path] = []
    for sub in sorted(augment_subdirs):
        directory = processed_dir / sub
        if not directory.exists():
            continue
        out.extend(sorted(directory.glob("*.parquet")))
    return out


__all__ = [
    "CACHE_CONTRACT_VERSION",
    "CACHE_SUBDIR",
    "_file_fingerprint",
    "_relative_path",
    "_scan_paths",
    "cache_dir",
    "index_path",
    "load_cache_index",
    "write_cache_index",
]
