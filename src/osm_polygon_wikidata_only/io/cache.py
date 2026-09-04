"""Local file-system cache for HTTP responses.

Used to avoid re-fetching the same Wikidata or Wikipedia payload on
re-runs. Cache keys are mapped to deterministic file paths under the
external data root, and entries are stored as JSON.

The cache is intentionally simple:

* no LRU eviction (caller decides when to clear);
* TTL respected on read: a stale entry is treated as a miss;
* failed fetches can be cached with a shorter TTL via the
  ``failed_ttl_s`` argument to :meth:`set`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .atomic import atomic_write_json

LOGGER = logging.getLogger(__name__)
_READ_ERROR = object()


@dataclass
class CacheEntry:
    """One cache record as returned to callers."""

    key: str
    retrieved_at: str
    status: str  # "ok" or "error"
    request_url: str
    response_metadata: dict[str, Any]
    parsed_result: Any


class JsonFileCache:
    """File-backed JSON cache with TTL support.

    Files are stored at ``<root>/<key>`` (after normalizing the key to
    avoid directory traversal). The on-disk format is a JSON object
    with a ``meta`` block and a ``payload`` block.
    """

    def __init__(
        self,
        root: Path,
        *,
        default_ttl_s: int = 60 * 60 * 24 * 30,
        contract_version: str = "v1",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.default_ttl_s = default_ttl_s
        self.contract_version = contract_version

    def _path_for(self, key: str) -> Path:
        # Replace path separators in the key with safe characters.
        safe = key.replace("/", "__").replace("\\", "__")
        if len(safe.encode()) > 160:
            digest = hashlib.sha256(key.encode()).hexdigest()
            safe = f"{safe[:80]}__{digest}"
        return self.root / f"{safe}.json"

    def get(self, key: str, *, now: float | None = None) -> CacheEntry | None:
        """Return the entry for ``key`` if present and fresh.

        Returns ``None`` on miss, on stale entries, or on parse errors.
        A corrupted entry (non-UTF-8 bytes, invalid JSON, or an invalid JSON
        shape) is treated as
        a miss, logged at WARNING so the operator notices, and removed
        so subsequent runs do not re-hit the same file.
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        raw = self._read_payload(path)
        if raw is _READ_ERROR:
            return None
        metadata = self._read_metadata(path, raw)
        if metadata is None:
            return None
        raw_dict, meta, response_metadata = metadata
        return self._fresh_entry(key, raw_dict, meta, response_metadata, now=now)

    @staticmethod
    def _remove_corrupt(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    @classmethod
    def _read_payload(cls, path: Path) -> object:
        try:
            return json_loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as error:
            LOGGER.warning(
                "Cache entry %s is corrupted (non-UTF-8 bytes: %s); removing it.",
                path,
                error,
            )
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Cache entry %s could not be parsed (%s); removing it.", path, error)
        cls._remove_corrupt(path)
        return _READ_ERROR

    @classmethod
    def _read_metadata(
        cls, path: Path, raw: object
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        try:
            if not isinstance(raw, dict):
                raise ValueError("cache root must be a JSON object")
            raw_dict = cast(dict[str, Any], raw)
            meta = raw_dict.get("meta", {})
            if not isinstance(meta, dict):
                raise ValueError("cache metadata must be a JSON object")
            meta_dict = cast(dict[str, Any], meta)
            float(meta_dict.get("expires_at", 0))
            response_metadata = meta_dict.get("response_metadata", {})
            if not isinstance(response_metadata, dict):
                raise ValueError("cache response metadata must be a JSON object")
            response_metadata_dict = cast(dict[str, Any], response_metadata)
        except (TypeError, ValueError) as error:
            LOGGER.warning("Cache entry %s has an invalid shape (%s); removing it.", path, error)
            cls._remove_corrupt(path)
            return None
        return raw_dict, meta_dict, response_metadata_dict

    def _fresh_entry(
        self,
        key: str,
        raw: dict[str, Any],
        meta: dict[str, Any],
        response_metadata: dict[str, Any],
        *,
        now: float | None,
    ) -> CacheEntry | None:
        if meta.get("contract_version", "v1") != self.contract_version:
            return None
        expires_at = float(meta.get("expires_at", 0))
        current_time = time.time() if now is None else now
        if expires_at and current_time > expires_at:
            return None
        return CacheEntry(
            key=key,
            retrieved_at=meta.get("retrieved_at", ""),
            status=meta.get("status", "ok"),
            request_url=meta.get("request_url", ""),
            response_metadata=response_metadata,
            parsed_result=raw.get("payload"),
        )

    def set(
        self,
        key: str,
        payload: Any,
        *,
        request_url: str = "",
        response_metadata: dict[str, Any] | None = None,
        status: str = "ok",
        ttl_s: int | None = None,
        now: float | None = None,
    ) -> CacheEntry:
        """Store ``payload`` under ``key``.

        Returns the cache entry that was written.
        """
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        current_time = time.time() if now is None else now
        expires_at = current_time + ttl
        meta: dict[str, Any] = {
            "retrieved_at": utc_now_iso(),
            "expires_at": expires_at,
            "status": status,
            "request_url": request_url,
            "response_metadata": dict(response_metadata) if response_metadata else {},
            "contract_version": self.contract_version,
        }
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"meta": meta, "payload": payload})
        rm_value: object = meta["response_metadata"]
        rm: dict[str, Any] = rm_value if isinstance(rm_value, dict) else {}
        return CacheEntry(
            key=key,
            retrieved_at=str(meta["retrieved_at"]),
            status=status,
            request_url=request_url,
            response_metadata=rm,
            parsed_result=payload,
        )

    def delete(self, key: str) -> None:
        """Remove one cache entry if present."""
        with contextlib.suppress(FileNotFoundError):
            self._path_for(key).unlink()

    def clear(self) -> None:
        """Remove all cached entries."""
        for p in self.root.glob("*.json"):
            p.unlink()


__all__ = ["CacheEntry", "JsonFileCache"]
