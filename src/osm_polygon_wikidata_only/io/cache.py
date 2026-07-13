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
from typing import Any

from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .atomic import atomic_write_text

LOGGER = logging.getLogger(__name__)


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
        A corrupted entry (non-UTF-8 bytes, invalid JSON) is treated as
        a miss, logged at WARNING so the operator notices, and removed
        so subsequent runs do not re-hit the same file.
        """
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = json_loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as e:
            LOGGER.warning(
                "Cache entry %s is corrupted (non-UTF-8 bytes: %s); removing it.",
                path,
                e,
            )
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        except (OSError, json.JSONDecodeError) as e:
            LOGGER.warning("Cache entry %s could not be parsed (%s); removing it.", path, e)
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        expires_at = float(raw.get("meta", {}).get("expires_at", 0))
        if raw.get("meta", {}).get("contract_version", "v1") != self.contract_version:
            return None
        if expires_at and (now or time.time()) > expires_at:
            return None
        meta = raw.get("meta", {})
        return CacheEntry(
            key=key,
            retrieved_at=meta.get("retrieved_at", ""),
            status=meta.get("status", "ok"),
            request_url=meta.get("request_url", ""),
            response_metadata=meta.get("response_metadata", {}),
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
        expires_at = (now or time.time()) + ttl
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
        atomic_write_text(path, json_dumps({"meta": meta, "payload": payload}) + "\n")
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

    def clear(self) -> None:
        """Remove all cached entries."""
        for p in self.root.glob("*.json"):
            p.unlink()


__all__ = ["CacheEntry", "JsonFileCache"]
