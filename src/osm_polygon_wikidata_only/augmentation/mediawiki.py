"""Cached Wikimedia transport for augmentation-only reads."""

from __future__ import annotations

import gzip
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.text_cleaning import (
    clean_article_text,
    count_words,
    estimate_tokens,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaSession,
    load_wikimedia_credentials,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.rate_limit import (
    defer_host,
    retry_after_seconds,
    wait_for_host,
)
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler
from osm_polygon_wikidata_only.utils.retry import with_retries
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import Document, document_id

LOGGER = logging.getLogger(__name__)


class AugmentationWikimediaClient:
    """Read exact Wikimedia revisions and Wikidata entities with a shared scheduler."""

    def __init__(
        self, settings: Settings, cache: JsonFileCache, *, environ: Mapping[str, str] | None = None
    ) -> None:
        source = os.environ if environ is None else environ
        credentials = load_wikimedia_credentials(source)
        rate = 1_200.0 if credentials else 180.0
        effective = replace(settings, request_timeout_s=max(settings.request_timeout_s, 60.0))
        self._settings = effective
        self._scheduler = AdaptiveRequestScheduler(
            max_in_flight=8 if credentials else 3,
            requests_per_minute=rate,
            max_requests_per_minute=rate,
            minimum_requests_per_minute=min(200.0 if credentials else 60.0, rate),
        )
        self._session = WikimediaSession(
            scheduler=self._scheduler,
            timeout_s=effective.request_timeout_s,
            user_agent=effective.user_agent,
            credentials=credentials,
        )
        self._cache = cache

    def get_json(self, url: str, *, key: str) -> dict[str, Any]:
        hit = self._cache.get(key)
        if hit is not None and hit.status == "ok" and isinstance(hit.parsed_result, dict):
            return hit.parsed_result
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme != "https":
            raise ValueError(f"Only HTTPS Wikimedia URLs are allowed: {url}")
        host = parsed_url.netloc
        request = urllib.request.Request(  # noqa: S310 - HTTPS is validated above
            url,
            headers={"User-Agent": self._settings.user_agent, "Accept-Encoding": "gzip"},
        )

        def read() -> tuple[bytes, str]:
            wait_for_host(host, min_interval_s=0.05)
            try:
                return self._session.read(request)
            except urllib.error.HTTPError as error:
                if error.code in (429, 503):
                    delay = retry_after_seconds(
                        error,
                        default_s=self._settings.rate_limit_retry_after_default_s,
                    )
                    defer_host(host, delay)
                    self._scheduler.report_host_throttled(host, delay)
                    LOGGER.warning(
                        "Wikimedia throttled %s (HTTP %d); retrying after %.1fs",
                        host,
                        error.code,
                        delay,
                    )
                raise

        raw, encoding = with_retries(
            read,
            attempts=self._settings.request_max_retries,
            base_delay=self._settings.request_base_delay_s,
            retry_on=(urllib.error.URLError, TimeoutError, OSError),
        )
        if encoding == "gzip":
            raw = gzip.decompress(raw)
        parsed: object = json.loads(raw.decode())
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {url}")
        self._cache.set(key, parsed, request_url=url, status="ok")
        return parsed

    def entities(self, qids: Iterable[str], *, props: str) -> dict[str, dict[str, Any]]:
        ids = sorted(set(qids))
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), 50):
            chunk = ids[start : start + 50]
            params = urllib.parse.urlencode(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": props,
                    "format": "json",
                    "formatversion": "2",
                    "maxlag": "5",
                }
            )
            data = self.get_json(
                f"https://www.wikidata.org/w/api.php?{params}",
                key=f"entities/{props.replace('|', '-')}/{'-'.join(chunk)}.json",
            )
            raw_entities = data.get("entities", [])
            entity_values = (
                raw_entities.values() if isinstance(raw_entities, dict) else raw_entities
            )
            for entity in entity_values:
                if isinstance(entity, dict) and entity.get("id"):
                    out[str(entity["id"])] = entity
        return out

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        host = f"{language}.{'wikipedia' if project == 'wikipedia' else 'wikivoyage'}.org"
        params = urllib.parse.urlencode(
            {
                "action": "parse",
                "oldid": str(revision_id),
                "prop": "text",
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
        data = self.get_json(
            f"https://{host}/w/api.php?{params}",
            key=f"sections/{project}/{language}/{revision_id}.json",
        )
        parsed = data.get("parse", {})
        text = parsed.get("text", "") if isinstance(parsed, dict) else ""
        if isinstance(text, dict):
            text = text.get("*", "")
        return str(text)

    def wikivoyage_document(
        self, qid: str, language: str, site: str, title: str
    ) -> Document | None:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "revisions|extracts|info",
                "titles": title,
                "rvprop": "ids|timestamp",
                "explaintext": "1",
                "inprop": "url",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
                "maxlag": "5",
            }
        )
        data = self.get_json(
            f"https://{language}.wikivoyage.org/w/api.php?{params}",
            key=f"wikivoyage/{language}/{urllib.parse.quote(title, safe='')}.json",
        )
        pages = (data.get("query") or {}).get("pages") or []
        if not pages or pages[0].get("missing"):
            return None
        page = pages[0]
        revisions = page.get("revisions") or []
        if not revisions:
            return None
        revision = revisions[0]
        text = clean_article_text(str(page.get("extract", "")))
        page_id, revision_id = int(page.get("pageid", 0)), int(revision.get("revid", 0))
        retrieved = utc_now_iso()
        return Document(
            document_id(qid, "wikivoyage", language, page_id, revision_id),
            "",
            qid,
            "wikivoyage",
            language,
            site,
            str(page.get("title", title)),
            str(page.get("fullurl", "")),
            page_id,
            revision_id,
            str(revision.get("timestamp", "")),
            retrieved,
            text,
            "plain_text",
            len(text),
            count_words(text),
            estimate_tokens(text),
            "CC BY-SA 4.0",
            f'Text from Wikivoyage article "{page.get("title", title)}"; revision {revision_id}; CC BY-SA.',
            "mediawiki_action_api",
            "ok" if text else "empty_text",
            "",
            __import__("hashlib").sha256(text.encode()).hexdigest(),
        )


__all__ = ["AugmentationWikimediaClient"]
