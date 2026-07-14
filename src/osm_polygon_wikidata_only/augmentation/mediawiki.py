"""Cached Wikimedia transport for augmentation-only reads."""

from __future__ import annotations

import logging
import os
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
from osm_polygon_wikidata_only.enrichment.wikimedia import read_wikimedia_json
from osm_polygon_wikidata_only.enrichment.wikimedia.transport import (
    _NonObjectJsonError,
)
from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaSession,
    load_wikimedia_credentials,
)
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.utils.request_scheduler import (
    SYSTEMIC_ACTIVE_HOST_WINDOW_S,
    SYSTEMIC_HOST_FRACTION,
    SYSTEMIC_MINIMUM_HOSTS,
    AdaptiveRequestScheduler,
)
from osm_polygon_wikidata_only.utils.retry import with_retries
from osm_polygon_wikidata_only.utils.time import utc_now_iso

from .models import Document, document_id

LOGGER = logging.getLogger(__name__)


class AugmentationWikimediaClient:
    """Read exact Wikimedia revisions and Wikidata entities with a shared scheduler."""

    def __init__(
        self,
        settings: Settings,
        cache: JsonFileCache,
        *,
        environ: Mapping[str, str] | None = None,
        scheduler: AdaptiveRequestScheduler | None = None,
        session: WikimediaSession | None = None,
    ) -> None:
        source = os.environ if environ is None else environ
        credentials = load_wikimedia_credentials(source)
        rate = 1_200.0 if credentials else 180.0
        effective = replace(settings, request_timeout_s=max(settings.request_timeout_s, 60.0))
        self._settings = effective
        self._scheduler = scheduler or AdaptiveRequestScheduler(
            max_in_flight=3,
            requests_per_minute=rate,
            max_requests_per_minute=rate,
            minimum_requests_per_minute=min(200.0 if credentials else 60.0, rate),
            active_host_window_s=SYSTEMIC_ACTIVE_HOST_WINDOW_S,
            minimum_systemic_hosts=SYSTEMIC_MINIMUM_HOSTS,
            systemic_host_fraction=SYSTEMIC_HOST_FRACTION,
        )
        self._session = session or WikimediaSession(
            scheduler=self._scheduler,
            timeout_s=effective.request_timeout_s,
            user_agent=effective.user_agent,
            credentials=credentials,
        )
        self._cache = cache

    def get_json(self, url: str, *, key: str) -> dict[str, Any]:
        # Cache hits short-circuit BEFORE any URL validation or transport
        # invocation: even a malformed cached URL must not be re-parsed.
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

        # The helper parses Retry-After exactly once per throttled attempt
        # and reports the parsed delay to the callback. We capture that
        # delay here and reuse it in the warning below, so the scheduler
        # notification and the logged warning always agree on the same
        # value (matters for HTTP-date headers, where re-parsing after a
        # sleep would yield a different number of seconds).
        captured_delay: float | None = None

        def on_throttled(h: str, delay: float) -> None:
            nonlocal captured_delay
            captured_delay = delay
            self._scheduler.report_host_throttled(h, delay)

        def read() -> dict[str, Any]:
            try:
                return read_wikimedia_json(
                    request,
                    self._session,
                    host=host,
                    anonymous_interval_s=self._settings.augmentation_min_interval_s,
                    authenticated_interval_s=self._settings.wikimedia_authenticated_min_interval_s,
                    throttle_callback=on_throttled,
                    default_throttle_s=self._settings.rate_limit_retry_after_default_s,
                )
            except _NonObjectJsonError:
                raise ValueError(f"Expected JSON object from {url}") from None

        try:
            parsed = with_retries(
                read,
                attempts=self._settings.request_max_retries,
                base_delay=self._settings.request_base_delay_s,
                retry_on=(urllib.error.URLError, TimeoutError, OSError),
            )
        except urllib.error.HTTPError as error:
            if error.code in (429, 503):
                LOGGER.warning(
                    "Wikimedia throttled %s (HTTP %d); retrying after %.1fs",
                    host,
                    error.code,
                    captured_delay if captured_delay is not None else 0.0,
                )
            raise
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
