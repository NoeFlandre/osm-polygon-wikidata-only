"""Runtime settings shared across modules.

The :class:`Settings` dataclass bundles user-facing knobs. The CLI
populates it from argparse; tests build it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .paths import DEFAULT_LOCAL_DATA_ROOT

DEFAULT_REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"


# Wikimedia requires a User-Agent identifying the project and a contact.
# This is overridable via env var so deployments can set their own.
DEFAULT_USER_AGENT = (
    "osm-polygon-wikidata-only/0.1.0 (https://github.com/NoeFlandre/osm-polygon-wikidata-only) "
    "datasets-pipeline"
)


# Wikimedia API endpoints.
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
MEDIAWIKI_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"


# Default language list when ``--all-languages`` is not passed.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "fr", "de", "es", "it")

# Default cap on the number of Wikipedia articles fetched per QID.
DEFAULT_MAX_ARTICLES_PER_QID: int = 5

# Old names re-exported to keep earlier imports working.
HF_REPO_ID = DEFAULT_REPO_ID
WIKIMEDIA_USER_AGENT_DEFAULT = DEFAULT_USER_AGENT


@dataclass(frozen=True)
class Settings:
    """Pipeline-wide runtime settings.

    Frozen so callers cannot accidentally mutate global state.
    """

    repo_id: str = DEFAULT_REPO_ID
    user_agent: str = DEFAULT_USER_AGENT
    contact_email: str = ""

    # Language selection. ``None`` means "all available languages".
    languages: tuple[str, ...] | None = None

    # Full text vs lead-only.
    fetch_full_text: bool = True

    # Cap on number of articles fetched per Wikidata QID. ``None`` for no cap.
    max_articles_per_qid: int | None = None

    # Network behavior.
    request_timeout_s: float = 30.0
    # ``None`` keeps retrying classified transient network failures until
    # connectivity returns or the user interrupts the process. Tests and
    # specialized callers may set a finite positive attempt count.
    request_max_retries: int | None = None
    request_base_delay_s: float = 2.0

    # Polite Wikimedia throttling.
    wikidata_min_interval_s: float = 1.2
    wikipedia_min_interval_s: float = 0.5
    augmentation_min_interval_s: float = 0.5
    # Per-host pacing interval used for hosts that have *verified*
    # authentication. Anonymous/rejected hosts always use the
    # per-kind anonymous interval above, so this tight value never
    # applies to hosts whose bot password was rejected.
    wikimedia_authenticated_min_interval_s: float = 0.05
    rate_limit_retry_after_default_s: float = 60.0
    enrichment_batch_size: int = 50
    enrichment_site_workers: int = 8
    wikimedia_max_in_flight: int = 3
    wikimedia_requests_per_minute: float = 180.0

    # Cache.
    cache_enabled: bool = True
    cache_ttl_s: int = 60 * 60 * 24 * 30  # 30 days

    # PBF processing.
    skip_existing: bool = False
    force: bool = False
    limit: int | None = None  # debug: stop after N polygons

    # Hugging Face authentication. ``None`` means "fall back to HF_TOKEN env
    # or the saved login token". An explicit value here wins over both.
    hf_token: str | None = None

    # Recommended local data root, overridable by --data-root / env var.
    default_data_root: str = str(DEFAULT_LOCAL_DATA_ROOT)

    extra: dict[str, str] = field(default_factory=dict)
