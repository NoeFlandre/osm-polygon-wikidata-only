"""Optional Bot Password authentication for Wikimedia API requests."""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, cast

from osm_polygon_wikidata_only.enrichment.wikimedia_http import PooledWikimediaTransport
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler

LOGGER = logging.getLogger(__name__)

WIKIMEDIA_BOT_USERNAME = "WIKIMEDIA_BOT_USERNAME"
WIKIMEDIA_BOT_PASSWORD = "WIKIMEDIA_BOT_PASSWORD"  # noqa: S105 - environment name
WIKIMEDIA_REQUESTS_PER_MINUTE = "WIKIMEDIA_REQUESTS_PER_MINUTE"
WIKIMEDIA_MAX_IN_FLIGHT = "WIKIMEDIA_MAX_IN_FLIGHT"


class WikimediaConfigurationError(ValueError):
    """Raised when Wikimedia environment configuration is invalid."""


class WikimediaAuthenticationError(RuntimeError):
    """Raised when Wikimedia rejects Bot Password authentication."""


@dataclass(frozen=True, repr=False, slots=True)
class WikimediaAuthSnapshot:
    """Point-in-time view of per-host authentication status.

    ``credentials_configured`` is True when bot-password environment
    variables were supplied; it does *not* mean any host has verified
    them. ``authenticated_hosts``/``anonymous_hosts``/``pending_hosts``
    count only hosts that have actually been contacted. ``pending_hosts``
    covers hosts whose first authentication attempt is either still in
    flight or has not yet started; these are deliberately excluded from
    the anonymous count so a host that *might* still verify is never
    mislabelled.
    """

    credentials_configured: bool
    authenticated_hosts: int
    anonymous_hosts: int
    pending_hosts: int = 0

    def __repr__(self) -> str:
        return (
            "WikimediaAuthSnapshot("
            f"credentials_configured={self.credentials_configured}, "
            f"authenticated_hosts={self.authenticated_hosts}, "
            f"anonymous_hosts={self.anonymous_hosts}, "
            f"pending_hosts={self.pending_hosts})"
        )


@dataclass(frozen=True, repr=False)
class WikimediaCredentials:
    """Bot Password credentials held only in process memory."""

    username: str
    password: str

    def __repr__(self) -> str:
        return f"WikimediaCredentials(username={self.username!r}, password=<redacted>)"


class _Response(Protocol):
    headers: Mapping[str, str]

    def read(self) -> bytes: ...

    def __enter__(self) -> _Response: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class _Opener(Protocol):
    def open(self, request: urllib.request.Request, *, timeout: float) -> _Response: ...


class WikimediaHttpSession(Protocol):
    """Transport boundary shared by Wikimedia API clients.

    The session is the single place that knows per-host authentication
    state, so it is also where per-host pacing is decided. Callers pass
    the anonymous and authenticated per-host minimum intervals for their
    request kind; the session picks the right one based on whether the
    specific host has verified authentication.
    """

    def read(
        self,
        request: urllib.request.Request,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]: ...


@dataclass
class _HostSession:
    opener: _Opener
    lock: threading.Lock
    authenticated: bool = False
    auth_skipped: bool = False  # True iff the bot password was rejected here


class WikimediaSession:
    """Cookie-preserving transport with optional lazy login per API host."""

    def __init__(
        self,
        *,
        scheduler: AdaptiveRequestScheduler,
        timeout_s: float,
        user_agent: str,
        credentials: WikimediaCredentials | None = None,
        opener_factory: Callable[[], _Opener] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._timeout_s = timeout_s
        self._user_agent = user_agent
        self._credentials = credentials
        self._opener_factory = opener_factory
        self._pooled_transport: PooledWikimediaTransport | None = None
        if opener_factory is None:
            self._pooled_transport = PooledWikimediaTransport(
                max_connections=getattr(scheduler, "max_in_flight", 8)
            )
        self._hosts: dict[str, _HostSession] = {}
        self._hosts_lock = threading.Lock()
        self._fallback_warning_lock = threading.Lock()
        self._fallback_warning_emitted = False

    def read(
        self,
        request: urllib.request.Request,
        *,
        min_interval_anonymous_s: float,
        min_interval_authenticated_s: float,
    ) -> tuple[bytes, str]:
        """Authenticate the request host when configured, pace it, and read.

        Centralises the per-host pacing decision: hosts that have
        *verified* authentication are paced at
        ``min_interval_authenticated_s``; hosts contacted anonymously
        (no credentials) or whose bot password was rejected are paced
        at ``min_interval_anonymous_s``. Every Wikidata, Wikipedia,
        Wikivoyage, and parse request goes through this method, so the
        decision is consistent across the whole pipeline.
        """
        parsed_url = urllib.parse.urlparse(request.full_url)
        if not parsed_url.hostname:
            raise ValueError("Wikimedia request URL must include a hostname")
        host = parsed_url.hostname
        state = self._host_session(host)
        if self._credentials is not None:
            self._ensure_authenticated(state, parsed_url.scheme, parsed_url.netloc)
        # Single source of truth for per-host pacing.
        if state.authenticated:
            min_interval = min_interval_authenticated_s
        else:
            min_interval = min_interval_anonymous_s
        self._scheduler.pace_host(host, min_interval_s=min_interval)
        return self._read(state.opener, request)

    def _host_session(self, hostname: str) -> _HostSession:
        with self._hosts_lock:
            state = self._hosts.get(hostname)
            if state is None:
                opener: _Opener
                pooled_transport = self._pooled_transport
                if pooled_transport is None:
                    factory = self._opener_factory
                    if factory is None:  # pragma: no cover - constructor invariant
                        raise RuntimeError("Wikimedia opener factory is unavailable")
                    opener = factory()
                else:
                    opener = cast(_Opener, pooled_transport.opener())
                state = _HostSession(opener=opener, lock=threading.Lock())
                self._hosts[hostname] = state
            return state

    def close(self) -> None:
        """Release persistent HTTP connections owned by this session."""
        if self._pooled_transport is not None:
            self._pooled_transport.close()

    def _ensure_authenticated(self, state: _HostSession, scheme: str, netloc: str) -> None:
        if state.authenticated or state.auth_skipped:
            return
        with state.lock:
            if state.authenticated or state.auth_skipped:
                return
            try:
                self._authenticate(state.opener, scheme, netloc)
            except WikimediaAuthenticationError as error:
                # A Wikimedia Bot Password is created against the central
                # SUL account and can authenticate on any Wikimedia wiki
                # where the account is attached, but the API ``login``
                # (and the resulting authenticated session cookies) is
                # performed per-host. A rejection therefore means this
                # specific host either does not have the bot's account
                # attached or does not accept the credentials. We mark the
                # host as "no auth possible" and continue anonymously so
                # the rest of the pipeline keeps running.
                state.auth_skipped = True
                self._log_authentication_fallback(netloc, error)
                return
            state.authenticated = True

    def _log_authentication_fallback(
        self, netloc: str, error: WikimediaAuthenticationError
    ) -> None:
        with self._fallback_warning_lock:
            first_fallback = not self._fallback_warning_emitted
            self._fallback_warning_emitted = True
        log = LOGGER.warning if first_fallback else LOGGER.debug
        log(
            "Wikimedia Bot Password rejected by %s; continuing anonymously for this host (%s)",
            netloc,
            error,
        )

    def auth_snapshot(self) -> WikimediaAuthSnapshot:
        """Count contacted hosts by their verified authentication status.

        Reads each host's state under its own lock (non-blocking) so:

        * Hosts whose login is currently in flight (lock held by the
          authenticating thread) are counted as ``pending``, never as
          anonymous, so a host that might still verify is never
          mislabelled.
        * Hosts whose authentication has already completed are counted
          as either ``authenticated`` or ``anonymous`` based on the
          verified flag.
        * Hosts contacted without credentials are counted as anonymous.
        """
        with self._hosts_lock:
            states = list(self._hosts.values())
        authenticated = anonymous = pending = 0
        for state in states:
            if not state.lock.acquire(blocking=False):
                # Another thread is currently authenticating this host;
                # do not wait and do not classify it as anonymous.
                pending += 1
                continue
            try:
                if state.authenticated:
                    authenticated += 1
                elif state.auth_skipped or self._credentials is None:
                    anonymous += 1
                else:
                    # Credentials configured but neither flag set yet
                    # (auth not started or just failed without setting
                    # auth_skipped). Either way, not safe to call it
                    # anonymous.
                    pending += 1
            finally:
                state.lock.release()
        return WikimediaAuthSnapshot(
            credentials_configured=self._credentials is not None,
            authenticated_hosts=authenticated,
            anonymous_hosts=anonymous,
            pending_hosts=pending,
        )

    def _authenticate(self, opener: _Opener, scheme: str, netloc: str) -> None:
        credentials = self._credentials
        if credentials is None:
            return
        endpoint = f"{scheme}://{netloc}/w/api.php"
        token_parameters = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "meta": "tokens",
                "type": "login",
            }
        )
        try:
            token_data = self._read_json(
                opener,
                urllib.request.Request(
                    f"{endpoint}?{token_parameters}",
                    headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                ),
            )
            query = token_data.get("query")
            if not isinstance(query, dict):
                raise ValueError("missing query")
            tokens = query.get("tokens")
            if not isinstance(tokens, dict):
                raise ValueError("missing tokens")
            token = tokens.get("logintoken")
            if not isinstance(token, str) or not token:
                raise ValueError("missing login token")
            login_body = urllib.parse.urlencode(
                {
                    "action": "login",
                    "format": "json",
                    "formatversion": "2",
                    "lgname": credentials.username,
                    "lgpassword": credentials.password,
                    "lgtoken": token,
                }
            ).encode()
            login_data = self._read_json(
                opener,
                urllib.request.Request(
                    endpoint,
                    data=login_body,
                    headers={
                        "User-Agent": self._user_agent,
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    method="POST",
                ),
            )
            login = login_data.get("login")
            if not isinstance(login, dict) or login.get("result") != "Success":
                raise ValueError("login rejected")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise WikimediaAuthenticationError(
                f"Wikimedia Bot Password authentication failed for {netloc}"
            ) from None

    def _read_json(self, opener: _Opener, request: urllib.request.Request) -> dict[str, object]:
        body, _ = self._read(opener, request)
        parsed: object = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    def _read(self, opener: _Opener, request: urllib.request.Request) -> tuple[bytes, str]:
        def operation() -> tuple[bytes, str]:
            try:
                with opener.open(request, timeout=self._timeout_s) as response:
                    return response.read(), response.headers.get("Content-Encoding", "")
            except urllib.error.HTTPError as error:
                # HTTPError is also the response object. Since ``open`` raised,
                # the context manager above was never entered and cannot close
                # its socket/file descriptor for us.
                error.close()
                raise

        result = self._scheduler.run(operation)
        self._scheduler.report_success()
        return result


def load_wikimedia_credentials(
    environ: Mapping[str, str] | None = None,
) -> WikimediaCredentials | None:
    """Load an optional all-or-nothing Bot Password pair."""
    source = os.environ if environ is None else environ
    username = source.get(WIKIMEDIA_BOT_USERNAME, "").strip()
    password = source.get(WIKIMEDIA_BOT_PASSWORD, "").strip()
    if not username and not password:
        return None
    if not username:
        raise WikimediaConfigurationError(f"Missing {WIKIMEDIA_BOT_USERNAME}")
    if not password:
        raise WikimediaConfigurationError(f"Missing {WIKIMEDIA_BOT_PASSWORD}")
    return WikimediaCredentials(username=username, password=password)


__all__ = [
    "WIKIMEDIA_BOT_PASSWORD",
    "WIKIMEDIA_BOT_USERNAME",
    "WIKIMEDIA_MAX_IN_FLIGHT",
    "WIKIMEDIA_REQUESTS_PER_MINUTE",
    "WikimediaAuthSnapshot",
    "WikimediaAuthenticationError",
    "WikimediaConfigurationError",
    "WikimediaCredentials",
    "WikimediaHttpSession",
    "WikimediaSession",
    "load_wikimedia_credentials",
]
