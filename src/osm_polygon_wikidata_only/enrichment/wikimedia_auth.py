"""Optional Bot Password authentication for Wikimedia API requests."""

from __future__ import annotations

import http.cookiejar
import json
import os
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, cast

from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler

WIKIMEDIA_BOT_USERNAME = "WIKIMEDIA_BOT_USERNAME"
WIKIMEDIA_BOT_PASSWORD = "WIKIMEDIA_BOT_PASSWORD"  # noqa: S105 - environment name
WIKIMEDIA_REQUESTS_PER_MINUTE = "WIKIMEDIA_REQUESTS_PER_MINUTE"


class WikimediaConfigurationError(ValueError):
    """Raised when Wikimedia environment configuration is invalid."""


class WikimediaAuthenticationError(RuntimeError):
    """Raised when Wikimedia rejects Bot Password authentication."""


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
    """Transport boundary shared by Wikimedia API clients."""

    def read(self, request: urllib.request.Request) -> tuple[bytes, str]: ...


@dataclass
class _HostSession:
    opener: _Opener
    lock: threading.Lock
    authenticated: bool = False


def _cookie_opener() -> _Opener:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    return cast(_Opener, opener)


class WikimediaSession:
    """Cookie-preserving transport with optional lazy login per API host."""

    def __init__(
        self,
        *,
        scheduler: AdaptiveRequestScheduler,
        timeout_s: float,
        user_agent: str,
        credentials: WikimediaCredentials | None = None,
        opener_factory: Callable[[], _Opener] = _cookie_opener,
    ) -> None:
        self._scheduler = scheduler
        self._timeout_s = timeout_s
        self._user_agent = user_agent
        self._credentials = credentials
        self._opener_factory = opener_factory
        self._hosts: dict[str, _HostSession] = {}
        self._hosts_lock = threading.Lock()

    def read(self, request: urllib.request.Request) -> tuple[bytes, str]:
        """Authenticate the request host when configured, then read its response."""
        parsed_url = urllib.parse.urlparse(request.full_url)
        if not parsed_url.hostname:
            raise ValueError("Wikimedia request URL must include a hostname")
        state = self._host_session(parsed_url.hostname)
        if self._credentials is not None:
            self._ensure_authenticated(state, parsed_url.scheme, parsed_url.netloc)
        return self._read(state.opener, request)

    def _host_session(self, hostname: str) -> _HostSession:
        with self._hosts_lock:
            state = self._hosts.get(hostname)
            if state is None:
                state = _HostSession(opener=self._opener_factory(), lock=threading.Lock())
                self._hosts[hostname] = state
            return state

    def _ensure_authenticated(self, state: _HostSession, scheme: str, netloc: str) -> None:
        if state.authenticated:
            return
        with state.lock:
            if state.authenticated:
                return
            self._authenticate(state.opener, scheme, netloc)
            state.authenticated = True

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
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WikimediaAuthenticationError(
                f"Wikimedia Bot Password authentication failed for {netloc}"
            ) from error

    def _read_json(self, opener: _Opener, request: urllib.request.Request) -> dict[str, object]:
        body, _ = self._read(opener, request)
        parsed: object = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    def _read(self, opener: _Opener, request: urllib.request.Request) -> tuple[bytes, str]:
        def operation() -> tuple[bytes, str]:
            with opener.open(request, timeout=self._timeout_s) as response:
                return response.read(), response.headers.get("Content-Encoding", "")

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
    "WIKIMEDIA_REQUESTS_PER_MINUTE",
    "WikimediaAuthenticationError",
    "WikimediaConfigurationError",
    "WikimediaCredentials",
    "WikimediaHttpSession",
    "WikimediaSession",
    "load_wikimedia_credentials",
]
