"""Tests for optional Wikimedia Bot Password authentication."""

from __future__ import annotations

import json
import threading
import traceback
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from types import TracebackType

import pytest

from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaAuthenticationError,
    WikimediaConfigurationError,
    WikimediaCredentials,
    WikimediaSession,
    load_wikimedia_credentials,
)
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode()
        self.headers = Message()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class RawResponse(FakeResponse):
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = Message()


class FakeOpener:
    def __init__(
        self,
        *,
        login_result: object = None,
        on_open: Callable[[], None] | None = None,
    ) -> None:
        self.requests: list[urllib.request.Request] = []
        self._login_result = login_result or {"login": {"result": "Success"}}
        self._on_open = on_open
        self._lock = threading.Lock()

    def open(self, request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        del timeout
        with self._lock:
            self.requests.append(request)
        if self._on_open is not None:
            self._on_open()
        parameters = urllib.parse.parse_qs(
            request.data.decode()
            if request.data is not None
            else urllib.parse.urlparse(request.full_url).query
        )
        action = parameters.get("action", [""])[0]
        if action == "query" and parameters.get("meta") == ["tokens"]:
            return FakeResponse({"query": {"tokens": {"logintoken": "LOGIN-TOKEN"}}})
        if action == "login":
            return FakeResponse(self._login_result)
        return FakeResponse({"query": {"ok": True}})


class MalformedLoginOpener(FakeOpener):
    def open(self, request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        parameters = urllib.parse.parse_qs(
            request.data.decode()
            if request.data is not None
            else urllib.parse.urlparse(request.full_url).query
        )
        if parameters.get("action") == ["login"]:
            return RawResponse(b"malformed-raw-secret-value")
        return super().open(request, timeout=timeout)


def make_scheduler() -> AdaptiveRequestScheduler:
    return AdaptiveRequestScheduler(
        requests_per_minute=100_000,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )


def test_absent_bot_password_environment_keeps_anonymous_mode() -> None:
    assert load_wikimedia_credentials({}) is None


def test_complete_bot_password_environment_loads_credentials() -> None:
    credentials = load_wikimedia_credentials(
        {
            "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret-value",
        }
    )

    assert credentials is not None
    assert credentials.username == "NoeFlandre@pipeline"
    assert credentials.password == "secret-value"
    assert "secret-value" not in repr(credentials)


@pytest.mark.parametrize(
    ("environment", "missing_name"),
    [
        ({"WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline"}, "WIKIMEDIA_BOT_PASSWORD"),
        ({"WIKIMEDIA_BOT_PASSWORD": "secret-value"}, "WIKIMEDIA_BOT_USERNAME"),
        (
            {
                "WIKIMEDIA_BOT_USERNAME": "NoeFlandre@pipeline",
                "WIKIMEDIA_BOT_PASSWORD": "   ",
            },
            "WIKIMEDIA_BOT_PASSWORD",
        ),
    ],
)
def test_partial_bot_password_environment_names_only_the_missing_variable(
    environment: dict[str, str], missing_name: str
) -> None:
    with pytest.raises(WikimediaConfigurationError) as captured:
        load_wikimedia_credentials(environment)

    message = str(captured.value)
    assert missing_name in message
    assert "secret-value" not in message


def test_anonymous_session_performs_no_login() -> None:
    openers: list[FakeOpener] = []

    def opener_factory() -> FakeOpener:
        opener = FakeOpener()
        openers.append(opener)
        return opener

    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        opener_factory=opener_factory,
    )

    body, _ = session.read(urllib.request.Request("https://en.wikipedia.org/w/api.php"))

    assert json.loads(body) == {"query": {"ok": True}}
    assert len(openers) == 1
    assert len(openers[0].requests) == 1


def test_authenticated_session_logs_in_then_reuses_host_session() -> None:
    opener = FakeOpener()
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: opener,
    )
    request = urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query")

    session.read(request)
    session.read(request)

    assert len(opener.requests) == 4
    token_request, login_request, first_query, second_query = opener.requests
    assert "meta=tokens" in token_request.full_url
    assert login_request.get_method() == "POST"
    login_parameters = urllib.parse.parse_qs(login_request.data.decode())
    assert login_parameters == {
        "action": ["login"],
        "format": ["json"],
        "formatversion": ["2"],
        "lgname": ["NoeFlandre@pipeline"],
        "lgpassword": ["secret-value"],
        "lgtoken": ["LOGIN-TOKEN"],
    }
    assert first_query.full_url == request.full_url
    assert second_query.full_url == request.full_url


def test_authenticated_session_logs_in_once_per_host() -> None:
    openers: list[FakeOpener] = []

    def opener_factory() -> FakeOpener:
        opener = FakeOpener()
        openers.append(opener)
        return opener

    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=opener_factory,
    )

    session.read(urllib.request.Request("https://www.wikidata.org/w/api.php?action=query"))
    session.read(urllib.request.Request("https://fr.wikipedia.org/w/api.php?action=query"))

    assert len(openers) == 2
    assert [len(opener.requests) for opener in openers] == [3, 3]


def test_concurrent_first_use_logs_in_once() -> None:
    login_started = threading.Event()
    release_login = threading.Event()

    def pause_first_request() -> None:
        if not login_started.is_set():
            login_started.set()
            release_login.wait(timeout=2)

    opener = FakeOpener(on_open=pause_first_request)
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: opener,
    )
    request = urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query")
    threads = [threading.Thread(target=session.read, args=(request,)) for _ in range(2)]

    for thread in threads:
        thread.start()
    assert login_started.wait(timeout=2)
    release_login.set()
    for thread in threads:
        thread.join(timeout=2)

    actions = [
        urllib.parse.parse_qs(
            item.data.decode()
            if item.data is not None
            else urllib.parse.urlparse(item.full_url).query
        ).get("action", [""])[0]
        for item in opener.requests
    ]
    assert actions.count("login") == 1
    assert len(opener.requests) == 4


@pytest.mark.parametrize(
    "login_result",
    [
        {"login": {"result": "Failed", "reason": "raw-secret-value"}},
        {"unexpected": "raw-secret-value"},
    ],
)
def test_authentication_failure_is_sanitized(login_result: object) -> None:
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: FakeOpener(login_result=login_result),
    )

    with pytest.raises(WikimediaAuthenticationError) as captured:
        session.read(urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"))

    message = str(captured.value)
    assert "en.wikipedia.org" in message
    assert "secret-value" not in message
    assert "raw-secret-value" not in message


def test_malformed_login_response_is_absent_from_traceback() -> None:
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=MalformedLoginOpener,
    )

    with pytest.raises(WikimediaAuthenticationError) as captured:
        session.read(urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"))

    rendered = "".join(
        traceback.format_exception(
            type(captured.value), captured.value, captured.value.__traceback__
        )
    )
    assert "raw-secret-value" not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
