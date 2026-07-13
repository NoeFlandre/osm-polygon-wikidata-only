"""Tests for optional Wikimedia Bot Password authentication."""

from __future__ import annotations

import json
import logging
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from types import TracebackType

import pytest

from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaConfigurationError,
    WikimediaCredentials,
    WikimediaSession,
    load_wikimedia_credentials,
)
from osm_polygon_wikidata_only.utils.request_scheduler import AdaptiveRequestScheduler

# The session now requires per-kind interval kwargs on every read.
# Existing tests don't care about the values; pass placeholders.
_SESSION_PACE = {
    "min_interval_anonymous_s": 0.0,
    "min_interval_authenticated_s": 0.0,
}


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
        exc_tb: TracebackType | None,
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

    body, _ = session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php"), **_SESSION_PACE
    )

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

    session.read(request, **_SESSION_PACE)
    session.read(request, **_SESSION_PACE)

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

    session.read(
        urllib.request.Request("https://www.wikidata.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )
    session.read(
        urllib.request.Request("https://fr.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

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
    threads = [
        threading.Thread(target=session.read, args=(request,), kwargs=_SESSION_PACE)
        for _ in range(2)
    ]

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
def test_authentication_failure_falls_back_to_anonymous_for_that_host(
    login_result: object, caplog: pytest.LogCaptureFixture
) -> None:
    """Per-host fallback: a rejected bot password must not crash the pipeline.

    Bot passwords are bound to a single wiki. When a session contacts
    a different wiki where the password is not registered, the auth
    attempt is rejected. The session must mark that host as
    non-authenticatable and continue to serve the request anonymously
    rather than raising and aborting the whole pipeline.
    """
    caplog.set_level(logging.WARNING, logger=WikimediaSession.__module__)
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: FakeOpener(login_result=login_result),
    )

    body, _ = session.read(
        urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

    # The data request still came back (anonymously).
    assert json.loads(body) == {"query": {"ok": True}}
    # The auth failure must be surfaced as a warning, not an exception.
    assert not any(
        record.levelno >= logging.ERROR and "raw-secret-value" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.WARNING
        and "ru.wikipedia.org" in record.getMessage()
        and "anonymous" in record.getMessage().lower()
        for record in caplog.records
    )


def test_authentication_failure_does_not_retry_on_subsequent_requests() -> None:
    """After a per-host auth failure, the session must not try again."""
    opener = FakeOpener(login_result={"login": {"result": "Failed"}})
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: opener,
    )

    for _ in range(3):
        session.read(
            urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
            **_SESSION_PACE,
        )

    actions = [
        urllib.parse.parse_qs(
            item.data.decode()
            if item.data is not None
            else urllib.parse.urlparse(item.full_url).query
        ).get("action", [""])[0]
        for item in opener.requests
    ]
    # First call: token + login (rejected) + data query. Subsequent
    # calls: only the data query because the host is marked as
    # ``auth_skipped``.
    assert actions.count("login") == 1
    assert actions.count("query") == 4


def test_authentication_failure_on_one_host_does_not_block_another() -> None:
    """A rejected password on host A must not leak into host B."""
    openers: list[FakeOpener] = []

    def factory() -> FakeOpener:
        opener = FakeOpener()
        openers.append(opener)
        return opener

    # First opener (ru.wikipedia) rejects the password; second
    # (en.wikipedia) accepts it.
    openers.append(FakeOpener(login_result={"login": {"result": "Failed"}}))
    en_opener = FakeOpener()
    state = {"index": 0}

    def round_robin_factory() -> FakeOpener:
        current = state["index"]
        state["index"] += 1
        return openers[current] if current == 0 else en_opener

    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=round_robin_factory,
    )

    # Contact ru.wikipedia first; auth should be silently skipped.
    body_ru, _ = session.read(
        urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )
    assert json.loads(body_ru) == {"query": {"ok": True}}

    # Now contact en.wikipedia; the second opener should log in
    # successfully and the data request should complete.
    body_en, _ = session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )
    assert json.loads(body_en) == {"query": {"ok": True}}

    en_actions = [
        urllib.parse.parse_qs(
            item.data.decode()
            if item.data is not None
            else urllib.parse.urlparse(item.full_url).query
        ).get("action", [""])[0]
        for item in en_opener.requests
    ]
    assert en_actions == ["query", "login", "query"]


def test_authentication_fallback_warns_once_per_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=WikimediaSession.__module__)
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: FakeOpener(login_result={"login": {"result": "Failed"}}),
    )

    session.read(
        urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )
    session.read(
        urllib.request.Request("https://de.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

    fallback_records = [
        record for record in caplog.records if "continuing anonymously" in record.getMessage()
    ]
    assert [record.levelno for record in fallback_records] == [logging.WARNING, logging.DEBUG]


def test_malformed_login_response_is_sanitized_in_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bot password rejected with a malformed body must not leak the body.

    The pipeline no longer raises on per-host auth failure; instead it
    logs a warning. The warning must scrub the raw body so that any
    echo of the secret is not written to logs.
    """
    caplog.set_level(logging.WARNING, logger=WikimediaSession.__module__)
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=MalformedLoginOpener,
    )

    body, _ = session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

    # The request must still come back, just anonymously.
    assert json.loads(body) == {"query": {"ok": True}}
    # The warning must mention the host but must not echo the body.
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "en.wikipedia.org" in rendered
    assert "raw-secret-value" not in rendered


# --- Auth status snapshot -----------------------------------------------


def test_auth_snapshot_counts_authenticated_and_anonymous_hosts() -> None:
    """The snapshot distinguishes verified-authenticated hosts from anonymous ones."""
    # ru.wikipedia rejects the bot password; en.wikipedia accepts it.
    ru_opener = FakeOpener(login_result={"login": {"result": "Failed"}})
    en_opener = FakeOpener()
    state = {"index": 0}

    def factory() -> FakeOpener:
        current = state["index"]
        state["index"] += 1
        return ru_opener if current == 0 else en_opener

    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=factory,
    )
    session.read(
        urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )
    session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

    snapshot = session.auth_snapshot()

    assert snapshot.credentials_configured is True
    assert snapshot.authenticated_hosts == 1
    assert snapshot.anonymous_hosts == 1
    assert "secret-value" not in repr(snapshot)


def test_auth_snapshot_reports_all_anonymous_without_credentials() -> None:
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        opener_factory=FakeOpener,
    )
    session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        **_SESSION_PACE,
    )

    snapshot = session.auth_snapshot()

    assert snapshot.credentials_configured is False
    assert snapshot.authenticated_hosts == 0
    assert snapshot.anonymous_hosts == 1


def test_auth_snapshot_starts_unverified_before_any_request() -> None:
    """Configured credentials must not be advertised as verified before contact."""
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=FakeOpener,
    )

    snapshot = session.auth_snapshot()

    assert snapshot.credentials_configured is True
    assert snapshot.authenticated_hosts == 0
    assert snapshot.anonymous_hosts == 0
    assert snapshot.pending_hosts == 0


def test_session_uses_authenticated_pacing_only_for_verified_hosts() -> None:
    """Per-host auth-aware pacing is centralised in the session.

    The session must pace a host at ``min_interval_authenticated_s`` only
    when that host has *verified* authentication, and at
    ``min_interval_anonymous_s`` otherwise (anonymous mode, or a host
    whose bot password was rejected). All request kinds — core
    Wikipedia/Wikidata, augmentation, Wikivoyage, parse — go through
    the same ``WikimediaSession.read``, so the centralised decision is
    the single source of truth.
    """
    observed: list[tuple[str, float]] = []
    observed_lock = threading.Lock()

    class RecordingScheduler:
        def __init__(self) -> None:
            self.real = make_scheduler()

        def pace_host(self, host, *, min_interval_s):
            with observed_lock:
                observed.append((host, min_interval_s))

        def run(self, operation):
            return self.real.run(operation)

        def report_success(self) -> None:
            self.real.report_success()

    rejected_opener = FakeOpener(login_result={"login": {"result": "Failed"}})
    accepted_opener = FakeOpener()
    state = {"idx": 0}

    def factory() -> FakeOpener:
        current = state["idx"]
        state["idx"] += 1
        return rejected_opener if current == 0 else accepted_opener

    scheduler = RecordingScheduler()
    session = WikimediaSession(
        scheduler=scheduler,  # type: ignore[arg-type]
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=factory,
    )

    session.read(
        urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
        min_interval_anonymous_s=0.5,
        min_interval_authenticated_s=0.05,
    )
    session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        min_interval_anonymous_s=0.5,
        min_interval_authenticated_s=0.05,
    )

    by_host = {host: interval for host, interval in observed}
    # ru.wikipedia.org rejected authentication -> anonymous pacing (0.5).
    assert by_host["ru.wikipedia.org"] == 0.5
    # en.wikipedia.org verified authentication -> authenticated pacing (0.05).
    assert by_host["en.wikipedia.org"] == 0.05


def test_session_uses_anonymous_pacing_when_no_credentials_configured() -> None:
    """Without bot-password credentials every host is paced anonymously."""
    observed: list[tuple[str, float]] = []

    class RecordingScheduler:
        def __init__(self) -> None:
            self.real = make_scheduler()

        def pace_host(self, host, *, min_interval_s):
            observed.append((host, min_interval_s))

        def run(self, operation):
            return self.real.run(operation)

        def report_success(self) -> None:
            self.real.report_success()

    scheduler = RecordingScheduler()
    session = WikimediaSession(
        scheduler=scheduler,  # type: ignore[arg-type]
        timeout_s=5,
        user_agent="test-agent",
        opener_factory=FakeOpener,
    )

    session.read(
        urllib.request.Request("https://en.wikipedia.org/w/api.php?action=query"),
        min_interval_anonymous_s=0.5,
        min_interval_authenticated_s=0.05,
    )
    assert observed == [("en.wikipedia.org", 0.5)]


def test_auth_snapshot_does_not_classify_in_progress_auth_as_anonymous() -> None:
    """A host whose login is still in flight must not be counted as anonymous."""
    auth_started = threading.Event()
    release_auth = threading.Event()

    class SlowOpener(FakeOpener):
        def open(self, request, *, timeout):
            params = urllib.parse.parse_qs(
                request.data.decode()
                if request.data is not None
                else urllib.parse.urlparse(request.full_url).query
            )
            if params.get("action") == ["login"]:
                auth_started.set()
                release_auth.wait(timeout=5)
            return super().open(request, timeout=timeout)

    opener = SlowOpener()
    session = WikimediaSession(
        scheduler=make_scheduler(),
        timeout_s=5,
        user_agent="test-agent",
        credentials=WikimediaCredentials("NoeFlandre@pipeline", "secret-value"),
        opener_factory=lambda: opener,
    )

    def first_read() -> None:
        session.read(
            urllib.request.Request("https://ru.wikipedia.org/w/api.php?action=query"),
            **_SESSION_PACE,
        )

    thread = threading.Thread(target=first_read)
    thread.start()
    assert auth_started.wait(timeout=5)

    snapshot = session.auth_snapshot()
    assert snapshot.credentials_configured is True
    # The host is mid-authentication: it must NOT appear in either bucket.
    assert snapshot.authenticated_hosts == 0
    assert snapshot.anonymous_hosts == 0
    assert snapshot.pending_hosts == 1

    release_auth.set()
    thread.join(timeout=5)
