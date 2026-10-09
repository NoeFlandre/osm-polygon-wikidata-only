"""Retry-with-backoff helpers used by the enrichment clients.

Kept tiny so tests can use it without mocking heavy networking stacks;
its only third-party import is httpx, the HTTP client the Hub and
Wikimedia layers already depend on. The actual HTTP layer is in the
:mod:`enrichment` package.
"""

from __future__ import annotations

import errno
import logging
import random
import socket
import ssl
import threading
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import TypeVar, cast

import httpx

LOGGER = logging.getLogger(__name__)


T = TypeVar("T")


class _RetryCancelled(BaseException):
    """Internal signal used to release retrying worker threads on shutdown."""


@dataclass(frozen=True, slots=True)
class _AttemptResult[T]:
    """Result of one operation attempt after retry filtering."""

    succeeded: bool
    value: T | None = None
    error: BaseException | None = None


_RETRY_CANCELLATION = threading.Event()


def _reset_retry_cancellation() -> None:
    _RETRY_CANCELLATION.clear()


def _cancel_pending_retries() -> None:
    _RETRY_CANCELLATION.set()


# Public lifecycle collaborators for the recovery and queue coordinators.
reset_retry_cancellation = _reset_retry_cancellation
cancel_pending_retries = _cancel_pending_retries


def _wait_for_retry(delay: float) -> None:
    if _RETRY_CANCELLATION.wait(delay):
        raise _RetryCancelled


def wait_for_retry_or_cancel(delay: float) -> bool:
    """Sleep up to *delay* seconds; return ``True`` if retries were cancelled.

    Unlike the internal wait used by :func:`with_retries`, this never raises,
    so callers that run on worker threads can surface their own last failure.
    """
    return _RETRY_CANCELLATION.wait(delay)


# 520-524 are Cloudflare origin failures (unknown origin error, web server down,
# origin connection timeout, origin unreachable, origin response timeout) returned
# by a fronting gateway when its origin stalls or is down. They are retried like
# the 504 gateway timeout beside them.
_TRANSIENT_HTTP_STATUS_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
)
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }
)


_MAX_CAUSE_DEPTH = 8


def is_transient_failure(error: BaseException, *, status_code: int | None = None) -> bool:
    """Return whether a failed remote call is worth retrying.

    An HTTP status, when the call produced a response, decides on its own.
    Otherwise the exception and its cause chain are checked for a network outage.
    """
    if status_code is not None:
        return status_code in _TRANSIENT_HTTP_STATUS_CODES
    return any(is_transient_network_error(cause) for cause in _cause_chain(error))


def _cause_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(chain) < _MAX_CAUSE_DEPTH:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_transient_network_error(error: BaseException) -> bool:
    """Return whether *error* represents a retryable network outage.

    The predicate is intentionally conservative: invalid payloads,
    authentication failures, certificate errors, and permanent HTTP
    statuses are not transient and must still reach the caller.
    """
    if _has_certificate_error(error):
        return False
    if isinstance(error, urllib.error.HTTPError):
        return _is_transient_http_error(error)
    if isinstance(error, urllib.error.ContentTooShortError):
        return True
    if isinstance(error, urllib.error.URLError):
        return _is_transient_url_error(error)
    return _is_transient_exception(error)


def _has_certificate_error(error: BaseException) -> bool:
    """Return whether a TLS certificate verification failure is in the cause chain.

    HTTPX reports a bad or expired certificate as a connect error, so the
    network-error classes would otherwise retry a permanent configuration failure.
    """
    return any(isinstance(cause, ssl.SSLCertVerificationError) for cause in _cause_chain(error))


def _is_transient_http_error(error: urllib.error.HTTPError) -> bool:
    return error.code in _TRANSIENT_HTTP_STATUS_CODES


def _is_transient_url_error(error: urllib.error.URLError) -> bool:
    reason = error.reason
    return isinstance(reason, BaseException) and is_transient_network_error(reason)


def _is_transient_exception(error: BaseException) -> bool:
    # httpx transport failures (timeouts, refused or dropped connections, proxy
    # errors) do not subclass the built-in timeout or connection errors, so they
    # are listed explicitly. httpx.RemoteProtocolError is raised when the server
    # drops a connection before replying. Client-side protocol errors are not
    # transient: the same request would fail the same way again.
    if isinstance(
        error,
        (
            socket.gaierror,
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ProxyError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    return isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS


def _retry_error_kind(error: BaseException) -> str:
    """Return the safe, stable label used in transient retry warnings."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}"
    if isinstance(error, urllib.error.URLError) and isinstance(error.reason, BaseException):
        return type(error.reason).__name__
    return type(error).__name__


def transient_retry_log_callback(
    context: str,
    *,
    logger: logging.Logger = LOGGER,
) -> Callable[[int, BaseException, float], None]:
    """Return a sparse, secret-safe warning callback for unbounded retries."""

    def on_retry(attempt: int, error: BaseException, delay: float) -> None:
        if attempt != 1 and attempt % 30 != 0:
            return
        logger.warning(
            "%s temporarily unavailable (%s); attempt %d failed; "
            "retrying in %.1fs; pipeline remains active",
            context,
            _retry_error_kind(error),
            attempt,
            delay,
        )

    return on_retry


def with_retries[T](
    func: Callable[[], T],
    *,
    attempts: int | None = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``func`` up to ``attempts`` times with exponential backoff + jitter.

    Parameters
    ----------
    func:
        Zero-arg callable; retried on failure.
    attempts:
        Maximum number of attempts (>= 1), or ``None`` to keep retrying.
    base_delay:
        Initial sleep in seconds before the second attempt.
    max_delay:
        Hard cap on the sleep between attempts.
    retry_on:
        Exception types that trigger a retry. Other exceptions propagate.
    should_retry:
        Optional predicate applied after ``retry_on``. Returning ``False``
        propagates the exception immediately.
    on_retry:
        Optional hook invoked between attempts as
        ``on_retry(attempt_index, exception, sleep_seconds)``.

    Notes
    -----
    Sleep = ``min(base_delay * 2 ** (i - 1), max_delay)`` plus uniform
    jitter in ``[0, base_delay)`` to avoid thundering-herd retries.
    """
    _validate_retry_arguments(attempts)
    return _run_retry_loop(
        func,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        should_retry=should_retry,
        on_retry=on_retry,
    )


def _run_retry_loop[T](
    func: Callable[[], T],
    *,
    attempts: int | None,
    base_delay: float,
    max_delay: float,
    retry_on: tuple[type[BaseException], ...],
    should_retry: Callable[[BaseException], bool] | None,
    on_retry: Callable[[int, BaseException, float], None] | None,
) -> T:
    last_exc: BaseException | None = None
    backoff_delay = min(base_delay, max_delay)
    for attempt in _attempt_numbers(attempts):
        _raise_if_cancelled()
        result = _run_attempt(func, retry_on, should_retry)
        if result.succeeded:
            return cast(T, result.value)
        last_exc = result.error
        next_delay = _next_retry_delay(
            attempt,
            result.error,
            attempts,
            backoff_delay,
            base_delay,
            max_delay,
            on_retry,
        )
        if next_delay is None:
            break
        backoff_delay = next_delay
    if last_exc is None:
        raise RuntimeError("Retry loop finished without an attempt")
    raise last_exc


def _run_attempt[T](
    func: Callable[[], T],
    retry_on: tuple[type[BaseException], ...],
    should_retry: Callable[[BaseException], bool] | None,
) -> _AttemptResult[T]:
    try:
        return _AttemptResult(True, value=func())
    except retry_on as error:
        if _should_reraise(error, should_retry):
            raise
        return _AttemptResult(False, error=error)


def _next_retry_delay(
    attempt: int,
    error: BaseException | None,
    attempts: int | None,
    backoff_delay: float,
    base_delay: float,
    max_delay: float,
    on_retry: Callable[[int, BaseException, float], None] | None,
) -> float | None:
    if _is_final_attempt(attempt, attempts):
        return None
    if error is None:
        raise RuntimeError("Retry attempt failed without an error")
    return _schedule_retry(
        attempt,
        error,
        attempts,
        backoff_delay,
        base_delay,
        max_delay,
        on_retry,
    )


def _validate_retry_arguments(attempts: int | None) -> None:
    if attempts is not None and attempts < 1:
        raise ValueError("attempts must be >= 1")


def _attempt_numbers(attempts: int | None) -> range | count:
    return count(1) if attempts is None else range(1, attempts + 1)


def _raise_if_cancelled() -> None:
    if _RETRY_CANCELLATION.is_set():
        raise _RetryCancelled


def _should_reraise(
    error: BaseException,
    should_retry: Callable[[BaseException], bool] | None,
) -> bool:
    return should_retry is not None and not should_retry(error)


def _is_final_attempt(attempt: int, attempts: int | None) -> bool:
    return attempts is not None and attempt == attempts


def _schedule_retry(
    attempt: int,
    error: BaseException,
    attempts: int | None,
    backoff_delay: float,
    base_delay: float,
    max_delay: float,
    on_retry: Callable[[int, BaseException, float], None] | None,
) -> float:
    if _is_final_attempt(attempt, attempts):
        return backoff_delay
    delay = backoff_delay + random.uniform(0, base_delay)
    LOGGER.debug(
        "Retry %d/%s after %.2fs due to %s: %s",
        attempt,
        attempts if attempts is not None else "unbounded",
        delay,
        type(error).__name__,
        error,
    )
    if on_retry is not None:
        on_retry(attempt, error, delay)
    _wait_for_retry(delay)
    return min(backoff_delay * 2, max_delay)
