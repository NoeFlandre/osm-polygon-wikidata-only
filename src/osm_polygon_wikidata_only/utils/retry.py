"""Retry-with-backoff helpers used by the enrichment clients.

Kept tiny and dependency-free so tests can use it without mocking
heavy networking stacks. The actual HTTP layer is in the
:mod:`enrichment` package.
"""

from __future__ import annotations

import errno
import logging
import random
import socket
import time
import urllib.error
from collections.abc import Callable
from itertools import count
from typing import TypeVar

LOGGER = logging.getLogger(__name__)


T = TypeVar("T")

_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
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


def is_transient_network_error(error: BaseException) -> bool:
    """Return whether *error* represents a retryable network outage.

    The predicate is intentionally conservative: invalid payloads,
    authentication failures, certificate errors, and permanent HTTP
    statuses are not transient and must still reach the caller.
    """
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _TRANSIENT_HTTP_STATUS_CODES
    if isinstance(error, urllib.error.ContentTooShortError):
        return True
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        return isinstance(reason, BaseException) and is_transient_network_error(reason)
    if isinstance(error, (socket.gaierror, TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS


def transient_retry_log_callback(
    context: str,
    *,
    logger: logging.Logger = LOGGER,
) -> Callable[[int, BaseException, float], None]:
    """Return a sparse, secret-safe warning callback for unbounded retries."""

    def on_retry(attempt: int, error: BaseException, delay: float) -> None:
        if attempt != 1 and attempt % 30 != 0:
            return
        if isinstance(error, urllib.error.HTTPError):
            error_kind = f"HTTP {error.code}"
        elif isinstance(error, urllib.error.URLError) and isinstance(error.reason, BaseException):
            error_kind = type(error.reason).__name__
        else:
            error_kind = type(error).__name__
        logger.warning(
            "%s temporarily unavailable (%s); attempt %d failed; "
            "retrying in %.1fs; pipeline remains active",
            context,
            error_kind,
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
    if attempts is not None and attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: BaseException | None = None
    backoff_delay = min(base_delay, max_delay)
    attempt_numbers = count(1) if attempts is None else range(1, attempts + 1)
    for i in attempt_numbers:
        try:
            return func()
        except retry_on as e:
            if should_retry is not None and not should_retry(e):
                raise
            last_exc = e
            if attempts is not None and i == attempts:
                break
            delay = backoff_delay
            delay += random.uniform(0, base_delay)
            LOGGER.debug(
                "Retry %d/%s after %.2fs due to %s: %s",
                i,
                attempts if attempts is not None else "unbounded",
                delay,
                type(e).__name__,
                e,
            )
            if on_retry is not None:
                on_retry(i, e, delay)
            time.sleep(delay)
            backoff_delay = min(backoff_delay * 2, max_delay)
    assert last_exc is not None
    raise last_exc
