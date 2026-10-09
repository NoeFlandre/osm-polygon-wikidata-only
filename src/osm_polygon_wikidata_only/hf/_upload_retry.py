"""Retry policy for upload callbacks.

Only failures classified as transient are retried, after a bounded
exponential backoff with jitter. Each failed attempt that will be retried
is logged. A process-wide retry cancellation stops the wait and surfaces the
current failure, so shutdown does not sit out a backoff.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
from osm_polygon_wikidata_only.hf._uploader.operations import sanitize_server_message
from osm_polygon_wikidata_only.utils.http_retry import parse_retry_after
from osm_polygon_wikidata_only.utils.retry import (
    is_transient_network_error,
    wait_for_retry_or_cancel,
)

LOGGER = logging.getLogger(__name__)

_BASE_DELAY_SECONDS = 0.5
_MAX_DELAY_SECONDS = 8.0
_MAX_LOGGED_MESSAGE_CHARS = 200
_MAX_SERVER_RETRY_AFTER_SECONDS = 300.0
_MAX_CAUSE_DEPTH = 10


def _is_transient(error: Exception) -> bool:
    if isinstance(error, UploadError):
        return error.transient
    return is_transient_network_error(error)


def _loggable_message(error: Exception) -> str:
    """Return the error text with secrets redacted and its length capped.

    Errors raised by upload callbacks are not sanitized upstream, so a
    transient one may still carry tokens or local paths.
    """
    message = sanitize_server_message(str(error))
    if len(message) <= _MAX_LOGGED_MESSAGE_CHARS:
        return message
    return f"{message[:_MAX_LOGGED_MESSAGE_CHARS]}..."


def _backoff_seconds(attempt: int) -> float:
    capped = min(_BASE_DELAY_SECONDS * 2 ** (attempt - 1), _MAX_DELAY_SECONDS)
    return capped + random.uniform(0, _BASE_DELAY_SECONDS)


def _rate_limit_delay(error: BaseException) -> float:
    """Return the server's Retry-After wait for a 429 in the cause chain, or 0.0.

    The Hub's HTTP error sits behind the translated UploadError as its cause, and
    only the response carries the header the server asked us to honour.
    """
    current: BaseException | None = error
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None:
            break
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429:
            headers = getattr(response, "headers", None)
            value = headers.get("Retry-After") if headers is not None else None
            return parse_retry_after(value, default_s=0.0, max_s=_MAX_SERVER_RETRY_AFTER_SECONDS)
        current = current.__cause__ or current.__context__
    return 0.0


def _should_retry(error: Exception, attempt: int, attempts: int) -> bool:
    return attempt < attempts and _is_transient(error)


def _wait_before_retry(message: str, attempt: int, attempts: int, error: Exception) -> bool:
    """Log the failed attempt, back off, and return ``True`` if retries were cancelled.

    A server Retry-After longer than the local backoff sets the minimum wait.
    """
    delay = max(_backoff_seconds(attempt), _rate_limit_delay(error))
    LOGGER.warning(
        "Upload '%s' attempt %d/%d failed (%s): %s; retrying in %.1fs",
        message,
        attempt,
        attempts,
        type(error).__name__,
        _loggable_message(error),
        delay,
    )
    return wait_for_retry_or_cancel(delay)


def _validate_attempts(attempts: int) -> None:
    # A zero budget would skip the upload without raising, so the caller would
    # treat the job as delivered.
    if attempts < 1:
        raise ValueError("attempts must be >= 1")


def _run_upload_attempts[OperationInput](
    upload: Callable[[OperationInput, str], None],
    ops: OperationInput,
    message: str,
    *,
    attempts: int,
) -> None:
    """Run an upload, retrying transient failures; re-raise permanent or final failures."""
    _validate_attempts(attempts)
    for attempt in range(1, attempts + 1):
        try:
            upload(ops, message)
            return
        except Exception as error:
            if not _should_retry(error, attempt, attempts):
                raise
            if _wait_before_retry(message, attempt, attempts, error):
                raise


run_upload_attempts = _run_upload_attempts
