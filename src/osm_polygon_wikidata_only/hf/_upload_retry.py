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
from osm_polygon_wikidata_only.hf._uploader.operations import _sanitize_server_message
from osm_polygon_wikidata_only.utils.retry import (
    is_transient_network_error,
    wait_for_retry_or_cancel,
)

LOGGER = logging.getLogger(__name__)

_BASE_DELAY_SECONDS = 0.5
_MAX_DELAY_SECONDS = 8.0
_MAX_LOGGED_MESSAGE_CHARS = 200


def _is_transient(error: Exception) -> bool:
    if isinstance(error, UploadError):
        return error.transient
    return is_transient_network_error(error)


def _loggable_message(error: Exception) -> str:
    """Return the error text with secrets redacted and its length capped.

    Errors raised by upload callbacks are not sanitized upstream, so a
    transient one may still carry tokens or local paths.
    """
    message = _sanitize_server_message(str(error))
    if len(message) <= _MAX_LOGGED_MESSAGE_CHARS:
        return message
    return f"{message[:_MAX_LOGGED_MESSAGE_CHARS]}..."


def _backoff_seconds(attempt: int) -> float:
    capped = min(_BASE_DELAY_SECONDS * 2 ** (attempt - 1), _MAX_DELAY_SECONDS)
    return capped + random.uniform(0, _BASE_DELAY_SECONDS)


def _should_retry(error: Exception, attempt: int, attempts: int) -> bool:
    return attempt < attempts and _is_transient(error)


def _wait_before_retry(message: str, attempt: int, attempts: int, error: Exception) -> bool:
    """Log the failed attempt, back off, and return ``True`` if retries were cancelled."""
    delay = _backoff_seconds(attempt)
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
