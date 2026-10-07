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
from osm_polygon_wikidata_only.utils.retry import (
    is_transient_network_error,
    wait_for_retry_or_cancel,
)

LOGGER = logging.getLogger(__name__)

_BASE_DELAY_SECONDS = 0.5
_MAX_DELAY_SECONDS = 8.0


def _is_transient(error: Exception) -> bool:
    if isinstance(error, UploadError):
        return error.transient
    return is_transient_network_error(error)


def _backoff_seconds(attempt: int) -> float:
    capped = min(_BASE_DELAY_SECONDS * 2 ** (attempt - 1), _MAX_DELAY_SECONDS)
    return capped + random.uniform(0, _BASE_DELAY_SECONDS)


def _run_upload_attempts[OperationInput](
    upload: Callable[[OperationInput, str], None],
    ops: OperationInput,
    message: str,
    *,
    attempts: int,
) -> None:
    """Run an upload, retrying transient failures; re-raise permanent or final failures."""
    for attempt in range(1, attempts + 1):
        try:
            upload(ops, message)
            return
        except Exception as error:
            if attempt == attempts or not _is_transient(error):
                raise
            delay = _backoff_seconds(attempt)
            LOGGER.warning(
                "Upload '%s' attempt %d/%d failed (%s): %s; retrying in %.1fs",
                message,
                attempt,
                attempts,
                type(error).__name__,
                error,
                delay,
            )
            if wait_for_retry_or_cancel(delay):
                raise


run_upload_attempts = _run_upload_attempts
