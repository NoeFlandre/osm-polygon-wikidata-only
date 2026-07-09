"""Retry-with-backoff helpers used by the enrichment clients.

Kept tiny and dependency-free so tests can use it without mocking
heavy networking stacks. The actual HTTP layer is in the
:mod:`enrichment` package.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

LOGGER = logging.getLogger(__name__)


T = TypeVar("T")


def with_retries[T](
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``func`` up to ``attempts`` times with exponential backoff + jitter.

    Parameters
    ----------
    func:
        Zero-arg callable; retried on failure.
    attempts:
        Maximum number of attempts (>= 1).
    base_delay:
        Initial sleep in seconds before the second attempt.
    max_delay:
        Hard cap on the sleep between attempts.
    retry_on:
        Exception types that trigger a retry. Other exceptions propagate.
    on_retry:
        Optional hook invoked between attempts as
        ``on_retry(attempt_index, exception, sleep_seconds)``.

    Notes
    -----
    Sleep = ``min(base_delay * 2 ** (i - 1), max_delay)`` plus uniform
    jitter in ``[0, base_delay)`` to avoid thundering-herd retries.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return func()
        except retry_on as e:
            last_exc = e
            if i == attempts:
                break
            delay = min(base_delay * (2 ** (i - 1)), max_delay)
            delay += random.uniform(0, base_delay)
            LOGGER.debug(
                "Retry %d/%d after %.2fs due to %s: %s",
                i,
                attempts,
                delay,
                type(e).__name__,
                e,
            )
            if on_retry is not None:
                on_retry(i, e, delay)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
