"""Logging configuration for the pipeline.

Single entry point so the CLI and any embedded callers can produce
consistent log output without each setting up their own handler.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> None:
    """Initialize module-wide logging.

    Idempotent: repeated calls with the same level do nothing. The
    format includes a module name and a short time so log lines are
    useful in a CI runner and on a developer terminal.
    """
    global _CONFIGURED
    if isinstance(level, str):
        level = level.upper()
    root = logging.getLogger()
    if _CONFIGURED:
        return
    if root.handlers:
        # Embedded callers and test runners may already own the root
        # handlers. Keep them so log capture and application logging are
        # not silently disconnected by the CLI's first invocation.
        root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
    _CONFIGURED = True
