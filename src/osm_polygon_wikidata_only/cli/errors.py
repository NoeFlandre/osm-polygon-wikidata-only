"""Shared error reporting for operator CLI commands.

Expected operator-facing failures are reported as one line on stderr with
exit status :data:`EXIT_FAILURE`, instead of a traceback.
"""

from __future__ import annotations

import sys

EXIT_FAILURE = 1


def report_cli_error(prog: str, err: BaseException) -> int:
    """Print ``<prog>: error: <err>`` on stderr and return :data:`EXIT_FAILURE`."""
    print(f"{prog}: error: {err}", file=sys.stderr)
    return EXIT_FAILURE


def report_deprecated(legacy: str, replacement: str) -> None:
    """Print a deprecation notice for *legacy* on stderr only.

    stdout and the exit status are left untouched, so JSON consumers that read
    stdout are unaffected.
    """
    print(f"{legacy}: warning: deprecated; use '{replacement}' instead.", file=sys.stderr)
