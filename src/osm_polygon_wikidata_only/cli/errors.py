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
