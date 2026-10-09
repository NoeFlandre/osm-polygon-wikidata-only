"""Shared error reporting for operator CLI commands.

Expected operator-facing failures are reported as one line on stderr with
exit status :data:`EXIT_FAILURE`, instead of a traceback. Exit status 2 is
reserved for argparse usage errors.
"""

from __future__ import annotations

import sys

EXIT_FAILURE = 1


class CliFailure(Exception):
    """An expected operator-facing failure whose message is shown as written."""


def report_cli_error(prog: str, err: BaseException) -> int:
    """Print ``<prog>: error: <err>`` on stderr and return :data:`EXIT_FAILURE`."""
    print(f"{prog}: error: {err}", file=sys.stderr)
    return EXIT_FAILURE
