"""Shared error reporting for operator CLI commands.

Expected operator-facing failures are reported as one line on stderr with
exit status :data:`EXIT_FAILURE`, instead of a traceback. Exit status 2 is
reserved for argparse usage errors.
"""

from __future__ import annotations

import copy
import functools
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typer import Typer
    from typer.models import CommandInfo

EXIT_FAILURE = 1


class CliFailure(Exception):
    """An expected operator-facing failure whose message is shown as written."""


def report_cli_error(prog: str, err: BaseException) -> int:
    """Print ``<prog>: error: <err>`` on stderr and return :data:`EXIT_FAILURE`."""
    print(f"{prog}: error: {err}", file=sys.stderr)
    return EXIT_FAILURE


def report_deprecated(legacy: str, replacement: str) -> None:
    """Print a deprecation notice for *legacy* on stderr only.

    stdout and the exit status are left untouched, so JSON consumers that read
    stdout are unaffected. When stderr is closed (``sys.stderr`` is ``None``)
    the notice is skipped: ``print`` would otherwise fall back to stdout.
    """
    if sys.stderr is None:
        return
    print(f"{legacy}: warning: deprecated; use '{replacement}' instead.", file=sys.stderr)


def run_legacy_typer(app: Typer, legacy: str, replacement: str) -> None:
    """Run *app* as the legacy console script *legacy*.

    Typer parses the command line before it calls a command body, and a usage
    error or ``--help`` exits during that parse. The notice is attached to each
    command body, so only a run that gets past parsing prints it, whatever its
    exit status. *app* itself is not changed: the run uses a copy whose
    commands are wrapped.
    """
    legacy_app = copy.copy(app)
    legacy_app.registered_commands = [
        _with_notice(command, legacy, replacement) for command in app.registered_commands
    ]
    legacy_app()


def _with_notice(command: CommandInfo, legacy: str, replacement: str) -> CommandInfo:
    body = command.callback
    assert body is not None, "Typer commands registered with @app.command() have a callback"

    @functools.wraps(body)
    def run_body(*args: Any, **kwargs: Any) -> Any:
        try:
            return body(*args, **kwargs)
        finally:
            report_deprecated(legacy, replacement)

    wrapped = copy.copy(command)
    wrapped.callback = run_body
    return wrapped
