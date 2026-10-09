"""Parse-then-dispatch path shared by the console entry point and ``main()``.

The helper parses the command line once, runs a tool subcommand when one was
selected, and otherwise hands the parsed namespace to the processing handler
that the caller supplies. It imports neither the command handlers nor the tool
modules at import time: ``--help`` and ``--version`` must stay free of
command-stack imports, and the handler module imports this one.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

RunHandler = Callable[[argparse.ArgumentParser, argparse.Namespace], int]


def parse_and_dispatch(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    run_handler: RunHandler,
) -> int:
    """Parse *argv* and run exactly one handler, returning its exit status."""
    args = parser.parse_args(argv)

    from .tools import dispatch_tool  # noqa: PLC0415

    tool_status = dispatch_tool(args)
    if tool_status is not None:
        return tool_status
    return run_handler(parser, args)


__all__ = ["RunHandler", "parse_and_dispatch"]
