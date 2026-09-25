"""Dispatch for operator tools exposed as ``osm-polygon-wikidata-only`` subcommands.

These tools also ship as standalone (deprecated) executables. The
subcommands call the same implementations, so behaviour and exit codes
match; none of them creates or prepares the data root the way the
processing commands do.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import typer

from osm_polygon_wikidata_only.cli import audit_containment, audit_remote, enforce_integrity
from osm_polygon_wikidata_only.cli.grid5000 import run_grid5000
from osm_polygon_wikidata_only.hf import trackio_snapshot, v2_trackio_snapshot
from osm_polygon_wikidata_only.v2.config import V2_TRACKIO_SPACE_ID


def _run_enforce_integrity(args: argparse.Namespace) -> int:
    return enforce_integrity.execute(args, prog="osm-polygon-wikidata-only enforce-integrity")


def _run_audit_remote(args: argparse.Namespace) -> int:
    return _typer_status(
        lambda: audit_remote.audit(
            data_root=args.data_root, repo_id=args.repo_id, hf_token=args.hf_token
        )
    )


def _run_trackio_snapshot(args: argparse.Namespace) -> int:
    if args.dataset_version == "v2":
        space_id = args.space_id or V2_TRACKIO_SPACE_ID
        return _typer_status(
            lambda: v2_trackio_snapshot.publish(data_root=args.data_root, space_id=space_id)
        )
    space_id = args.space_id or trackio_snapshot.TRACKIO_SPACE_ID
    return _typer_status(
        lambda: trackio_snapshot.publish(data_root=args.data_root, space_id=space_id)
    )


def _typer_status(call: Callable[[], None]) -> int:
    """Run a Typer command body and translate its ``typer.Exit`` into a status."""
    try:
        call()
    except typer.Exit as exit_:
        return exit_.exit_code
    return 0


TOOL_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "enforce-integrity": _run_enforce_integrity,
    "audit-remote": _run_audit_remote,
    "trackio-snapshot": _run_trackio_snapshot,
    "grid5000": run_grid5000,
    "audit-containment": audit_containment.run,
}


def dispatch_tool(args: argparse.Namespace) -> int | None:
    """Run *args.command* if it is a tool subcommand; return ``None`` otherwise."""
    handler = TOOL_HANDLERS.get(args.command)
    return None if handler is None else handler(args)


__all__ = ["TOOL_HANDLERS", "dispatch_tool"]
