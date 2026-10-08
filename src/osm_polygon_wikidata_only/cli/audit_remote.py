"""Read-only argparse operator interface for local/remote reconciliation audits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.config.paths import (
    DataRoot,
    DataRootError,
    repository_root,
    resolve_data_root,
)
from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlan, ReconciliationPlanner
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.uploader import UploadError

from .errors import CliFailure, report_cli_error
from .parser import AUDIT_REMOTE_DESCRIPTION, add_audit_remote_arguments

PROG = "osm-polygon-wikidata-only audit-remote"
STANDALONE_PROG = "osm-polygon-wikidata-only-audit-remote"
DEFAULT_REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"
CORPORA = (
    "polygons",
    "polygon_articles",
    "wikipedia/documents",
    "wikipedia/sections",
    "wikivoyage/documents",
    "wikivoyage/sections",
    "wikidata/facts",
)


def _console() -> Console:
    """Create the console at call time so that test capture of stdout applies."""
    return Console()


def _print_plan(console: Console, plan: ReconciliationPlan) -> None:
    """Render one deterministic, public-facing reconciliation summary."""
    table = Table(title="Missing remote canonical files")
    table.add_column("Corpus")
    table.add_column("Count", justify="right")
    table.add_column("Regions")
    for corpus in CORPORA:
        table.add_row(corpus, *_plan_row(plan, corpus))
    console.print(table)
    _print_path_group(console, plan.unexpected, "Unexpected remote canonical files")
    _print_path_group(console, plan.repository_refresh, "Missing repository-level metadata assets")


def _plan_row(plan: ReconciliationPlan, corpus: str) -> tuple[str, str]:
    stems = sorted(stem for stem, missing_corpus in plan.missing if missing_corpus == corpus)
    return str(len(stems)), ", ".join(f"{stem}.parquet" for stem in stems) or "—"


def _print_path_group(console: Console, paths: Collection[str], title: str) -> None:
    if not paths:
        return
    console.print(f"\n[bold]{title}[/]")
    for path in sorted(paths):
        console.print(f"  • {path}")


def audit(
    data_root: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    hf_token: str | None = None,
) -> int:
    """Audit remote versus local canonical dataset files and return the exit status.

    Expected operator failures are reported on stderr with status 1. Every
    other exception propagates.
    """
    console = _console()
    try:
        resolved_root = _resolve_root(data_root)
        inventory = _fetch_inventory(console, repo_id, hf_token)
        local_stems = sorted(
            path.stem for path in resolved_root.processed_polygons.glob("*.parquet")
        )
        console.print(f"Local finalized regions: [bold]{len(local_stems)}[/]")
        augmentation_current = _augmentation_state(resolved_root, local_stems)
        plan = _build_plan(resolved_root, inventory, local_stems, augmentation_current)
    except CliFailure as failure:
        return report_cli_error(PROG, failure)
    _print_plan(console, plan)
    return 0


def _failure(context: str, error: Exception) -> CliFailure:
    """Describe an expected operator failure as ``context: error``."""
    return CliFailure(f"{context}: {error}")


def _resolve_root(data_root: Path | None) -> DataRoot:
    repo_root = repository_root()
    try:
        return resolve_data_root(data_root, repo_root=repo_root)
    except (DataRootError, OSError) as error:
        raise _failure("cannot resolve data root", error) from None


def _fetch_inventory(console: Console, repo_id: str, hf_token: str | None) -> RemoteInventory:
    console.print(f"Fetching remote inventory for [bold]{repo_id}[/]…")
    try:
        return RemoteInventory.fetch(repo_id=repo_id, token=hf_token)
    except UploadError as error:
        raise _failure("failed to fetch remote inventory", error) from None


def _augmentation_state(resolved_root: DataRoot, local_stems: list[str]) -> dict[str, bool]:
    return {
        stem: augmentation_is_current(resolved_root, stem)
        for stem in tqdm(
            local_stems,
            desc="Checking local augmentation",
            unit="region",
            disable=not sys.stderr.isatty(),
        )
    }


def _build_plan(
    resolved_root: DataRoot,
    inventory: RemoteInventory,
    local_stems: list[str],
    augmentation_current: dict[str, bool],
) -> ReconciliationPlan:
    try:
        return ReconciliationPlanner(
            resolved_root,
            inventory,
            stems=set(local_stems),
            augmentation_current=augmentation_current,
        ).plan()
    except (OSError, ValueError) as error:
        raise _failure("failed to compute reconciliation plan", error) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Installed console-script entry point for ``osm-polygon-wikidata-only-audit-remote``."""
    parser = argparse.ArgumentParser(prog=STANDALONE_PROG, description=AUDIT_REMOTE_DESCRIPTION)
    add_audit_remote_arguments(parser)
    args = parser.parse_args(argv)
    return audit(data_root=args.data_root, repo_id=args.repo_id, hf_token=args.hf_token)


__all__ = ["audit", "main"]
