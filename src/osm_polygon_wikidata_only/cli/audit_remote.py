"""Read-only Typer operator interface for local/remote reconciliation audits."""

from __future__ import annotations

import sys
from collections.abc import Collection
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.config.paths import DataRoot, resolve_data_root
from osm_polygon_wikidata_only.hf.reconciliation import ReconciliationPlan, ReconciliationPlanner
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory

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

app = typer.Typer(
    add_completion=False,
    help="Audit remote versus local canonical dataset files without changing either side.",
)


def _console() -> Console:
    """Create a console at call time so Typer's test capture remains effective."""
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


@app.command()
def audit(
    data_root: Annotated[
        Path | None,
        typer.Option(help="Local dataset root; defaults to OSM_POLYGON_DATA_ROOT."),
    ] = None,
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face dataset repository.")
    ] = DEFAULT_REPO_ID,
    hf_token: Annotated[
        str | None,
        typer.Option(
            help="Hugging Face token; defaults to configured credentials.", hide_input=True
        ),
    ] = None,
) -> None:
    """Audit remote versus local canonical dataset files."""
    console = _console()
    resolved_root = _resolve_root(console, data_root)
    inventory = _fetch_inventory(console, repo_id, hf_token)
    local_stems = sorted(path.stem for path in resolved_root.processed_polygons.glob("*.parquet"))
    console.print(f"Local finalized regions: [bold]{len(local_stems)}[/]")
    augmentation_current = _augmentation_state(resolved_root, local_stems)
    plan = _build_plan(console, resolved_root, inventory, local_stems, augmentation_current)
    _print_plan(console, plan)


def _resolve_root(console: Console, data_root: Path | None) -> DataRoot:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return resolve_data_root(data_root, repo_root=repo_root)
    except Exception as error:
        console.print(f"[bold red]Error resolving data root:[/] {error}")
        raise typer.Exit(1) from None


def _fetch_inventory(console: Console, repo_id: str, hf_token: str | None) -> RemoteInventory:
    console.print(f"Fetching remote inventory for [bold]{repo_id}[/]…")
    try:
        return RemoteInventory.fetch(repo_id=repo_id, token=hf_token)
    except Exception as error:
        console.print(f"[bold red]Failed to fetch remote inventory:[/] {error}")
        raise typer.Exit(1) from None


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
    console: Console,
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
    except Exception as error:
        console.print(f"[bold red]Failed to compute reconciliation plan:[/] {error}")
        raise typer.Exit(1) from None


def run() -> None:
    """Installed console-script entry point."""
    app()


__all__ = ["app", "audit", "run"]
