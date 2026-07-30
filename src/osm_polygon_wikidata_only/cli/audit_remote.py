"""Read-only Typer operator interface for local/remote reconciliation audits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.config.paths import resolve_data_root
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
        stems = sorted(stem for stem, missing_corpus in plan.missing if missing_corpus == corpus)
        table.add_row(
            corpus, str(len(stems)), ", ".join(f"{stem}.parquet" for stem in stems) or "—"
        )
    console.print(table)

    if plan.unexpected:
        console.print("\n[bold]Unexpected remote canonical files[/]")
        for path in sorted(plan.unexpected):
            console.print(f"  • {path}")
    if plan.repository_refresh:
        console.print("\n[bold]Missing repository-level metadata assets[/]")
        for path in sorted(plan.repository_refresh):
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
    repo_root = Path(__file__).resolve().parents[3]
    try:
        resolved_root = resolve_data_root(data_root, repo_root=repo_root)
    except Exception as error:
        console.print(f"[bold red]Error resolving data root:[/] {error}")
        raise typer.Exit(1) from None

    console.print(f"Fetching remote inventory for [bold]{repo_id}[/]…")
    try:
        inventory = RemoteInventory.fetch(repo_id=repo_id, token=hf_token)
    except Exception as error:
        console.print(f"[bold red]Failed to fetch remote inventory:[/] {error}")
        raise typer.Exit(1) from None

    local_stems = sorted(path.stem for path in resolved_root.processed_polygons.glob("*.parquet"))
    console.print(f"Local finalized regions: [bold]{len(local_stems)}[/]")
    augmentation_current = {
        stem: augmentation_is_current(resolved_root, stem)
        for stem in tqdm(
            local_stems,
            desc="Checking local augmentation",
            unit="region",
            disable=not sys.stderr.isatty(),
        )
    }

    try:
        plan = ReconciliationPlanner(
            resolved_root,
            inventory,
            stems=set(local_stems),
            augmentation_current=augmentation_current,
        ).plan()
    except Exception as error:
        console.print(f"[bold red]Failed to compute reconciliation plan:[/] {error}")
        raise typer.Exit(1) from None

    _print_plan(console, plan)


def run() -> None:
    """Installed console-script entry point."""
    app()


__all__ = ["app", "audit", "run"]
