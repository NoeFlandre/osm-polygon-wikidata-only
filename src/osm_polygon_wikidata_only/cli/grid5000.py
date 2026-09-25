"""``osm-polygon-wikidata-only grid5000 controller|job`` sentence-splitting commands.

``controller`` runs and resumes the local Grid5000 sentence-splitting
controller; ``job`` runs one CUDA-required sentence batch on a reserved
node. ``scripts/grid5000_sentence_controller.py`` and
``scripts/grid5000_sentence_job.py`` are thin shims over the ``*_main``
functions below.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000.sentence_controller import (
    DEFAULT_GRID5000_GPU_MODEL,
    DEFAULT_GRID5000_QUEUE,
    run_grid5000_sentence_controller,
)
from osm_polygon_wikidata_only.grid5000.sentence_job import run_sentence_job
from osm_polygon_wikidata_only.grid5000.sentence_protocol import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRID5000_SITE,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_STEMS,
    DEFAULT_WALLTIME,
)
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

CONTROLLER_DESCRIPTION = "Run and resume the local Grid5000 sentence-splitting controller."
JOB_DESCRIPTION = "Run one CUDA-required sentence batch on a reserved Grid5000 node."


def add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the controller options on *parser*."""
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--site", default=DEFAULT_GRID5000_SITE)
    parser.add_argument("--queue", default=DEFAULT_GRID5000_QUEUE)
    parser.add_argument("--gpu-model", default=DEFAULT_GRID5000_GPU_MODEL)
    parser.add_argument("--repo-id", default=V2_REPO_ID)
    parser.add_argument("--max-stems", type=int, default=DEFAULT_MAX_STEMS)
    parser.add_argument("--max-input-bytes", type=int, default=DEFAULT_MAX_INPUT_BYTES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument("--run-id")
    parser.add_argument("--hf-token")


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the reserved-node job options on *parser*."""
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--receipt", type=Path, required=True)


def add_grid5000_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``grid5000 controller|job`` on the root subparsers."""
    grid = sub.add_parser("grid5000", help="Grid5000 GPU sentence-splitting controller and job")
    grid_sub = grid.add_subparsers(dest="grid5000_command", required=True)
    add_controller_arguments(
        grid_sub.add_parser(
            "controller", help=CONTROLLER_DESCRIPTION, description=CONTROLLER_DESCRIPTION
        )
    )
    add_job_arguments(grid_sub.add_parser("job", help=JOB_DESCRIPTION, description=JOB_DESCRIPTION))


def run_controller(args: argparse.Namespace) -> int:
    """Run the controller until the ledger is complete."""
    data_root = DataRoot(args.data_root)
    data_root.ensure()
    ledger = run_grid5000_sentence_controller(
        data_root,
        site=args.site,
        queue=args.queue,
        gpu_model=args.gpu_model,
        repo_id=args.repo_id,
        max_stems=args.max_stems,
        max_input_bytes=args.max_input_bytes,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        walltime=args.walltime,
        run_id=args.run_id,
        hf_token=args.hf_token,
    )
    published = sum(batch.get("state") == "published" for batch in ledger["batches"])
    print(f"Grid5000 sentence run {ledger['run_id']} complete: {published} batches published")
    return 0


def run_job(args: argparse.Namespace) -> int:
    """Execute one sentence batch on the reserved node."""
    receipt = run_sentence_job(
        DataRoot(args.data_root),
        stems=args.stems,
        model_cache=args.model_cache,
        source_commit=args.source_commit,
        job_id=args.job_id,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        receipt_path=args.receipt,
    )
    print(f"Grid5000 sentence job {receipt.job_id}: {receipt.status}")
    return 0


def run_grid5000(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``grid5000`` subcommand."""
    if args.grid5000_command == "controller":
        return run_controller(args)
    return run_job(args)


def controller_main(argv: Sequence[str] | None = None) -> int:
    """Legacy ``scripts/grid5000_sentence_controller.py`` entry point."""
    parser = argparse.ArgumentParser(description=CONTROLLER_DESCRIPTION)
    add_controller_arguments(parser)
    return run_controller(parser.parse_args(argv))


def job_main(argv: Sequence[str] | None = None) -> int:
    """Legacy ``scripts/grid5000_sentence_job.py`` entry point."""
    parser = argparse.ArgumentParser(description=JOB_DESCRIPTION)
    add_job_arguments(parser)
    return run_job(parser.parse_args(argv))


__all__ = [
    "add_grid5000_parser",
    "controller_main",
    "job_main",
    "run_controller",
    "run_grid5000",
    "run_job",
]
