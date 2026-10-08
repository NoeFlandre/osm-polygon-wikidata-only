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

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000.sentence_controller import (
    ControllerLimits,
    ControllerTarget,
    run_grid5000_sentence_controller,
)
from osm_polygon_wikidata_only.grid5000.sentence_job import run_sentence_job

from .parser import (
    CONTROLLER_DESCRIPTION,
    JOB_DESCRIPTION,
    add_grid5000_parser,
)
from .parser import (
    add_grid5000_controller_arguments as add_controller_arguments,
)
from .parser import (
    add_grid5000_job_arguments as add_job_arguments,
)


def run_controller(args: argparse.Namespace) -> int:
    """Run the controller until the ledger is complete."""
    data_root = DataRoot(args.data_root)
    data_root.ensure()
    ledger = run_grid5000_sentence_controller(
        data_root,
        target=ControllerTarget(
            site=args.site,
            queue=args.queue,
            gpu_model=args.gpu_model,
            repo_id=args.repo_id,
        ),
        limits=ControllerLimits(
            max_stems=args.max_stems,
            max_input_bytes=args.max_input_bytes,
            batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size,
            walltime=args.walltime,
        ),
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
