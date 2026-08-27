"""Run one CUDA-required sentence batch on a reserved Grid5000 node."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000.sentence_job import run_sentence_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the reserved-node arguments and execute one sentence batch."""
    args = _parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
