"""Run and resume the local Grid5000 sentence-splitting controller."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000.sentence_controller import (
    run_grid5000_sentence_controller,
)
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--site", default="grenoble")
    parser.add_argument("--repo-id", default=V2_REPO_ID)
    parser.add_argument("--max-stems", type=int, default=4)
    parser.add_argument("--max-input-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--walltime", default="0:30")
    parser.add_argument("--run-id")
    parser.add_argument("--hf-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse controller options and run until the ledger is complete."""
    args = _parser().parse_args(argv)
    data_root = DataRoot(args.data_root)
    data_root.ensure()
    ledger = run_grid5000_sentence_controller(
        data_root,
        site=args.site,
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


if __name__ == "__main__":
    raise SystemExit(main())
