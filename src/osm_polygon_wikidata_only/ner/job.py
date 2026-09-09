"""Run one geographic-name extraction shard inside an OAR GPU reservation."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from osm_polygon_wikidata_only.io.atomic import atomic_write_json
from osm_polygon_wikidata_only.ner.otter import OtterLocationExtractor
from osm_polygon_wikidata_only.ner.pipeline import Contract, run_shard


def gpu_identity() -> dict[str, str]:
    """Fail before downloading anything if execution is outside a GPU allocation."""
    if not os.environ.get("OAR_JOB_ID", "").isdigit():
        raise RuntimeError("Geographic NER requires an OAR reservation")
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Geographic NER requires CUDA; CPU fallback is disabled")
    return {
        "name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    return parser


def _record_execution(directory: Path, execution: dict[str, Any]) -> None:
    path = directory / "receipt.json"
    if path.is_file():
        receipt = json.loads(path.read_text())
        receipt.setdefault("executions", []).append(execution)
        atomic_write_json(path, receipt)


def _language_collection(value: object) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Contract languages must be a list or tuple")
    return tuple(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Download the pinned snapshot only on the GPU node, then resume its shard."""
    args = _parser().parse_args(argv)
    if not 0 < args.seconds <= 1020:
        raise ValueError("Job budget must be between 1 and 1020 seconds")
    started = time.monotonic()
    gpu = gpu_identity()
    raw = json.loads(args.contract.read_text())
    raw["languages"] = _language_collection(raw["languages"])
    contract = Contract(**raw)
    directory = snapshot_download(
        repo_id=contract.model_id,
        revision=contract.model_revision,
        cache_dir=args.model_cache,
        max_workers=2,
    )
    extractor = OtterLocationExtractor(
        Path(directory), batch_size=args.inference_batch_size, threshold=contract.threshold
    )
    execution = {"job_id": os.environ["OAR_JOB_ID"], "gpu": gpu, "status": "failed"}
    try:
        receipt = run_shard(
            args.source,
            args.output_dir,
            extractor,
            contract,
            batch_size=args.batch_size,
            deadline=started + args.seconds,
        )
        execution["status"] = receipt["status"]
    finally:
        execution["elapsed_seconds"] = time.monotonic() - started
        _record_execution(args.output_dir, execution)
    print(json.dumps({"status": receipt["status"], "processed_rows": receipt["processed_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
