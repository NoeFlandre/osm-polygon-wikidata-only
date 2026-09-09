"""Prepare a reproducible secondary-model geographic NER pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from osm_polygon_wikidata_only.ner.pipeline import WIKINEURAL_LANGUAGES, Contract


def build_contract() -> Contract:
    """Return the explicit, unvalidated WikiNEuRal cross-check contract."""
    from osm_polygon_wikidata_only.ner.pipeline import (
        LABEL,
        WIKINEURAL_MODEL_ID,
        WIKINEURAL_MODEL_REVISION,
    )

    return Contract(
        languages=WIKINEURAL_LANGUAGES,
        validation_status="silver_unvalidated",
        implementation_revision="geographic-ner-silver-v1",
        model_id=WIKINEURAL_MODEL_ID,
        model_revision=WIKINEURAL_MODEL_REVISION,
        label=LABEL,
    )


def prepare(pilot_dir: Path, output_dir: Path) -> Path:
    """Copy the immutable primary sample and write the secondary contract."""
    pilot_dir, output_dir = Path(pilot_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Cross-check directory already exists: {output_dir}")
    for name in ("input.parquet", "selection.json"):
        source = pilot_dir / name
        if not source.is_file():
            raise ValueError(f"Primary pilot is missing {name}: {source}")
    output_dir.mkdir(parents=True)
    shutil.copy2(pilot_dir / "input.parquet", output_dir / "input.parquet")
    shutil.copy2(pilot_dir / "selection.json", output_dir / "selection.json")
    contract = asdict(build_contract())
    contract["languages"] = list(WIKINEURAL_LANGUAGES)
    (output_dir / "contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = prepare(args.pilot_dir, args.output_dir)
    print(json.dumps({"output_dir": str(output), "status": "prepared"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
