"""Run and resume the serial Grid5000 geographic NER pilot."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.grid5000.ner_controller import (
    DEFAULT_GPU_MODEL,
    DEFAULT_PERIOD,
    DEFAULT_QUEUE,
    DEFAULT_SITE,
    run_grid5000_ner_controller,
)
from osm_polygon_wikidata_only.ner.pipeline import INPUT_COLUMNS, Contract

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCK = Path("requirements/geographic-ner-gpu.txt")
_SOURCE_FILES = (
    Path("__init__.py"),
    Path("io/__init__.py"),
    Path("io/atomic.py"),
    Path("io/hashing.py"),
    Path("io/run_lock.py"),
    Path("utils/__init__.py"),
    Path("utils/json.py"),
)


def prepare_geographic_ner_staging(
    staging_dir: Path,
    *,
    source: Path,
    contract: Path,
    source_root: Path | None = None,
    requirements_lock: Path | None = None,
) -> Path:
    """Build one complete, immutable-ready worker tree without remote effects."""
    staging_dir = Path(staging_dir)
    root = Path(source_root or _REPO_ROOT).resolve()
    source = Path(source)
    contract = Path(contract)
    lock = Path(requirements_lock or root / _DEFAULT_LOCK)
    normalized_contract = _read_contract(contract)
    _validate_input(source)
    _validate_hashed_lock(lock)
    package = root / "src" / "osm_polygon_wikidata_only"
    _validate_source_tree(package)
    if staging_dir.exists():
        raise ValueError(f"Staging directory already exists: {staging_dir}")
    staging_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{staging_dir.name}-", dir=staging_dir.parent) as raw:
        temporary = Path(raw)
        _copy_file(source, temporary / "input.parquet")
        _write_contract(temporary / "contract.json", normalized_contract)
        _copy_file(lock, temporary / "code" / _DEFAULT_LOCK)
        _copy_source_code(package, temporary / "code" / "src" / "osm_polygon_wikidata_only")
        os.replace(temporary, staging_dir)
    return staging_dir


def _read_contract(path: Path) -> Contract:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("languages"), list):
            raise ValueError("contract languages must be a JSON list")
        raw["languages"] = tuple(raw["languages"])
        return Contract(**raw)
    except (OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid geographic NER contract: {path}") from error


def _validate_input(path: Path) -> None:
    names, rows = _input_metadata(path)
    missing = sorted(set(INPUT_COLUMNS) - names)
    if missing:
        raise ValueError(f"Geographic NER input is missing columns: {missing}")
    if rows < 1:
        raise ValueError("Geographic NER input must contain at least one row")


def _input_metadata(path: Path) -> tuple[set[str], int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Geographic NER input is missing or unsafe: {path}")
    try:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        rows = parquet.metadata.num_rows if parquet.metadata is not None else 0
    except (OSError, ValueError, pa.ArrowException) as error:
        raise ValueError(f"Invalid geographic NER input: {path}") from error
    return names, rows


def _validate_hashed_lock(path: Path) -> None:
    blocks = _lock_blocks(path)
    if not blocks or any(
        re.search(r"--hash=sha256:[0-9a-fA-F]{64}(?=\s|$)", block) is None for block in blocks
    ):
        raise ValueError("GPU requirements lock must hash every pinned package")


def _lock_blocks(path: Path) -> tuple[str, ...]:
    return _split_lock_blocks(_read_lock(path))


def _read_lock(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"GPU requirements lock is missing or unsafe: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"GPU requirements lock is unreadable: {path}") from error
    return text


def _split_lock_blocks(text: str) -> tuple[str, ...]:
    return tuple(
        block for block in re.split(r"(?m)^(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)", text) if "==" in block
    )


def _validate_source_tree(package: Path) -> None:
    if not package.is_dir() or _missing_source_files(package):
        raise ValueError("Source root is missing the geographic NER runtime")


def _missing_source_files(package: Path) -> tuple[Path, ...]:
    required = (package / "ner", *(package / relative for relative in _SOURCE_FILES))
    return tuple(path for path in required if not path.exists())


def _copy_source_code(package: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for relative in _SOURCE_FILES:
        _copy_file(package / relative, target / relative)
    _copy_tree(package / "ner", target / "ner")


def _copy_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Required staging file is missing or unsafe: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"Required staging directory is missing or unsafe: {source}")
    shutil.copytree(
        source,
        target,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if _has_symlink(target):
        raise ValueError(f"Symlinks are not allowed in staged source: {source}")


def _has_symlink(root: Path) -> bool:
    return any(path.is_symlink() for path in root.rglob("*"))


def _write_contract(path: Path, contract: Contract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--site", default=DEFAULT_SITE)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--gpu-model", default=DEFAULT_GPU_MODEL)
    parser.add_argument("--period", choices=("day", "night"), default=DEFAULT_PERIOD)
    parser.add_argument("--repo-id")
    parser.add_argument("--source", "--input", dest="source", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--requirements-lock", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _prepare_from_args(args):
        print(json.dumps({"staging_dir": str(args.staging_dir), "state": "prepared"}))
        return 0
    _require_controller_args(args)
    ledger = _run_controller(args)
    print(json.dumps({"run_id": ledger["run_id"], "state": ledger["state"]}, sort_keys=True))
    return 0


def _prepare_from_args(args: argparse.Namespace) -> bool:
    if not _preparation_requested(args):
        return False
    _require_prepare_args(args)
    prepare_geographic_ner_staging(
        args.staging_dir,
        source=args.source,
        contract=args.contract,
        source_root=args.source_root,
        requirements_lock=args.requirements_lock,
    )
    return args.prepare_only


def _preparation_requested(args: argparse.Namespace) -> bool:
    return bool(args.prepare_only or args.source is not None or args.contract is not None)


def _require_prepare_args(args: argparse.Namespace) -> None:
    if args.source is None or args.contract is None:
        raise SystemExit("--source/--input and --contract are required for staging preparation")


def _require_controller_args(args: argparse.Namespace) -> None:
    if _missing_controller_args(args):
        raise SystemExit("--run-dir, --run-id, and --repo-id are required to run the controller")


def _missing_controller_args(args: argparse.Namespace) -> bool:
    return any(value is None for value in (args.run_dir, args.run_id, args.repo_id))


def _run_controller(args: argparse.Namespace) -> dict[str, object]:
    return run_grid5000_ner_controller(
        staging_dir=args.staging_dir,
        run_dir=args.run_dir,
        run_id=args.run_id,
        site=args.site,
        queue=args.queue,
        gpu_model=args.gpu_model,
        period=args.period,
        repo_id=args.repo_id,
        publish=args.publish,
    )


if __name__ == "__main__":
    raise SystemExit(main())
