"""Compute function-level CRAP scores from coverage and Radon JSON reports."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class CrapEntry:
    """One function-level CRAP observation."""

    path: str
    name: str
    complexity: int
    coverage: float
    line: int = 0

    @property
    def crap(self) -> float:
        """Return the standard CRAP score for this function."""

        return crap_score(self.complexity, self.coverage)


def crap_score(complexity: int, coverage: float) -> float:
    """Return ``complexity^2 * (1 - coverage)^3 + complexity``."""

    if isinstance(complexity, bool) or not isinstance(complexity, int) or complexity < 1:
        raise ValueError("complexity must be a positive integer")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise ValueError("coverage must be a finite fraction between 0 and 1")
    fraction = float(coverage)
    return complexity**2 * (1.0 - fraction) ** 3 + complexity


def evaluate_threshold(entries: Iterable[CrapEntry], *, maximum: float) -> list[CrapEntry]:
    """Return entries whose CRAP score is at or above ``maximum``."""

    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum must be a non-negative finite number")
    return sorted(
        (entry for entry in entries if entry.crap >= maximum),
        key=lambda entry: (-entry.crap, entry.path, entry.line, entry.name),
    )


def entries_from_reports(
    coverage_report: Mapping[str, object],
    complexity_report: Mapping[str, object],
) -> list[CrapEntry]:
    """Join coverage.py function data with Radon function complexity."""

    coverage_files = _mapping_value(coverage_report, "files")
    entries: list[CrapEntry] = []
    for raw_path, raw_blocks in complexity_report.items():
        if not isinstance(raw_path, str) or not isinstance(raw_blocks, list):
            raise ValueError("Radon report has an invalid file entry")
        for raw_block in raw_blocks:
            block = _mapping(raw_block, "Radon function entry")
            if block.get("type") not in {"function", "method"}:
                continue
            name = block.get("name")
            classname = block.get("classname")
            complexity = block.get("complexity")
            line = block.get("lineno", 0)
            if (
                not isinstance(name, str)
                or not isinstance(complexity, int)
                or isinstance(complexity, bool)
            ):
                raise ValueError("Radon function entry is malformed")
            if classname is not None and not isinstance(classname, str):
                raise ValueError("Radon function classname is malformed")
            if not isinstance(line, int) or isinstance(line, bool):
                raise ValueError("Radon function line is malformed")
            qualified_name = f"{classname}.{name}" if classname else name
            function = _function_coverage(coverage_files, raw_path, qualified_name)
            entries.append(
                CrapEntry(
                    raw_path,
                    qualified_name,
                    complexity,
                    _function_coverage_fraction(function),
                    line,
                )
            )
    if not entries:
        raise ValueError("reports contain no function entries")
    return entries


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], dict(value))


def _mapping_value(value: Mapping[str, object], key: str) -> dict[str, object]:
    return _mapping(value.get(key), f"report {key}")


def _function_coverage(files: Mapping[str, object], path: str, name: str) -> dict[str, object]:
    raw_file = files.get(path)
    if raw_file is None:
        suffix = Path(path).as_posix()
        matches = [value for candidate, value in files.items() if str(candidate).endswith(suffix)]
        raw_file = matches[0] if len(matches) == 1 else None
    file_data = _mapping(raw_file, f"coverage file {path}")
    functions = _mapping_value(file_data, "functions")
    function = functions.get(name)
    return {} if function is None else _mapping(function, f"coverage function {path}:{name}")


def _function_coverage_fraction(function: Mapping[str, object]) -> float:
    """Return a function's coverage fraction, treating omitted functions as uncovered."""
    if not function:
        return 0.0
    summary = _mapping(function.get("summary"), "function coverage")
    percent = summary.get("percent_statements_covered", summary.get("percent_covered"))
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        raise ValueError("coverage function summary has no percentage")
    return float(percent) / 100.0


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON report {path}: {exc}") from exc
    return _mapping(value, f"JSON report {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--maximum", type=float, default=6.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the CRAP report and return non-zero for threshold violations."""

    args = _parser().parse_args(argv)
    entries = entries_from_reports(_load_json(args.coverage), _load_json(args.complexity))
    offenders = evaluate_threshold(entries, maximum=args.maximum)
    print(f"CRAP functions: {len(entries)}; maximum allowed: {args.maximum:.2f}")
    for entry in sorted(entries, key=lambda item: (-item.crap, item.path, item.line, item.name)):
        print(
            f"{entry.crap:7.2f}  complexity={entry.complexity:2d} "
            f"coverage={entry.coverage * 100:5.1f}%  {entry.path}:{entry.line} {entry.name}"
        )
    if offenders:
        print(f"CRAP threshold exceeded by {len(offenders)} function(s)")
        return 1
    print("CRAP threshold passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
