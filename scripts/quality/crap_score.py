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

    valid_complexity = _validate_complexity(complexity)
    fraction = _validate_coverage(coverage)
    return valid_complexity**2 * (1.0 - fraction) ** 3 + valid_complexity


def _validate_complexity(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("complexity must be a positive integer")
    if not isinstance(value, int) or value < 1:
        raise ValueError("complexity must be a positive integer")
    return value


def _validate_coverage(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coverage must be a finite fraction between 0 and 1")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("coverage must be a finite fraction between 0 and 1")
    return fraction


def evaluate_threshold(entries: Iterable[CrapEntry], *, maximum: float) -> list[CrapEntry]:
    """Return entries whose CRAP score is at or above ``maximum``."""

    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum must be a non-negative finite number")
    return sorted(
        (entry for entry in entries if entry.crap >= maximum),
        key=lambda entry: (-entry.crap, entry.path, entry.line, entry.name),
    )


def _entries_for_file(
    raw_path: object,
    raw_blocks: object,
    coverage_files: Mapping[str, object],
) -> list[CrapEntry]:
    path, blocks = _radon_file_parts(raw_path, raw_blocks)
    entries: list[CrapEntry] = []
    for raw_block in blocks:
        block = _mapping(raw_block, "Radon function entry")
        if block.get("type") not in {"function", "method"}:
            continue
        qualified_name, complexity, line, endline = _radon_function_parts(block)
        file_data = _coverage_file(coverage_files, path)
        function = _function_coverage(file_data, qualified_name)
        coverage = (
            _function_coverage_fraction(function)
            if function
            else _line_coverage_fraction(file_data, line, endline)
        )
        entries.append(CrapEntry(path, qualified_name, complexity, coverage, line))
    return entries


def _radon_file_parts(
    raw_path: object,
    raw_blocks: object,
) -> tuple[str, list[object]]:
    if not isinstance(raw_path, str):
        raise ValueError("Radon report has an invalid file entry")
    if not isinstance(raw_blocks, list):
        raise ValueError("Radon report has an invalid file entry")
    return raw_path, cast(list[object], raw_blocks)


def _radon_function_parts(
    block: Mapping[str, object],
) -> tuple[str, int, int, int]:
    name = _required_string(block.get("name"), "Radon function entry is malformed")
    classname = _optional_string(block.get("classname"), "Radon function classname is malformed")
    complexity = _required_int(
        block.get("complexity"),
        "Radon function entry is malformed",
    )
    line = _required_int(block.get("lineno", 0), "Radon function line is malformed")
    endline = _required_int(
        block.get("endline", line),
        "Radon function endline is malformed",
    )
    if endline < line:
        raise ValueError("Radon function endline is malformed")
    qualified_name = f"{classname}.{name}" if classname else name
    return qualified_name, complexity, line, endline


def _required_string(value: object, error: str) -> str:
    if not isinstance(value, str):
        raise ValueError(error)
    return value


def _optional_string(value: object, error: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, error)


def _required_int(value: object, error: str) -> int:
    if isinstance(value, bool):
        raise ValueError(error)
    if not isinstance(value, int):
        raise ValueError(error)
    return value


def entries_from_reports(
    coverage_report: Mapping[str, object],
    complexity_report: Mapping[str, object],
) -> list[CrapEntry]:
    """Join coverage.py function data with Radon function complexity."""

    coverage_files = _mapping_value(coverage_report, "files")
    entries = [
        entry
        for raw_path, raw_blocks in complexity_report.items()
        for entry in _entries_for_file(raw_path, raw_blocks, coverage_files)
    ]
    if not entries:
        raise ValueError("reports contain no function entries")
    return entries


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], dict(value))


def _mapping_value(value: Mapping[str, object], key: str) -> dict[str, object]:
    return _mapping(value.get(key), f"report {key}")


def _coverage_file(files: Mapping[str, object], path: str) -> dict[str, object]:
    raw_file = files.get(path)
    if raw_file is None:
        suffix = Path(path).as_posix()
        matches = [value for candidate, value in files.items() if str(candidate).endswith(suffix)]
        raw_file = matches[0] if len(matches) == 1 else None
    return _mapping(raw_file, f"coverage file {path}")


def _function_coverage(file_data: Mapping[str, object], name: str) -> dict[str, object]:
    functions = _mapping_value(file_data, "functions")
    function = functions.get(name)
    return {} if function is None else _mapping(function, f"coverage function {name}")


def _line_coverage_fraction(file_data: Mapping[str, object], start: int, end: int) -> float:
    """Estimate function coverage from executed/missing line data.

    Coverage.py can omit tiny helpers from its function map while still
    recording their executed lines. Restricting the calculation to lines
    coverage.py classified as executable keeps the fallback faithful to the
    report instead of counting docstrings and blank lines as misses.
    """
    executed = _line_numbers(file_data.get("executed_lines", []), "executed_lines")
    missing = _line_numbers(file_data.get("missing_lines", []), "missing_lines")
    scope = set(range(start, end + 1))
    executable = (executed | missing) & scope
    if not executable:
        return 0.0
    return len(executed & executable) / len(executable)


def _line_numbers(value: object, label: str) -> set[int]:
    if not isinstance(value, list):
        raise ValueError(f"coverage {label} must be a list of line numbers")
    if any(_invalid_line_number(line) for line in value):
        raise ValueError(f"coverage {label} must be a list of line numbers")
    return {cast(int, line) for line in value}


def _invalid_line_number(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int)


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
