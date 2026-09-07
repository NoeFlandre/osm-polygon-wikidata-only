"""Validate exact, source-bound reviews; never infer equivalence from a score."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate review key: {key}")
        result[key] = value
    return result


def _source_path(name: str) -> Path:
    parts = name.split(".")
    if len(parts) < 2 or not all(part.isidentifier() for part in parts):
        raise ValueError(f"Invalid exact mutant name: {name}")
    prefix = Path() if parts[0] == "scripts" else Path("src")
    return prefix.joinpath(*parts[:-1]).with_suffix(".py")


def _review_fields(entry: object) -> dict[str, str]:
    required = {"source_sha256", "mutant_sha256", "reason"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError("Review requires exact source, mutant and reason fields")
    fields = cast(dict[str, object], entry)
    return {key: _review_text(fields[key]) for key in required}


def _review_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Review fields must be non-empty strings")
    return value


def _mutant_digest(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one generated mutant: {name}")
    return hashlib.sha256(ast.dump(matches[0]).encode()).hexdigest()


def _verify_review(name: str, entry: object, source_root: Path, mutants_root: Path) -> None:
    fields = _review_fields(entry)
    relative = _source_path(name)
    source_digest = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if source_digest != fields["source_sha256"]:
        raise ValueError(f"Source changed since equivalence review: {name}")
    digest = _mutant_digest(mutants_root / relative, name.rsplit(".", 1)[-1])
    if digest != fields["mutant_sha256"]:
        raise ValueError(f"Mutation changed since equivalence review: {name}")


def reviewed_equivalents(
    results: Sequence[tuple[str, str]],
    path: Path,
    *,
    source_root: Path,
    mutants_root: Path,
) -> frozenset[str]:
    """Reject stale, malformed or drifted reviews before accepting a survivor."""
    reviews = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(reviews, dict):
        raise ValueError("Equivalence reviews must be an object")
    statuses = dict(results)
    verified: set[str] = set()
    for name, entry in reviews.items():
        if statuses.get(name) != "survived":
            raise ValueError(f"Stale equivalence review: {name}")
        _verify_review(name, entry, source_root, mutants_root)
        verified.add(name)
    return frozenset(verified)
