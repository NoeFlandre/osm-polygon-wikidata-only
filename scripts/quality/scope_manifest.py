"""Validated file boundaries for deterministic quality scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class QualityScope:
    """A named set of source and test files covered by one quality gate."""

    name: str
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]


NER_SCOPE: Final = QualityScope(
    name="ner",
    source_paths=(
        "src/osm_polygon_wikidata_only/ner/job.py",
        "src/osm_polygon_wikidata_only/ner/otter.py",
        "src/osm_polygon_wikidata_only/ner/pipeline.py",
        "src/osm_polygon_wikidata_only/ner/publication.py",
        "src/osm_polygon_wikidata_only/grid5000/ner_controller.py",
        "scripts/grid5000_geographic_ner.py",
        "scripts/prepare_geographic_ner_pilot.py",
    ),
    test_paths=(
        "tests/ner/test_job.py",
        "tests/ner/test_otter.py",
        "tests/ner/test_pipeline.py",
        "tests/ner/test_publication.py",
        "tests/grid5000/test_ner_controller.py",
        "tests/ner/test_pilot.py",
        "tests/ner/test_utf8_metadata.py",
    ),
)


def validate_scope(scope: QualityScope, root: Path) -> None:
    """Raise ``ValueError`` when a scope has duplicate or missing files."""
    for kind, paths in (("source", scope.source_paths), ("test", scope.test_paths)):
        if len(paths) != len(set(paths)):
            raise ValueError(f"{scope.name} has a duplicate {kind} path")
        for relative in paths:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{scope.name} has an unsafe {kind} path: {relative}")
            if not (root / path).is_file():
                raise ValueError(f"{scope.name} {kind} path does not exist: {relative}")

    overlap = sorted(set(scope.source_paths).intersection(scope.test_paths))
    if overlap:
        raise ValueError(f"{scope.name} paths appear in both source and test paths: {overlap}")


def validate_scopes(scopes: tuple[QualityScope, ...], root: Path) -> None:
    """Validate each scope and reject duplicate scope names."""
    names = tuple(scope.name for scope in scopes)
    if len(names) != len(set(names)):
        raise ValueError("quality scopes contain a duplicate name")
    for scope in scopes:
        validate_scope(scope, root)


SCOPES: Final = (NER_SCOPE,)


def scope_by_name(name: str) -> QualityScope:
    """Return one named scope or raise a useful error for an unknown name."""
    for scope in SCOPES:
        if scope.name == name:
            return scope
    raise ValueError(f"unknown quality scope: {name}")


def render_scope_paths(scope: QualityScope, kind: str) -> str:
    """Render one scope's paths for a shell command substitution."""
    if kind == "source":
        paths = scope.source_paths
    elif kind == "test":
        paths = scope.test_paths
    else:
        raise ValueError(f"unknown quality scope path kind: {kind}")
    return " ".join(paths)


def main(argv: tuple[str, ...] | None = None) -> int:
    """Print a manifest path list for Justfile command substitution."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--kind", choices=("source", "test"), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    validate_scopes(SCOPES, root)
    print(render_scope_paths(scope_by_name(args.scope), args.kind))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
