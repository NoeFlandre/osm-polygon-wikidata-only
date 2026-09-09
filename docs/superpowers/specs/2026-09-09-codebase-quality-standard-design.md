# Codebase Quality Standard Design

**Status:** Approved for implementation on 2026-09-09.

## Goal

Raise the practical engineering standard of the repository without changing
public behavior, data formats, command contracts, or publication semantics.
The staged pilot-only geographic-NER cleanup remains preserved and is not
expanded with HF or Grid5000 activity.

## Approach

Use a measurement-driven sequence rather than a broad rewrite:

1. Establish one source of truth for deterministic quality scopes and validate
   that every declared source and test exists.
2. Bring the nested `preprocessing/` package into representative local and CI
   checks without coupling its lockfile or imports to the root package.
3. Verify the built wheel and packaged runtime resources from an installed
   artifact, not only from the source checkout.
4. Make documentation site assembly reproducible locally and in Pages.
5. Strengthen typing, exception boundaries, and tests only where the preceding
   checks expose a concrete defect or drift.

Each change starts with a focused failing test, is implemented minimally, and
is refactored only while the relevant tests remain green. Existing APIs and
outputs are treated as contracts. Remote publication, Grid5000 jobs, and
production data are out of scope.

## Quality gates

For every implementation slice: focused tests, full tracked tests, Ruff check
and format check, strict typing, documentation build, package build/install
smoke, CRAP below 6 for configured scopes, mutation results with no
unexplained survivors, and `git diff --check`. Any unavailable gate is
reported with its exact environmental blocker rather than being inferred.

## Deliberate non-goals

- Do not delete compatibility facades or public modules based only on static
  reachability.
- Do not change dataset-card content, HF repositories, Grid5000 state, or
  sentence/NER artifacts.
- Do not weaken coverage, mutation, typing, or lint configuration to make a
  check pass.
