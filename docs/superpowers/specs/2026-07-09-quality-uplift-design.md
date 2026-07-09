# Quality Uplift Design

## Goal

Make the codebase easier to understand, safer to extend, and accurate for
public users without changing CLI behavior, public imports, dataset rows, or
network semantics.

## Scope

- Replace untyped dynamic batch-method discovery with typed runtime-checkable
  capability protocols while retaining single-item client compatibility.
- Extract the shared article-row construction from `processor.py` into a
  focused helper so deduplication and metadata construction are explicit.
- Clarify package exports and public API documentation.
- Correct README examples and document cache, batching, ordering, retries,
  and operational workflows.
- Add contract tests for capability fallback and README-backed client setup.

## Constraints

No schema, eligibility, request selection, cache-key, retry, ordering, or CLI
change is allowed. Every refactor is protected by a focused failing test plus
the full existing suite and static checks.
