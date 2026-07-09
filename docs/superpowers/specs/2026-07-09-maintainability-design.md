# Maintainability Design

## Goal

Improve internal clarity and typing without changing public APIs, CLI behavior,
schemas, request semantics, or output artifacts.

## Boundaries

- Domain records remain immutable, slot-backed value objects.
- Processor helpers accept concrete domain/enrichment types rather than
  untyped values.
- Public documentation explains the package layering and stable extension
  points; internal helpers stay private.
- Unused private compatibility remnants are removed only after the existing
  test suite proves they are not part of the supported surface.
