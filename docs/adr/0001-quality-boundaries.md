# ADR 0001: Quality boundaries

- Status: Accepted
- Date: 2026-09-12

## Decision

- Project-local `domain` imports may point only to the local `domain` layer;
  standard-library and third-party imports remain outside that local graph rule.
- The architecture check uses Python's standard-library `ast` parser to build a
  project-local import graph and detect cycles, domain violations, and
  pipeline-to-CLI edges.
- Full-source CRAP covers the configured `src`, `scripts`, and preprocessing
  source inventories, including nested functions via Radon's
  `--show-closures`. Mutation testing remains scoped to deterministic helpers
  and quality tools.

## Rationale and limitations

The import graph and CRAP report make structural and function-level quality
boundaries measurable without executing external services. Mutation remains
scoped because live network, GPU, publication, and large-data paths produce
slow or environment-dependent mutants; those boundaries use focused behavioral
and local integration tests instead.

The AST graph does not prove behavior for dynamic imports or third-party side
effects, and static checks do not prove runtime side-effect safety. When such a
boundary is isolated behind a deterministic adapter, the cleanup path is to add
behavioral tests for the boundary and expand the mutation scope to that
adapter. Until then, the integration tests are the applicable evidence.
