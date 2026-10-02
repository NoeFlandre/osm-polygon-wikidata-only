# ADR 0001: Quality boundaries

- Status: Accepted
- Date: 2026-09-12

## Decision

- A project-local `domain` import can point only to the local `domain` layer. The rule does not apply to imports from the standard library or from third parties.
- The architecture check uses the `ast` parser of the Python standard library. It builds a project-local import graph. It finds cycles, domain violations, and edges from the pipeline to the CLI.
- The full-source CRAP report covers the configured source lists for `src`, `scripts`, and preprocessing. It includes nested functions through the Radon option `--show-closures`. Mutation testing stays limited to deterministic helpers and quality tools.

## Rationale and limits

The import graph and the CRAP report make the structural and function-level quality boundaries measurable. They do not run external services. Mutation testing stays limited because these paths make mutants that are slow or depend on the environment: live network, GPU, publication, and large data. Focused behavioral tests and local integration tests cover these paths.

The AST graph does not prove the behavior of dynamic imports or the side effects of third-party code. Static checks do not prove that runtime side effects are safe. When a deterministic adapter isolates such a boundary, add behavioral tests for the boundary. Then increase the mutation scope to include the adapter. Until then, the integration tests are the applicable evidence.
