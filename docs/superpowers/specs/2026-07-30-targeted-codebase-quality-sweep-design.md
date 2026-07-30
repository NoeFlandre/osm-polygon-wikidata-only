# Targeted Codebase Quality Sweep

## Goal

Improve maintainability and public-facing quality without changing dataset
content, schemas, manifests, cache contracts, command-line behavior, request
semantics, resumability, or publication output.

## Constraints

- Keep `main` as the only development branch.
- Use uv for environments and commands, Ruff for linting/formatting, and ty for
  production and maintained-script type checking.
- Remove mypy from dependencies, configuration, CI, and current documentation.
- Follow RED -> GREEN TDD for every production refactor.
- Preserve stable facade modules and object identities where they are currently
  public or characterized by tests.
- Do not access or mutate the Seagate data root, Wikimedia, or Hugging Face.
- Do not split files merely to meet an arbitrary line-count target.

## Current State

The source tree is already organized into coherent top-level packages:
`augmentation`, `cli`, `config`, `domain`, `enrichment`, `hf`, `io`,
`pipeline`, and `utils`. A wholesale package redesign would add churn without
public value.

The maintainability issue is concentrated in five modules:

| Module | Approximate LOC | Primary issue |
|---|---:|---|
| `pipeline/link_migration.py` | 1,131 | Models, planning, conversion, and transactional application share one file |
| `pipeline/_wikidata_recovery/repair.py` | 1,085 | Coordination, fetching, merging, validation, and manifest staging are mixed |
| `hf/publication.py` | 1,046 | Snapshots, validation, loading, and operation assembly are mixed |
| `cli/run_sync.py` | 1,029 | A 580-line composition root contains too many nested collaborators |
| `hf/_dataset_stats/augmentation.py` | 885 | File scanning, cache serialization, aggregation, and orchestration are mixed |

The first ty baseline over `src scripts` reports 17 errors and 3 warnings.
Most errors are unvalidated JSON/object boundaries rather than broad typing
failures.

## Architecture

### Stable facades

Existing import paths remain the compatibility surface. Focused private modules
own implementation:

- `pipeline/_link_migration/` owns models, planning/conversion, and transaction
  execution; `pipeline/link_migration.py` remains a facade.
- `pipeline/_wikidata_recovery/` keeps `repair.py` as the coordinator and moves
  row merging/validation and manifest staging into focused siblings.
- `hf/_publication/` owns snapshot generation, artifact validation/loading, and
  upload assembly; `hf/publication.py` remains a facade.
- `cli/_sync/` owns preflight, publication cleanup/queue construction, and
  recovery helpers; `cli/run_sync.py` remains the `execute` composition root.
- `hf/_dataset_stats/` splits augmentation scanning/cache serialization from
  aggregation; `augmentation.py` remains the public-internal coordinator.

### Type-checking boundary

Pin ty in the uv development dependency group and check `src` plus maintained
`scripts`. Tests remain covered by pytest and Ruff. Resolve diagnostics using
explicit runtime validation, `TypedDict`, `Protocol`, `TypeGuard`, or casts only
after validation. Do not add blanket ignores or disable diagnostic classes to
make the gate green.

### Documentation

Keep the public README focused on installation, dataset semantics, commands, and
contribution quality gates. Keep internal architecture and package ownership in
`docs/architecture.md`; keep local development commands in
`docs/development.md`. Each implementation package receives a concise package
docstring describing ownership. Historical plans remain historical and are not
rewritten to pretend they used ty.

## Verification

Every extraction starts with a characterization or ownership test that fails
for the intended missing boundary. Focused tests must pass before the full
suite. Final gates:

```bash
uv sync --frozen
uv run pytest -q
uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q
uv run ruff check .
uv run ruff format --check .
uv run ty check src scripts
uv build
git diff --check
```

The final diff must contain no dataset files, credentials, generated cache
artifacts, schema changes, manifest changes, or remote-path changes.
