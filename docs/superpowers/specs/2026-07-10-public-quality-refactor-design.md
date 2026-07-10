# Public Quality Refactor Design

## Goal

Raise the repository to a high public-facing engineering standard without
adding features or changing runtime behavior. Improve cohesion, modularity,
typing, tests, documentation, packaging metadata, and continuous verification
while preserving the pipeline's complete multilingual data contract.

## Compatibility contract

The refactor preserves:

- both CLI commands, arguments, defaults, exit behavior, and logging meaning;
- polygon eligibility, source order, limits, deterministic identifiers, and
  enrichment selection;
- Wikimedia URLs, request batching, retries, rate coordination, exact-revision
  fallback behavior, caching semantics, and completeness failures;
- all three Parquet schemas, values, row ordering, paths, manifests, dataset
  card behavior, and Hugging Face upload ordering;
- documented client classes and functions, plus existing common imports used by
  the test suite and README; and
- cache and durable upload-job formats so interrupted production work resumes.

Underscore-prefixed helpers and undocumented implementation details may move.
Compatibility facades retain established imports from `wikipedia_client.py`,
`wikidata_client.py`, `processor.py`, and `commands.py`.

## Refactoring strategy

Use characterization-first refactoring. Before moving behavior, add focused
tests at the current boundary and observe them pass against the existing code.
Move one responsibility at a time behind the same interface, rerun focused and
full tests, then commit. Do not combine a behavior change with a structural
move. If a characterization test reveals ambiguous existing behavior, preserve
it and document it rather than choosing a new interpretation.

Avoid framework layers, dependency-injection containers, abstract base classes
without multiple implementations, generic repositories, or configuration
systems. Small functions, typed protocols, immutable value objects, and direct
composition are sufficient.

## Target package structure

### Wikimedia enrichment

Create focused internal packages while retaining the current public facades:

```text
enrichment/
  wikipedia/
    models.py       # WikipediaArticle, FetchResult, capability protocols
    parsing.py      # query/batch/parse response parsing and HTML conversion
    transport.py    # URL construction and scheduled HTTP JSON requests
    cache.py        # versioned cached client adapter and serialization
    client.py       # HTTP and in-memory clients; fallback orchestration
  wikidata/
    models.py       # WikidataEntity, sitelink types, capability protocol
    parsing.py      # QID/site validation and response parsing
    transport.py    # URL construction and scheduled HTTP JSON requests
    cache.py        # versioned cached client adapter and serialization
    client.py       # abstract, HTTP, and in-memory clients
  wikipedia_client.py  # compatibility re-exports only
  wikidata_client.py   # compatibility re-exports only
```

Transport modules own HTTP mechanics but not domain parsing or cache policy.
Parsing modules are pure. Cache modules own keys and serialization but delegate
network work. Client modules compose these pieces and keep the established
public behavior. `article_linker.py` depends on public model/protocol types,
not concrete transport details.

### Pipeline

Reduce `processor.py` to the single-PBF workflow:

```text
pipeline/
  rows.py           # polygon enrichment copies, article rows, link rows
  completeness.py   # incomplete-enrichment error and audits
  publication.py    # temporary Parquet paths, atomic promotion, manifest update
  processor.py      # stage sequencing and ProcessResult/PbfStem facade
```

Each module receives explicit typed inputs and returns domain values. Row
construction remains deterministic and pure. Publication owns filesystem
mutation. The processor records stage timing and coordinates the modules.

### CLI and background publication

Reduce `commands.py` to application flow:

```text
cli/
  parser.py         # argparse construction and language parsing
  dependencies.py   # settings, data root, scheduler, clients, caches
  publication.py    # dataset-card snapshot and background upload composition
  commands.py       # main(), orchestration, shutdown, exit status; re-exports
```

The CLI continues using direct construction. No service locator or framework is
introduced. Upload state remains under the external data root and is never
written into the repository.

### Tests

Mirror production responsibilities without discarding end-to-end contracts:

```text
tests/enrichment/   # parsing, clients, cache adapters, linker
tests/pipeline/     # rows, completeness, publication, processor, orchestrator
tests/cli/          # parser, dependencies, command lifecycle
tests/hf/           # cards, layouts, upload queue, uploader
```

Shared builders live in small `conftest.py` files nearest their consumers.
Tests assert outputs and state transitions rather than private call sequences.
Existing end-to-end and schema tests remain as compatibility sentinels.

## Typing and API clarity

- Keep mypy strict for all source files.
- Replace broad `Any` at internal boundaries with dataclasses, mappings,
  protocols, callables, and typed aliases. Retain `Any` only at decoded JSON or
  third-party boundaries, narrowing immediately through validation.
- Give every public class, function, and module a concise docstring explaining
  its contract, failure behavior, and side effects.
- Define `__all__` for compatibility facades and intentional public modules.
- Avoid re-exporting new internal helpers. Public imports stay small and stable.
- Use immutable, slot-backed records where identity/value semantics apply; do
  not convert mutable lifecycle objects such as queues into value objects.

## Documentation and packaging

Improve public presentation without promising unsupported behavior:

- Rewrite README navigation around purpose, guarantees, quick start,
  production operation, resume/failure behavior, dataset layout, development,
  and licensing.
- Expand `docs/architecture.md` with package boundaries, dependency direction,
  processing sequence, completeness transaction, and background publication.
- Add `docs/development.md` with environment setup, test strategy, quality
  commands, TDD workflow, repository/data separation, and release checklist.
- Add `docs/api.md` documenting supported Python entry points and compatibility
  policy.
- Add `CONTRIBUTING.md` and `SECURITY.md`. Identify Noé Flandre as maintainer;
  security reporting uses a non-sensitive public contact mechanism already
  present in project metadata, not an invented private address.
- Complete PEP 621 metadata in `pyproject.toml`: author, license file, keywords,
  classifiers, project URLs, typed-package marker, and supported Python version.
- Add `py.typed` to the wheel and verify source/wheel contents.
- Add a GitHub Actions workflow that installs with `uv` and runs tests with
  coverage, Ruff lint, formatting check, mypy, and package build on Python 3.12.
- Keep generated data, caches, PBFs, and Parquet files outside the repository.

No changelog automation, documentation site generator, release bot, dependency
bot, benchmark service, or plugin system is introduced.

## Test and quality gates

Follow red-green-refactor for behavior tests and characterization-move-green for
pure structural changes. Required gates:

1. existing tests pass before every extraction;
2. focused characterization tests cover moved public behavior and failure paths;
3. aggregate branch-aware coverage is raised from 77% to at least 80%, with no
   exclusion pragmas added merely to satisfy the number;
4. `ruff check`, `ruff format --check`, strict mypy, and package build pass;
5. wheel inspection confirms `py.typed`, license, and package modules;
6. import-compatibility tests confirm supported legacy module imports;
7. snapshot/row-equivalence tests confirm schemas, ordering, manifests, cache
   keys, and dataset-card content remain unchanged; and
8. automated tests perform no production-scale PBF run and no network access.

CI runs the same commands documented for contributors. Local verification is
the source of truth before merge; CI configuration does not replace it.

## Error handling and observability

Exception types and user-facing failure meaning remain stable. Refactoring may
improve exception chaining, internal diagnostic context, and docstrings, but it
must not downgrade errors into warnings or change fail-closed publication.

Logs retain existing stage and completion messages. Internal module moves do
not add noisy per-item logging, expose article text, or reveal credentials.

## Completion criteria

The work is complete when the target boundaries are in place, compatibility
facades are thin, tests mirror responsibilities, documentation and metadata are
public-ready, all quality gates pass, and a clean checkout can build and invoke
the CLI. File count or line count alone is not a success metric; each extracted
module must have one clear responsibility and reduce coupling in its caller.
