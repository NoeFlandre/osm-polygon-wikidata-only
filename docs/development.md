# Development guide

## Setup

The project supports Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/)
for the locked environment:

```bash
uv sync --frozen
uv run pre-commit install
just --list
```

The source checkout contains code, tests, and documentation. Keep PBFs, Parquet
files, credentials, generated cards, and other run output in an
operator-selected data root outside the checkout.

Quality reports, temporary files, local tool state, and generated documentation
also stay outside the checkout. `QUALITY_RUNTIME_DIR` defaults to the portable
`../quality-runtime`; `QUALITY_TMP_DIR` defaults to its `tmp` directory. Set
them explicitly when sharing a runtime location:

```bash
export QUALITY_RUNTIME_DIR="${QUALITY_RUNTIME_DIR:-../quality-runtime}"
export QUALITY_TMP_DIR="${QUALITY_TMP_DIR:-${QUALITY_RUNTIME_DIR}/tmp}"
just quality-runtime
```

The optional V2 sentence stage is installed separately so the normal pipeline
does not pull a large model runtime:

```bash
uv sync --extra sentence-splitting
```

Its exact routing and resumability contract is documented in the
[sentence-splitting guide](sentence-splitting.md).

The GPU extra is reserved for the Grid5000 compute-node job:

```bash
uv sync --extra sentence-splitting-gpu
```

The local controller, short-job policy, CUDA requirement, token boundary, and
resume/publish contract are documented in the
[Grid5000 sentence operations guide](grid5000-sentence-splitting.md). The
external data root remains authoritative; the controller stages only bounded
batch inputs and keeps HF authentication local.

## Docker reproducibility

The checked-in `Dockerfile` has `runtime` and `development` targets. Both are
built from the locked `uv.lock` environment; the runtime image runs as a
non-root, unprivileged `app` user and contains no source data or credentials.
The `docker build` command selects either target without changing the host data
root.

Build and exercise the harmless default command:

```bash
just docker-build
just docker-help
```

Run development checks without production data:

```bash
just docker-test
just docker-check
```

The development image declares no ambient `OSM_POLYGON_DATA_ROOT`; the suite
resolves its own temporary roots, and only the runtime image declares that
contract with the `/data` volume behind it. The image also runs
`pytest -m "not repository"`, because `presentations/` is deliberately kept out
of the build context. Those repository-completeness contracts run in the
`quality` CI job against a full checkout.

To run the opt-in workflow, provide a host data root containing `raw/`:

```bash
just docker-run /path/to/osm-polygon-data
```

The recipe uses Docker's explicit `--mount` form, mounts the data root at
`/data`, and mounts `/data/raw` read-only. It passes `HF_TOKEN` and optional
Wikimedia credentials only at runtime. Removing the container does not remove
the host's resumable state; press `Ctrl-C` and rerun the same command to resume.
Docker builds, help, and tests do not read a real PBF or make Hugging
Face/Wikimedia requests.

## Wikimedia credentials

Wikimedia authentication is optional. The [README authentication section](https://github.com/NoeFlandre/osm-polygon-wikidata-only#wikimedia-bot-password-authentication)
describes how to create and revoke a least-privilege Bot Password. Keep the
password out of source files, logs, issues, and pull requests. The
`WIKIMEDIA_BOT_USERNAME`/`WIKIMEDIA_BOT_PASSWORD` pair is all-or-nothing:
supply the password securely, never log it, and do not commit it. The
`WIKIMEDIA_REQUESTS_PER_MINUTE` environment variable selects the client-side
request ceiling; that ceiling remains subject to Wikimedia's service limits.

The test suite never uses live credentials. Authentication tests pass explicit
environment mappings and fake transports (for example,
`tests/enrichment/test_wikimedia_auth.py` and
`tests/cli/test_dependencies.py`) and assert that errors do not echo secrets.

## Tests and quality checks

Use red-green-refactor for behavior or configuration changes: add one focused
failing test, confirm the expected failure, implement the smallest change, and
then refactor while the test remains green. Contract tests should check the
observable CLI, schema, workflow, or documentation behavior rather than private
implementation details.

The deterministic pre-completion gate is:

```bash
just quality-gauntlet
```

The command runs the current quality recipes once, in a fixed fail-fast order:

```bash
just baseline
just ruff
just ty
just tests
just property-tests
just acceptance-tests
just architecture-checks
just crap-report
just mutation
just smoke-test
just diff-review
```

`just property-tests` runs deterministic Hypothesis properties for lossless
sentence routing and batch-boundary invariance. `just acceptance-tests` runs
the pytest-bdd resumability scenario plus local pipeline integration tests.
Those checks use real local Parquet, JSON, and manifest formats while stubbing
external clients; they do not require live network or GPU services. A resumed
fixture run is compared with a clean replay so retry behavior cannot silently
change rows, offsets, or routing.

`just architecture-checks` runs the local import-graph rules for cycles,
domain purity, and pipeline-to-CLI direction, alongside package, documentation,
and CLI contract checks. The smoke stage checks both public CLI help paths
without reading a data root or making a network request. The Docker runtime has
its own `docker-help` recipe and CI container contract.
`diff-review` runs `git diff --check` and a short branch status check.

`just mutation` deletes the generated `mutants/` tree before each run. Mutation
scope is configured in `pyproject.toml` and uses `mutate_only_covered_lines`,
so the mutant population depends on mutmut's own coverage attribution. Reusing
incremental mutmut state after a source edit was observed to generate 2801
mutants instead of the 3038 produced from a clean tree, which weakens the gate
without failing it. Regenerating from scratch keeps the reported mutant count
reproducible, at the cost of a full run each time. The gate itself refuses any
non-killed result and has no configured equivalence exemptions.

The root coverage report combines `osm_polygon_wikidata_only` and `scripts`
with branch measurement and enforces the configured 90% total coverage floor.
Preprocessing coverage is reported separately before its full-source CRAP check;
the aggregate floor does not replace the function-level CRAP threshold.

### Nested preprocessing package

The repository contains a separate locked distribution under `preprocessing/`.
Its tests and wheel build use `preprocessing/uv.lock`; the root Ruff and ty
installations lint and type-check its source without merging the two package
environments. Run the boundary gate before changing that package:

    just preprocessing-check

The gate runs frozen preprocessing tests, Ruff, ty, and an offline isolated wheel
build/install smoke.
It does not read production data or publish artifacts.

`just check` and the short alias `just qa-gauntlet` run the same deterministic
completion gate locally:

```bash
just check
```

### Mutation and complexity gates

The canonical gate runs full-source CRAP after the root tests and the
preprocessing checks included in `architecture-checks`. `just crap-report`
joins those coverage reports with Radon reports for `src`, `scripts`, and
`preprocessing/src`; `--show-closures` includes nested functions, and any
function at CRAP 6 or higher fails. `just crap-all` is the standalone variant
that refreshes both coverage reports before reporting. Historical `crap-*`
aliases delegate to that same full-source run; they are compatibility names,
not separate focused inventories.

`just mutation` runs mutmut over the explicit deterministic helper and quality
tool scope. The gate rejects unreviewed survivors, timeouts, untested results,
and other non-killed statuses; exact source-bound equivalent mutations may
pass only through the reviewed equivalence mechanism. Mutation remains
intentionally scoped: network clients, large data, publication, live GPU work,
and other external side effects are covered by focused integration or
operational checks instead. Static Ruff and ty checks constrain source shape
and types; they do not prove runtime side-effect safety. Reports and temporary
files stay under the configured quality runtime, and mutation uses two workers
to bound memory without reading production data. HTMLParser trampoline
mutations remain unsupported by mutmut 3.7 and are not actionable.

```bash
just crap
just mutation
just quality-advanced
```

Run `uv run pre-commit run --all-files` before opening a pull request. The
hooks intentionally run the fast Ruff and `ty` subset; `just check` and
GitHub Actions both use the complete `just quality-gauntlet` gate.

## Test strength checks

The normal gate already runs full-source CRAP and the scoped mutation gate.
The opt-in `just quality-strength` recipe repeats the standalone full-source
CRAP refresh and mutation checks; it does not add a narrower module inventory.

```bash
just quality-strength
```

Reviewed source-bound equivalent mutations are reported separately from
killed mutants; incomplete or unreviewed mutation results still fail. The
mutation scope deliberately excludes live network, GPU, publication, and
large-data behavior, while full-source CRAP covers the configured source
inventory including nested functions.

## Documentation and contribution

Build the site without starting a server:

```bash
just docs
```

The strict build writes the site and its assembled public artifacts below the
configured quality runtime. It is safe to point the runtime at a portable
operator-owned location; no generated site or report belongs in the checkout.

Navigation targets must exist under `docs/`, links and images must resolve in a
clean checkout, and public examples must use current CLI options. The Pages
workflow builds with `--strict`, uploads only the generated site, and deploys
that artifact with least-privilege permissions.

Please read the repository's [contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md)
before proposing changes. Keep pull requests small, explain any compatibility
effect on schemas or CLI options, and never commit source data or credentials.

## Read-only operator audit

The `osm-polygon-wikidata-only-audit-remote` command reports local/remote
publication differences without uploading or deleting files:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer parses this command, Rich renders the report, and tqdm shows progress only
when stderr is interactive. It is separate from the stable argparse processing
CLI and does not change dataset output.

## Release checklist

Before a release, run the complete gate, inspect the wheel for `py.typed` and
the license, verify CLI help from the built artifact, review dataset schemas and
attribution, and update the version intentionally. Publishing software or
datasets is a maintainer action; ordinary tests do not publish anything.
