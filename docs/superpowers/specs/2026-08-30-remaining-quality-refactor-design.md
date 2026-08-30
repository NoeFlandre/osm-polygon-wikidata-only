# Remaining Quality Refactor Design

## Goal

Reduce the last tested complexity-B hotspots without changing public APIs,
observable outputs, subprocess arguments, Hugging Face requests, exception
messages, or credential-handling behavior.

## Scope

The refactor is limited to two existing boundaries:

- `grid5000/sentence_job.py`: separate the external `nvidia-smi` invocation
  from deterministic GPU-output parsing.
- `hf/remote_inventory.py`: share Hub-client construction and separate remote
  path metadata decoding from the network request and fallback behavior.

No new runtime capability, retry policy, cache, dependency, or public symbol is
introduced. Existing external calls remain at the same boundary with the same
arguments and error translation.

## Alternatives

1. Leave the hotspots unchanged. This has no regression risk but leaves the
   remaining complexity and duplicated client setup in place.
2. Rewrite the Grid5000 and Hugging Face flows around new service classes.
   This could create larger test seams but adds abstractions and compatibility
   risk without a concrete need.
3. Extract small private helpers for pure parsing, client selection, and
   metadata construction while retaining the current orchestration. This is
   the recommended approach because it lowers complexity at the existing
   boundaries and keeps behavior localized.

## Design

`_query_gpus()` will continue to execute the fixed `nvidia-smi` command and
retain its return-code and empty-output errors. A private parser will receive
the captured stdout, ignore blank lines, preserve line order, and reuse the
existing per-line validation and error text. Direct tests will cover blank
output, multiple GPUs, malformed lines, and command failure.

`RemoteInventory.fetch()` and `fetch_paths()` will use one private client
resolver that preserves the explicit-hub short circuit and otherwise resolves
the token exactly once before calling `_build_hf_api()`. `fetch_paths()` will
retain its legacy-client fallback to `fetch()`, requested-path order in the
Hub call, metadata filtering, and translated exceptions. Private helpers will
validate path/size fields and normalize an optional 64-character SHA-256 value
without changing the resulting `RemoteFileInfo` values.

## Verification

Each slice follows RED → GREEN → REFACTOR:

1. add one focused test and observe its expected failure;
2. implement the smallest private helper or delegation needed to pass;
3. run the focused tests, then the complete relevant test scope;
4. run CRAP for both scopes and require every measured function below 6;
5. run the full tracked regression suite, Ruff, format, ty, documentation,
   architecture, acceptance, and the configured mutation gate.

The configured mutmut scope remains unchanged: it covers deterministic
helpers, while these network/subprocess and file-system-adjacent boundaries
remain protected by focused tests and CRAP rather than unstable external
runtime mutation campaigns.
