# Atomic Publication Deep Module Design

## Problem

`io/atomic.py` is the package's declared home for durable local file
publication, but it is a *shallow* module: it exposes two fixed convenience
writers (`atomic_write_text`, `atomic_write_parquet`) and hides no reusable
mechanism. Every caller that needs a different payload kind — a byte copy, a
deterministic JSON document, an uncompressed Parquet table, a matplotlib PNG,
a streamed `ParquetWriter` — therefore re-implements the same six-step ritual:

1. `path.parent.mkdir(parents=True, exist_ok=True)`
2. `tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)`
3. close the descriptor
4. write the payload into the temporary sibling
5. `os.replace(temporary, path)`
6. unlink the temporary sibling if anything raised

That ritual is currently hand-rolled in eleven places, including two
byte-for-byte identical `_copy_file_atomically` functions in
`grid5000/sentence_protocol.py` and `grid5000/sentence_controller.py`, a third
copy of the same function as `_atomic_copy` in
`pipeline/_wikidata_recovery/transaction.py`, and an
`_atomic_overwrite_parquet` in `augmentation/integrity.py` that duplicates
`io.atomic.atomic_write_parquet` exactly. A separate deterministic-JSON idiom,
`atomic_write_text(path, json_dumps(value) + "\n")`, is repeated at seven call
sites.

Duplicated durability code is the expensive kind: each copy independently
decides whether cleanup catches `Exception` or `BaseException`, whether the
temporary file is a hidden sibling of the target (so `os.replace` stays on one
filesystem and is therefore atomic), and whether missing parent directories are
created. `pipeline/containment_migration.py` already diverges by cleaning up in
a `finally` block instead of `except BaseException`, and
`hf/_geographic/rendering.py` diverges by using `NamedTemporaryFile(delete=False)`
and never creating parent directories.

## Goal

Turn `io.atomic` into a deep module: one small interface that owns the whole
publication ritual, with the existing writers expressed on top of it. No
public API of any other module changes, and no byte written to disk changes.

## Scope

- Add `atomic_replacement(path)`, a context manager that yields the temporary
  sibling and installs it over `path` when the block exits normally.
- Add `atomic_write_json(path, value)` for the deterministic
  `utils.json.dumps` + trailing-newline format used across the pipeline.
- Add `atomic_copy_file(source, target)` for durable byte copies.
- Express `atomic_write_text` and `atomic_write_parquet` on
  `atomic_replacement` without changing their observable behavior.
- Replace the eleven hand-rolled rituals and the seven deterministic-JSON call
  sites with these helpers.

Explicitly out of scope, because they are a different transaction shape and not
duplication of this ritual:

- multi-file transactions that must roll back together
  (`pipeline/persistence.py`, `v2/storage.py`);
- whole-directory publication via `mkdtemp` + `rmtree` + `os.replace`
  (`augmentation/checkpoints.py`, `pipeline/_wikidata_recovery/checkpoints.py`).

## Byte-compatibility constraints

Two JSON formats coexist and must both be preserved:

- deterministic, compact `utils.json.dumps` output for pipeline state and
  checkpoints — this is what `atomic_write_json` provides;
- human-readable `indent=2, sort_keys=True` output for migration journals in
  `pipeline/link_migration.py` and `pipeline/_link_migration/transaction.py`.

The second format is *not* unified into `atomic_write_json`; those two call
sites keep their own formatting and only adopt the shared ritual. Likewise
`pipeline/containment_migration.py` keeps its explicit `pq.write_table` call
rather than adopting `atomic_write_parquet`, so its Parquet encoding options
are unchanged by this refactor, and `_install_file` keeps `shutil.copy2` so it
still preserves source metadata.

## Alternatives

1. Leave the duplication in place. Zero risk, but every future durability fix
   has to be applied in eleven files, and the two already-diverging copies show
   that does not happen reliably.
2. Introduce an `AtomicPublisher` class with per-payload subclasses. This adds
   an inheritance hierarchy and lifecycle for a mechanism whose entire state is
   one temporary path; the interface would be larger than the code it hides.
3. Extract one context manager that owns the ritual and express every writer on
   it. Recommended: the interface is a single function, callers stop naming
   `tempfile`, `os.replace`, and cleanup at all, and the durability contract
   becomes testable in one place.

## Verification

RED → GREEN → REFACTOR per slice:

1. add focused failing tests for `atomic_replacement`, `atomic_write_json`, and
   `atomic_copy_file` and observe the import failure;
2. implement the smallest helpers that pass;
3. convert call sites one module at a time, keeping the full suite green;
4. add a bounded `crap-atomic` scope and require every measured function below
   6;
5. add `io/atomic.py` to the mutmut source scope so the consolidated durability
   logic keeps the mutation coverage it had while it lived inside
   `grid5000/sentence_protocol.py`, and require zero survivors;
6. run the complete `just quality-gauntlet`.
