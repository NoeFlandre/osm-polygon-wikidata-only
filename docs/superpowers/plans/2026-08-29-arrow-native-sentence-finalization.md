# Arrow-Native Sentence Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant Python row materialization from resumable V2 sentence checkpoint validation and finalization while preserving every sentence output and restart contract.

**Architecture:** Keep SentenceCheckpoint.load_batch() as the compatibility API that returns Python dictionaries, and add load_batch_table() as the Arrow-native API for validation and output. The runner will use table row counts for resumed batches and write validated Arrow tables directly in deterministic checkpoint order. The sentence schema will be cached because it is immutable and argument-free.

**Tech Stack:** Python 3.12+, PyArrow Parquet, pytest, Ruff, ty, CRAP, mutmut, and the existing V2 sentence checkpoint contracts.

---

## File map

- Modify src/osm_polygon_wikidata_only/v2/sentence_logic.py: cache the immutable sentence_schema() result.
- Modify src/osm_polygon_wikidata_only/v2/sentence_checkpoints.py: expose schema-validated Arrow batches and use them for validity discovery.
- Modify src/osm_polygon_wikidata_only/v2/sentence_runner.py: avoid Python conversion for resumed batches and final output writes.
- Modify src/osm_polygon_wikidata_only/hf/_dataset_stats/scanning.py: read known single Parquet files without dataset discovery.
- Modify tests/v2/test_sentence_logic.py: lock the schema-cache contract.
- Modify tests/v2/test_sentence_checkpoints.py: lock Arrow-table loading and non-materializing completed-batch discovery.
- Modify tests/v2/test_sentence_runner.py: lock Arrow-native output finalization and exact output values/schema.
- Create tests/hf/test_dataset_stats_scanning.py: lock the single-file reader path and selected-column behavior.
- Do not modify Grid5000 controllers, HF publication code, manifests, dataset-card content, or processed data.

### Task 1: Add failing tests for the new Arrow-native contract

**Files:**
- Modify: tests/v2/test_sentence_logic.py
- Modify: tests/v2/test_sentence_checkpoints.py
- Modify: tests/v2/test_sentence_runner.py

- [ ] **Step 1: Add the schema identity test.**

Append this test to tests/v2/test_sentence_logic.py:

~~~python
def test_sentence_schema_is_cached() -> None:
    assert sentence_schema() is sentence_schema()
~~~

- [ ] **Step 2: Add the Arrow batch and validity tests.**

Add imports for pyarrow as pa, pytest, and sentence_schema, then append this test to tests/v2/test_sentence_checkpoints.py:

~~~python
def test_sentence_checkpoint_loads_validated_arrow_tables_without_row_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(0, [{"sentence_id": "sentence-1", "text": "First."}])

    table = checkpoint.load_batch_table(0)

    assert isinstance(table, pa.Table)
    assert table is not None
    assert table.schema.equals(sentence_schema(), check_metadata=True)
    assert table.to_pylist()[0]["text"] == "First."

    monkeypatch.setattr(
        checkpoint,
        "load_batch",
        lambda index: pytest.fail(f"Python rows were materialized for batch {index}"),
    )
    assert checkpoint.completed_batches == (0,)
~~~

- [ ] **Step 3: Add the output-path test.**

Add sentence_schema and _write_output to the existing imports in tests/v2/test_sentence_runner.py, then append:

~~~python
def test_sentence_output_writes_validated_checkpoint_tables_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SentenceCheckpoint(
        tmp_path / "checkpoints",
        "region-latest",
        "wikipedia",
        input_fingerprint="input-a",
        model_id="segment-any-text/sat-3l-sm",
        model_revision="model-a",
        batch_size=2,
    )
    checkpoint.write_batch(
        0,
        [
            {"sentence_id": "sentence-1", "text": "First."},
            {"sentence_id": "sentence-2", "text": "Second."},
        ],
    )
    checkpoint.mark_complete(batch_count=1, row_count=2)
    monkeypatch.setattr(
        checkpoint,
        "load_batch",
        lambda index: pytest.fail(f"Python rows were materialized for batch {index}"),
    )

    output_path = tmp_path / "sentences.parquet"
    _write_output(output_path, checkpoint, batch_count=1)
    output = pq.read_table(output_path)

    assert output.schema.equals(sentence_schema(), check_metadata=True)
    assert output.column("sentence_id").to_pylist() == ["sentence-1", "sentence-2"]
    assert output.column("text").to_pylist() == ["First.", "Second."]
~~~

- [ ] **Step 4: Run the focused tests and confirm RED.**

Run:

~~~bash
.venv/bin/pytest -q tests/v2/test_sentence_logic.py::test_sentence_schema_is_cached tests/v2/test_sentence_checkpoints.py::test_sentence_checkpoint_loads_validated_arrow_tables_without_row_conversion tests/v2/test_sentence_runner.py::test_sentence_output_writes_validated_checkpoint_tables_directly
~~~

Expected: failure because sentence_schema() is not cached, load_batch_table() does not exist, and _write_output() still calls load_batch().

- [ ] **Step 5: Commit only the failing tests.**

~~~bash
git add tests/v2/test_sentence_logic.py tests/v2/test_sentence_checkpoints.py tests/v2/test_sentence_runner.py
git commit -m "test: specify arrow-native sentence finalization"
~~~

### Task 2: Implement schema caching and the checkpoint table API

**Files:**
- Modify: src/osm_polygon_wikidata_only/v2/sentence_logic.py
- Modify: src/osm_polygon_wikidata_only/v2/sentence_checkpoints.py

- [ ] **Step 1: Cache the stable schema.**

In sentence_logic.py, import lru_cache from functools and decorate the existing sentence_schema() function with lru_cache(maxsize=1). Do not change its fields, order, or types.

- [ ] **Step 2: Add the schema-validated Arrow loader.**

In SentenceCheckpoint, add this method and make load_batch delegate to it:

~~~python
def load_batch_table(self, index: int) -> pa.Table | None:
    """Read one batch as a schema-validated Arrow table."""
    path = self._batch_path(index)
    try:
        with pq.ParquetFile(path) as parquet_file:
            table = parquet_file.read()
    except (OSError, pa.ArrowException):
        return None
    if not table.schema.equals(sentence_schema(), check_metadata=True):
        return None
    return table

def load_batch(self, index: int) -> list[dict[str, Any]] | None:
    """Read one batch, returning None for a missing or invalid file."""
    table = self.load_batch_table(index)
    return None if table is None else table.to_pylist()
~~~

Change completed_batches to call self.load_batch_table(index) is not None, not load_batch().

- [ ] **Step 3: Run the focused tests and confirm GREEN for the checkpoint changes.**

~~~bash
.venv/bin/pytest -q tests/v2/test_sentence_logic.py::test_sentence_schema_is_cached tests/v2/test_sentence_checkpoints.py::test_sentence_checkpoint_loads_validated_arrow_tables_without_row_conversion
~~~

Expected: PASS.

- [ ] **Step 4: Run all checkpoint and logic tests.**

~~~bash
.venv/bin/pytest -q tests/v2/test_sentence_logic.py tests/v2/test_sentence_checkpoints.py
~~~

Expected: PASS.

- [ ] **Step 5: Commit the checkpoint API.**

~~~bash
git add src/osm_polygon_wikidata_only/v2/sentence_logic.py src/osm_polygon_wikidata_only/v2/sentence_checkpoints.py
git commit -m "perf: expose arrow-native sentence checkpoint batches"
~~~

### Task 3: Use Arrow tables during resumed processing and finalization

**Files:**
- Modify: src/osm_polygon_wikidata_only/v2/sentence_runner.py

- [ ] **Step 1: Use the table row count for completed source batches.**

In _process_source(), replace the checkpoint load branch with:

~~~python
            table = checkpoint.load_batch_table(batch_index)
            if table is None:
                rows, _ = split_sections(
                    sections,
                    segmenter=segmenter,
                    batch_size=batch_size,
                )
                checkpoint.write_batch(batch_index, rows)
                row_count += len(rows)
            else:
                row_count += table.num_rows
            batch_count = batch_index + 1
~~~

Keep counter updates and source iteration unchanged.

- [ ] **Step 2: Write validated Arrow tables directly.**

In _write_output(), replace the load_batch() and pa.Table.from_pylist() block with:

~~~python
        for batch_index in range(batch_count):
            table = checkpoint.load_batch_table(batch_index)
            if table is None:
                raise ValueError(f"Invalid sentence checkpoint batch: {batch_index}")
            if table.num_rows:
                writer.write_table(table)
~~~

Retain the existing ParquetWriter, Snappy compression, close handling, schema validation, and atomic os.replace().

- [ ] **Step 3: Run the new focused tests and confirm GREEN.**

~~~bash
.venv/bin/pytest -q tests/v2/test_sentence_logic.py::test_sentence_schema_is_cached tests/v2/test_sentence_checkpoints.py::test_sentence_checkpoint_loads_validated_arrow_tables_without_row_conversion tests/v2/test_sentence_runner.py::test_sentence_output_writes_validated_checkpoint_tables_directly
~~~

Expected: PASS.

- [ ] **Step 4: Run the complete sentence regression tests.**

~~~bash
.venv/bin/pytest -q tests/v2/test_sentence_logic.py tests/v2/test_sentence_checkpoints.py tests/v2/test_sentence_runner.py tests/v2/test_sentence_docs.py tests/v2/test_sentence_cli.py tests/v2/test_sat.py
~~~

Expected: PASS with unchanged sentence values, unsupported-language behavior, manifest accounting, and CLI contracts.

- [ ] **Step 5: Commit the runner optimization.**

~~~bash
git add src/osm_polygon_wikidata_only/v2/sentence_runner.py
git commit -m "perf: finalize sentence sidecars with arrow tables"
~~~

### Task 4: Optimize the local dataset-card/statistics scanner

**Files:**
- Modify: src/osm_polygon_wikidata_only/hf/_dataset_stats/scanning.py
- Create: tests/hf/test_dataset_stats_scanning.py

- [ ] **Step 1: Add a failing scanner-path test.**

Create tests/hf/test_dataset_stats_scanning.py:

~~~python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._dataset_stats.scanning import safe_table


def test_safe_table_reads_selected_columns_without_dataset_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "values.parquet"
    pq.write_table(pa.table({"kept": [1, 2], "ignored": [3, 4]}), parquet_path)

    def fail_if_dataset_reader_is_used(*args: object, **kwargs: object) -> None:
        raise AssertionError("single-file dataset reader was used")

    monkeypatch.setattr(pq, "read_table", fail_if_dataset_reader_is_used)

    table = safe_table(parquet_path, ["kept"])

    assert table is not None
    assert table.column_names == ["kept"]
    assert table.column("kept").to_pylist() == [1, 2]
~~~

- [ ] **Step 2: Run the scanner test and confirm RED.**

~~~bash
.venv/bin/pytest -q tests/hf/test_dataset_stats_scanning.py::test_safe_table_reads_selected_columns_without_dataset_discovery
~~~

Expected: FAIL because safe_table() currently calls pq.read_table().

- [ ] **Step 3: Implement the direct single-file read.**

Replace the return in safe_table() with:

~~~python
        with pq.ParquetFile(parquet_path) as parquet_file:
            return parquet_file.read(columns=list(columns))
~~~

Keep the existing exception tuple, warning message, function signature, and
column-list conversion unchanged.

- [ ] **Step 4: Run scanner and dataset-statistics tests GREEN.**

~~~bash
.venv/bin/pytest -q tests/hf/test_dataset_stats_scanning.py tests/hf/test_dataset_stats.py
~~~

Expected: PASS with the existing factual stats and skip-on-error behavior.

- [ ] **Step 5: Run the full read-only statistics benchmark and compare the exact hash.**

~~~bash
time PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from osm_polygon_wikidata_only.hf._dataset_stats.aggregation import compute_dataset_stats; stats = compute_dataset_stats(Path("/Volumes/Seagate M3/projects/osm-polygon-wikidata-only/processed")); print(stats)'
~~~

Record wall time and verify the exact serialized stats hash remains
878029edfc723db6f52b8d64120e19c785bd0df823a5c5a7ebd252b10c422ee4. Do not
modify processed data or the dataset card.

### Task 5: Run all quality and performance gates

**Files:**
- No additional production files.

- [ ] **Step 1: Run the repository regression suite.**

~~~bash
.venv/bin/pytest -q
~~~

Expected: PASS; any collection or environment failure is reported as a blocker, not treated as success.

- [ ] **Step 2: Run static and type gates.**

~~~bash
just ruff
just ty
~~~

Expected: PASS.

- [ ] **Step 3: Run the CRAP gate.**

~~~bash
just crap
~~~

Expected: PASS with every changed function below CRAP 6.

- [ ] **Step 4: Run focused mutation verification.**

~~~bash
just mutation
~~~

Expected: zero surviving mutants for changed sentence checkpoint and runner logic. If the command reports unrelated pre-existing mutants, separate them from the changed scope and do not claim a clean mutation gate.

- [ ] **Step 5: Re-run the bounded finalization benchmark.**

Use the same 8,192-row, 32-batch benchmark fixture and compare the finalization median with the recorded 1.062-second baseline. Verify output row count, ordered sentence IDs, schema equality, and file readability.

- [ ] **Step 6: Review the diff and repository state.**

~~~bash
git diff origin/codex/grid5000-sentence-splitting...HEAD --check
git status --short --branch
git log -5 --oneline
~~~

Expected: only the committed spec, plan, tests, and sentence performance implementation are present; no data, HF, Grid5000, or unrelated user files are modified.
