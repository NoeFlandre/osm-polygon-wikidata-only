# Fast Sentence Publication Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify published sentence batches from Hub metadata whenever exact LFS SHA-256 values are available, avoiding redundant downloads while preserving exact-byte fallback verification.

**Architecture:** Keep the existing publication plan and controller state machine intact. Extend `RemoteInventory` with a metadata-backed exact-path lookup, then make the sentence publisher choose metadata hashing for LFS files and the current download/hash path for regular Git files. Missing files and mismatches continue to fail publication before the ledger reaches `published`.

**Tech Stack:** Python, `huggingface_hub.HfApi.get_paths_info`, dataclasses, pytest, Ruff, ty, CRAP, mutmut.

---

### Task 1: Add the remote metadata contract and RED tests

**Files:**
- Modify: `src/osm_polygon_wikidata_only/hf/_uploader/protocol.py`
- Modify: `src/osm_polygon_wikidata_only/hf/remote_inventory.py`
- Modify: `src/osm_polygon_wikidata_only/hf/_uploader/stub.py`
- Test: `tests/hf/test_reconciliation.py`

- [ ] **Step 1: Write the failing metadata lookup test**

Add a fake Hub path-info response and assert that an exact-path lookup returns only requested files, their sizes, and the LFS SHA-256 value. The fake must expose `get_paths_info` but not rely on the existing full-repository listing:

```python
from types import SimpleNamespace

def test_fetch_paths_reads_exact_file_metadata() -> None:
    class Hub:
        token = "token"

        def get_paths_info(self, **kwargs):
            assert kwargs == {
                "repo_id": "test/repo",
                "paths": ["wikipedia/sentences/alpha-latest.parquet", "README.md"],
                "repo_type": "dataset",
            }
            return [
                SimpleNamespace(
                    path="wikipedia/sentences/alpha-latest.parquet",
                    size=12,
                    lfs=SimpleNamespace(sha256="a" * 64),
                ),
                SimpleNamespace(path="README.md", size=8, lfs=None),
            ]

    inventory = RemoteInventory.fetch_paths(
        "test/repo",
        paths=["wikipedia/sentences/alpha-latest.parquet", "README.md"],
        hub=Hub(),
    )

    assert inventory.files == {
        "wikipedia/sentences/alpha-latest.parquet",
        "README.md",
    }
    assert inventory.metadata("wikipedia/sentences/alpha-latest.parquet").sha256 == "a" * 64
    assert inventory.metadata("README.md").sha256 is None
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/hf/test_reconciliation.py -k fetch_paths -q`

Expected: FAIL because `RemoteInventory.fetch_paths` and its metadata accessor do not exist yet.

- [ ] **Step 3: Add the minimal metadata types and API method**

Add a frozen `RemoteFileInfo(path: str, size: int, sha256: str | None)` dataclass. Store an optional path-to-info mapping in `RemoteInventory`. Add `fetch_paths` that calls `get_paths_info(repo_id=repo_id, paths=list(paths), repo_type="dataset")`, ignores non-file entries without a valid size, translates Hub exceptions with `_translate_hf_error`, and returns an inventory whose `files` set contains the returned file paths. Add `metadata(path)` returning the stored info.

Extend the `HfHub` protocol and `StubHfHub` test double with:

```python
def get_paths_info(
    self,
    repo_id: str,
    paths: list[str],
    *,
    repo_type: str,
) -> Iterable[Any]: ...
```

If a runtime Hub client does not expose `get_paths_info`, make `fetch_paths` delegate to the existing `fetch` method so the caller retains the previous full-inventory plus exact-download verification behavior.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `pytest tests/hf/test_reconciliation.py -k fetch_paths -q`

Expected: PASS.

- [ ] **Step 5: Commit the metadata contract**

```bash
git add src/osm_polygon_wikidata_only/hf/_uploader/protocol.py src/osm_polygon_wikidata_only/hf/remote_inventory.py tests/hf/test_reconciliation.py
git commit -m "perf: add exact Hub path metadata lookup"
```

### Task 2: Switch sentence verification to metadata-first hashing

**Files:**
- Modify: `src/osm_polygon_wikidata_only/grid5000/sentence_controller.py`
- Test: `tests/grid5000/test_sentence_controller.py`

- [ ] **Step 1: Write the failing no-download verification test**

Create local sentence, manifest, README, and map artifacts from the existing test helper. Patch `RemoteInventory.fetch_paths` to return `RemoteFileInfo` entries with correct sizes and SHA-256 values for all expected files, and patch `_download_hf_file` to raise an assertion. Call `HfHubSentencePublisher.verify_sentence_batch`; it must complete without downloading any file.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/grid5000/test_sentence_controller.py -k metadata -q`

Expected: FAIL because the publisher still calls `RemoteInventory.fetch` and downloads every expected file.

- [ ] **Step 3: Implement metadata-first verification**

Replace the full inventory call with `RemoteInventory.fetch_paths` for the exact publication paths. Preserve the current missing-path error. For every expected local file:

```python
info = inventory.metadata(remote)
if info is not None and info.sha256:
    if info.size != local.stat().st_size or sha256_file(local) != info.sha256:
        raise ControllerRunError(f"HF sentence publication hash mismatch: {remote}")
    continue
downloaded = _download_hf_file(...)
if sha256_file(downloaded) != sha256_file(local):
    raise ControllerRunError(f"HF sentence publication hash mismatch: {remote}")
```

Keep the fallback in the existing temporary verification directory so non-LFS files and Hub responses without LFS digests retain exact-byte checking.

- [ ] **Step 4: Add the regular-file fallback regression test**

Return metadata with `sha256=None` for one expected file, patch `_download_hf_file` to copy a matching local file into the requested temporary directory, and assert that the downloader was called for that file. Also assert a wrong LFS size or digest raises the existing mismatch error.

- [ ] **Step 5: Run focused verification tests and verify GREEN**

Run: `pytest tests/grid5000/test_sentence_controller.py -k 'metadata or publisher' -q`

Expected: PASS with no download for LFS entries and download fallback for regular entries.

- [ ] **Step 6: Commit the metadata-first publisher**

```bash
git add src/osm_polygon_wikidata_only/grid5000/sentence_controller.py tests/grid5000/test_sentence_controller.py
git commit -m "perf: verify sentence uploads from Hub metadata"
```

### Task 3: Run the complete quality and publication safety checks

**Files:**
- No additional production files.

- [ ] **Step 1: Run the full project quality gauntlet**

Run: `just quality-gauntlet`

Expected: all stages pass, including tests, coverage, CRAP with every score below 6, and mutation testing with no survivors.

- [ ] **Step 2: Inspect the diff and source state**

Run: `git diff origin/codex/grid5000-sentence-splitting...HEAD --check` and `git status --short --branch`

Expected: only the design, plan, metadata contract, and publisher changes are present; no generated dataset or unrelated user files are staged.

- [ ] **Step 3: Push the optimized source branch**

```bash
git push origin codex/grid5000-sentence-splitting
```

- [ ] **Step 4: Resume the existing ledger with the optimized controller**

Stop the old controller only after the current batch is in `ready_to_publish` or `published`, then restart the same command using the existing external data root and `HF_HUB_DISABLE_XET=1`. Never reset the ledger or resubmit a batch whose state is already `published`.

- [ ] **Step 5: Verify the next publication path**

Confirm the next batch reaches `published`, the protected README and comparison-map hashes are unchanged, and the ledger remains valid JSON before allowing the controller to continue through all remaining batches.
