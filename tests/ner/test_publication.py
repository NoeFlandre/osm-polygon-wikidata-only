"""Offline contract tests for additive NER publication."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.io.hashing import sha256_file

MODULE = "osm_polygon_wikidata_only.ner.publication"

CONTRACT_ID_MISMATCH = "contract_id mismatch"
ARTIFACT_ORDER = "Artifact order is not contiguous"
BATCH_REQUIRED = "At least one batch is required"
BATCH_ROWS = "Processed rows differ from completed batch rows"


PREFIX = "geographic_ner/pilot-1/"


def raises_exactly(message, kind=ValueError):
    """Pin an operator-facing diagnostic so message-only mutations cannot survive."""
    return pytest.raises(kind, match=rf"^{re.escape(message)}$")


def module():
    return importlib.import_module(MODULE)


def extension_receipt(**overrides):
    """A minimally valid namespace receipt for the additive-extension invariants."""
    value = dict(
        contract={"model": "whoisjones/otter-cross-mmbert"},
        contract_id="a" * 64,
        source_sha256="b" * 64,
        source_rows=10,
        batch_size=1,
        processed_rows=1,
        status="paused",
        artifacts=[{"path": "batch-000000.parquet", "sha256": "c" * 64, "rows": 1}],
    )
    value.update(overrides)
    return value


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def publisher():
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "NER publisher is not implemented"
    return importlib.import_module(MODULE).publish_ner_shard


def receipt(
    root: Path,
    *,
    batches: int = 1,
    status: str = "paused",
    source_rows: int = 10,
    batch_size: int = 1,
    artifact_rows: int = 1,
) -> dict:
    artifacts = []
    for index in range(batches):
        path = root / f"batch-{index:06d}.parquet"
        pq.write_table(pa.table({"text": [f"Paris {index}"] * artifact_rows}), path)
        artifacts.append({"path": path.name, "sha256": sha256_file(path), "rows": artifact_rows})
    contract = {"model": "whoisjones/otter-cross-mmbert"}
    value = dict(
        contract=contract,
        contract_id=contract_digest(contract),
        source_sha256="b" * 64,
        source_rows=source_rows,
        batch_size=batch_size,
        processed_rows=batches * artifact_rows,
        status=status,
        artifacts=artifacts,
    )
    write_receipt(root, value)
    return value


def write_receipt(root: Path, value: dict) -> None:
    (root / "receipt.json").write_text(json.dumps(value))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    state = SimpleNamespace(
        files={"README.md": b"original card", "sentences/data.parquet": b"keep"},
        revisions={},
        reads=[],
        download_tokens=[],
        download_dirs=[],
        commits=[],
        corrupt=None,
        race=False,
        expected_prefix=PREFIX,
    )
    state.revisions["parent"] = dict(state.files)
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="parent")
    api.list_repo_files.side_effect = lambda repo_id, **kw: list(state.revisions[kw["revision"]])

    def download(repo_id, filename, *, revision, local_dir, **kwargs):
        state.reads.append((revision, filename))
        state.download_tokens.append(kwargs.get("token"))
        state.download_dirs.append(Path(local_dir))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(state.revisions[revision][filename])
        return str(target)

    def commit(repo_id, operations, *, parent_commit, **kwargs):
        assert parent_commit == "parent"
        assert kwargs["repo_type"] == "dataset"
        if state.race:
            raise RuntimeError("parent commit conflict")
        operations = list(operations)
        state.commits.append(operations)
        result = dict(state.revisions[parent_commit])
        for operation in operations:
            assert isinstance(operation, huggingface_hub.CommitOperationAdd)
            assert operation.path_in_repo.startswith(state.expected_prefix)
            with operation.as_file() as stream:
                result[operation.path_in_repo] = stream.read()
        if state.corrupt:
            result[state.corrupt] = b"corrupt"
        state.revisions["commit"] = result
        return huggingface_hub.CommitInfo(
            commit_url="https://huggingface.co/datasets/a/b/commit/commit",
            commit_message="test",
            commit_description="",
            oid="commit",
        )

    api.create_commit.side_effect = commit
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    state.api = api
    state.api_factory = huggingface_hub.HfApi
    return state


@pytest.mark.parametrize("status", ["paused", "completed"])
def test_publishes_only_receipt_and_batches_and_verifies_revision(tmp_path, hub, status):
    receipt(tmp_path, status=status, source_rows=1 if status == "completed" else 10)
    (tmp_path / "README.md").write_text("never upload")
    result = publisher()(tmp_path, repo_id="a/b", run_id="pilot-1", token="explicit")
    assert result == "commit"
    hub.api_factory.assert_called_once_with(token="explicit")
    hub.api.repo_info.assert_called_once_with("a/b", repo_type="dataset", revision="main")
    assert hub.download_tokens
    assert set(hub.download_tokens) == {"explicit"}
    assert {path.name for path in hub.download_dirs} == {"remote"}
    assert {path.parent.parent for path in hub.download_dirs} == {tmp_path}
    commit_call = hub.api.create_commit.call_args
    assert commit_call.args == ("a/b",)
    assert commit_call.kwargs["repo_type"] == "dataset"
    assert commit_call.kwargs["revision"] == "main"
    assert commit_call.kwargs["parent_commit"] == "parent"
    assert commit_call.kwargs["commit_message"] == f"Add geographic NER sidecars: {PREFIX}"
    assert commit_call.kwargs["num_threads"] == 1
    assert {op.path_in_repo for op in hub.commits[0]} == {
        PREFIX + "receipt.json",
        PREFIX + "batch-000000.parquet",
    }
    assert hub.revisions[result]["README.md"] == b"original card"
    assert (result, PREFIX + "batch-000000.parquet") in hub.reads
    assert (result, PREFIX + "receipt.json") in hub.reads
    assert ("parent", "README.md") in hub.reads
    assert (result, "README.md") in hub.reads


def test_completed_receipt_must_account_for_every_source_row(tmp_path, hub):
    receipt(tmp_path, status="completed")
    with raises_exactly("completed receipt must include every source row"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_contract_id_must_bind_the_receipt_contract(tmp_path, hub):
    value = receipt(tmp_path)
    value["contract"]["model"] = "unapproved/model"
    write_receipt(tmp_path, value)
    with raises_exactly(CONTRACT_ID_MISMATCH):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_contract_digest_is_ordered_unicode_and_separator_stable(tmp_path, hub):
    value = receipt(tmp_path)
    value["contract"] = {"z": "é", "a": "Paris"}
    value["contract_id"] = contract_digest(value["contract"])
    write_receipt(tmp_path, value)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"


def test_contract_digest_rejects_nan(tmp_path, hub):
    value = receipt(tmp_path)
    value["contract"] = {"score": float("nan")}
    value["contract_id"] = "0" * 64
    write_receipt(tmp_path, value)
    with pytest.raises(ValueError, match=r"^Receipt contract is not canonical JSON$"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_uppercase_sha256_digests_are_accepted(tmp_path, hub):
    value = receipt(tmp_path)
    value["contract_id"] = value["contract_id"].upper()
    value["source_sha256"] = value["source_sha256"].upper()
    value["artifacts"][0]["sha256"] = value["artifacts"][0]["sha256"].upper()
    write_receipt(tmp_path, value)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"


def test_snapshot_uses_canonical_receipt_path_case(tmp_path, monkeypatch):
    value = receipt(tmp_path)
    module = importlib.import_module(MODULE)
    calls = []

    def copy(source, target):
        calls.append((source.name, target.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    original_receipt = module._receipt

    def read_receipt(path):
        assert path.name == "receipt.json"
        return original_receipt(path)

    monkeypatch.setattr(module, "_copy", copy)
    monkeypatch.setattr(module, "_receipt", read_receipt)
    module._snapshot(tmp_path, tmp_path / "stage")

    assert calls[0] == ("receipt.json", "receipt.json")
    assert value["artifacts"][0]["path"] == "batch-000000.parquet"


def test_expected_files_reads_canonical_receipt_path_case(tmp_path, monkeypatch):
    receipt(tmp_path)
    module = importlib.import_module(MODULE)
    original_hash = module.sha256_file

    def hash_file(path):
        assert path.name == "receipt.json"
        return original_hash(path)

    monkeypatch.setattr(module, "sha256_file", hash_file)
    expected = module._expected_files(
        PREFIX, tmp_path, json.loads((tmp_path / "receipt.json").read_text())
    )

    assert PREFIX + "receipt.json" in expected


def test_batch_size_is_part_of_the_existing_run_identity(tmp_path, hub):
    receipt(tmp_path, batch_size=128)
    seed(tmp_path, hub)
    receipt(tmp_path, batches=2, batch_size=256)
    with raises_exactly("Run namespace identity differs: batch_size"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


@pytest.mark.parametrize("run_id", ["", "..", "../x", "/x", "x/y", "x\\y", "x\n", ".hidden"])
def test_rejects_unsafe_run_ids_before_hub(tmp_path, hub, run_id):
    receipt(tmp_path)
    with raises_exactly("Unsafe run_id"):
        publisher()(tmp_path, repo_id="a/b", run_id=run_id)
    hub.api.create_commit.assert_not_called()


def test_accepts_uppercase_run_id(tmp_path, hub):
    receipt(tmp_path)
    hub.expected_prefix = "geographic_ner/Pilot-1/"
    assert publisher()(tmp_path, repo_id="a/b", run_id="Pilot-1") == "commit"


@pytest.mark.parametrize("parent_sha", ["", None, 123])
def test_requires_a_nonempty_string_parent_commit(tmp_path, hub, parent_sha):
    receipt(tmp_path)
    hub.api.repo_info.return_value = SimpleNamespace(sha=parent_sha)
    with pytest.raises(ValueError, match=r"^Dataset has no parent commit$"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


LOCAL_DEFECT_MESSAGES = {
    "hash": "SHA256 mismatch: batch-000000.parquet",
    "path": "Unsafe artifact path",
    "duplicate": ARTIFACT_ORDER,
    "empty": BATCH_REQUIRED,
    "status": "Receipt is not publishable",
    "digest": "Invalid SHA256 digest",
    "rows": BATCH_ROWS,
    "count": "Processed rows exceed source rows",
    "batch_size": "Batch size must be positive",
    "object": BATCH_REQUIRED,
    "symlink": "Symlink is not a publishable artifact: batch-000000.parquet",
}


NAMESPACE_DEFECT_MESSAGES = {
    "contract": CONTRACT_ID_MISMATCH,
    "contract_id": CONTRACT_ID_MISMATCH,
    "source_sha256": "Run namespace identity differs: source_sha256",
    "source_rows": "Run namespace identity differs: source_rows",
    "regress": BATCH_ROWS,
    "remove": ARTIFACT_ORDER,
    "remote_bytes": f"Remote SHA256 mismatch: {PREFIX}batch-000000.parquet",
    "orphan": "Existing namespace differs from its receipt",
    "no_receipt": "Existing namespace has no receipt",
    "artifact_metadata": BATCH_ROWS,
}


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "hash",
        "path",
        "duplicate",
        "empty",
        "status",
        "digest",
        "rows",
        "count",
        "batch_size",
        "object",
        "symlink",
    ],
)
def test_rejects_invalid_local_artifacts_before_upload(tmp_path, hub, defect):
    value = receipt(tmp_path)
    batch = tmp_path / "batch-000000.parquet"
    if defect == "missing":
        batch.unlink()
    elif defect == "hash":
        batch.write_bytes(b"corrupt")
    elif defect == "path":
        value["artifacts"][0]["path"] = "../batch-000000.parquet"
    elif defect == "duplicate":
        value["artifacts"] *= 2
    elif defect == "empty":
        value["artifacts"] = []
    elif defect == "status":
        value["status"] = "running"
    elif defect == "digest":
        value["source_sha256"] = "invalid"
    elif defect == "rows":
        value["artifacts"][0]["rows"] = 2
    elif defect == "count":
        value["processed_rows"] = 11
    elif defect == "batch_size":
        value["batch_size"] = 0
    elif defect == "object":
        value["artifacts"] = {"batch-000000.parquet": value["artifacts"][0]}
    else:
        batch.rename(tmp_path / "other.parquet")
        batch.symlink_to(tmp_path / "other.parquet")
    write_receipt(tmp_path, value)
    expectation = (
        pytest.raises(
            FileNotFoundError,
            match=r"^\[Errno 2\] No such file or directory: '.*/batch-000000\.parquet'$",
        )
        if defect == "missing"
        else raises_exactly(LOCAL_DEFECT_MESSAGES[defect])
    )
    with expectation:
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def seed(root, hub):
    for name in ["receipt.json", "batch-000000.parquet"]:
        hub.revisions["parent"][PREFIX + name] = (root / name).read_bytes()


def test_identical_retry_has_no_commit(tmp_path, hub):
    receipt(tmp_path)
    seed(tmp_path, hub)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "parent"
    hub.api.create_commit.assert_not_called()
    assert ("parent", PREFIX + "batch-000000.parquet") in hub.reads


def test_identical_completed_retry_has_no_commit(tmp_path, hub):
    receipt(tmp_path, status="completed", source_rows=1)
    seed(tmp_path, hub)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "parent"
    hub.api.create_commit.assert_not_called()


def test_zero_row_batch_is_a_valid_nonnegative_count(tmp_path, hub):
    receipt(tmp_path, source_rows=0, artifact_rows=0)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"


def test_extends_receipt_without_reuploading_old_batch(tmp_path, hub):
    receipt(tmp_path)
    seed(tmp_path, hub)
    receipt(tmp_path, batches=2)
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"
    assert {op.path_in_repo for op in hub.commits[0]} == {
        PREFIX + "receipt.json",
        PREFIX + "batch-000001.parquet",
    }


def test_completed_namespace_cannot_regress_to_paused(tmp_path, hub):
    receipt(tmp_path, status="completed", source_rows=1)
    seed(tmp_path, hub)
    receipt(tmp_path, status="paused", source_rows=1)
    with raises_exactly("completed namespace cannot regress"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_rejects_noncontiguous_batch_names(tmp_path, hub):
    value = receipt(tmp_path)
    (tmp_path / "batch-000000.parquet").rename(tmp_path / "batch-000002.parquet")
    value["artifacts"][0]["path"] = "batch-000002.parquet"
    write_receipt(tmp_path, value)
    with raises_exactly(ARTIFACT_ORDER):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


@pytest.mark.parametrize(
    "defect",
    [
        "contract",
        "contract_id",
        "source_sha256",
        "source_rows",
        "regress",
        "remove",
        "remote_bytes",
        "orphan",
        "no_receipt",
        "artifact_metadata",
    ],
)
def test_rejects_namespace_conflicts(tmp_path, hub, defect):
    value = receipt(tmp_path)
    seed(tmp_path, hub)
    if defect in {"contract_id", "source_sha256"}:
        value[defect] = "c" * 64
    elif defect == "contract":
        value[defect] = {"model": "different"}
    elif defect == "source_rows":
        value[defect] = 20
    elif defect == "regress":
        value["processed_rows"] = 0
    elif defect == "remove":
        value["artifacts"][0]["path"] = "batch-000001.parquet"
        (tmp_path / "batch-000000.parquet").rename(tmp_path / "batch-000001.parquet")
    elif defect == "remote_bytes":
        hub.revisions["parent"][PREFIX + "batch-000000.parquet"] = b"bad"
    elif defect == "orphan":
        hub.revisions["parent"][PREFIX + "unexpected"] = b"bad"
    elif defect == "no_receipt":
        del hub.revisions["parent"][PREFIX + "receipt.json"]
    else:
        old = json.loads(hub.revisions["parent"][PREFIX + "receipt.json"])
        old["artifacts"][0]["rows"] = 2
        hub.revisions["parent"][PREFIX + "receipt.json"] = json.dumps(old).encode()
    write_receipt(tmp_path, value)
    with raises_exactly(NAMESPACE_DEFECT_MESSAGES[defect]):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


@pytest.mark.parametrize(
    "path", ["README.md", PREFIX + "receipt.json", PREFIX + "batch-000000.parquet"]
)
def test_rejects_corrupt_returned_commit(tmp_path, hub, path):
    receipt(tmp_path)
    hub.corrupt = path
    message = (
        "README changed during publication"
        if path == "README.md"
        else f"Remote SHA256 mismatch: {path}"
    )
    with raises_exactly(message):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")


def test_rejects_extra_file_in_returned_namespace(tmp_path, hub):
    receipt(tmp_path)
    hub.corrupt = PREFIX + "unexpected.parquet"
    with raises_exactly("Published namespace differs from receipt"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")


def test_parent_conflict_is_not_retried_blindly(tmp_path, hub):
    receipt(tmp_path)
    hub.race = True
    with raises_exactly("parent commit conflict", RuntimeError):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    assert hub.api.create_commit.call_count == 1


def test_rejects_receipt_progress_not_backed_by_batches(tmp_path, hub):
    value = receipt(tmp_path)
    value["processed_rows"] = 2
    write_receipt(tmp_path, value)
    with raises_exactly(BATCH_ROWS):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_parquet_hashes_are_streamed_and_upload_uses_snapshot(tmp_path, hub, monkeypatch):
    value = receipt(tmp_path)
    original_read = Path.read_bytes

    def small_read(path):
        assert path.suffix != ".parquet", "Parquets must not be loaded into memory"
        return original_read(path)

    def worker_progress(*args, **kwargs):
        (tmp_path / "batch-000000.parquet").write_bytes(b"worker changed local file")
        return SimpleNamespace(sha="parent")

    monkeypatch.setattr(Path, "read_bytes", small_read)
    hub.api.repo_info.side_effect = worker_progress
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"
    assert (
        hashlib.sha256(hub.revisions["commit"][PREFIX + "batch-000000.parquet"]).hexdigest()
        == value["artifacts"][0]["sha256"]
    )
    assert not list(tmp_path.glob(".ner-publication-*"))


def test_dataset_without_readme_does_not_create_one(tmp_path, hub):
    receipt(tmp_path)
    del hub.revisions["parent"]["README.md"]
    assert publisher()(tmp_path, repo_id="a/b", run_id="pilot-1") == "commit"
    assert "README.md" not in hub.revisions["commit"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1", "Receipt row counts must be integers"),
        (True, "Receipt row counts must be integers"),
        (1.0, "Receipt row counts must be integers"),
        (None, "Receipt row counts must be integers"),
        (-1, "Receipt row counts must be nonnegative"),
    ],
)
def test_receipt_counts_must_be_nonnegative_integers(value, message):
    with raises_exactly(message):
        module()._count(value)


@pytest.mark.parametrize("value", [None, [], "batch-000000.parquet"])
def test_artifact_records_must_be_mappings(value):
    with raises_exactly("Invalid artifact record"):
        module()._artifact(value)


def test_duplicate_artifact_paths_are_rejected_by_the_index_invariant():
    record = {"path": "batch-000000.parquet", "sha256": "a" * 64, "rows": 1}
    with raises_exactly("Duplicate artifact paths"):
        module()._index_artifacts([record, dict(record)])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "Receipt must be an object"),
        ('{"contract": "not-an-object"}', "Receipt contract must be an object"),
    ],
)
def test_receipt_must_be_an_object_carrying_an_object_contract(tmp_path, payload, message):
    path = tmp_path / "receipt.json"
    path.write_text(payload)
    with raises_exactly(message):
        module()._receipt(path)


def test_extension_rejects_regressing_processed_rows():
    with raises_exactly("Processed rows must not regress"):
        module()._extension(extension_receipt(processed_rows=2), extension_receipt())


def test_extension_rejects_changing_an_existing_artifact():
    new = extension_receipt(
        artifacts=[{"path": "batch-000000.parquet", "sha256": "d" * 64, "rows": 1}]
    )
    with raises_exactly("Existing artifact changed or removed: batch-000000.parquet"):
        module()._extension(extension_receipt(), new)


def test_snapshot_rejects_a_declared_row_count_the_parquet_contradicts(tmp_path, hub):
    value = receipt(tmp_path, artifact_rows=2)
    value["artifacts"][0]["rows"] = 1
    value["processed_rows"] = 1
    write_receipt(tmp_path, value)
    with raises_exactly("Row count mismatch: batch-000000.parquet"):
        publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")
    hub.api.create_commit.assert_not_called()


def test_publication_stages_into_a_hidden_namespaced_directory(tmp_path, hub):
    receipt(tmp_path)
    observed: list[list[str]] = []
    original = hub.api.create_commit.side_effect

    def spy(*args, **kwargs):
        observed.append(sorted(p.name for p in tmp_path.iterdir() if p.is_dir()))
        return original(*args, **kwargs)

    hub.api.create_commit.side_effect = spy
    publisher()(tmp_path, repo_id="a/b", run_id="pilot-1")

    assert len(observed) == 1
    hidden = [name for name in observed[0] if name.startswith(".")]
    assert hidden == [name for name in hidden if name.startswith(".ner-publication-")]
    assert len(hidden) == 1
