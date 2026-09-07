"""Receipt-bound, additive geographic NER publication to an existing dataset."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf.uploader import resolve_hf_token
from osm_polygon_wikidata_only.io.atomic import atomic_copy_file
from osm_polygon_wikidata_only.io.hashing import sha256_file

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _matches(pattern: str, value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _count(value: object) -> int:
    _require(type(value) is int, "Receipt row counts must be integers")
    assert isinstance(value, int)
    _require(value >= 0, "Receipt row counts must be nonnegative")
    return value


def _digest(value: object) -> None:
    _require(_matches(r"[0-9a-fA-F]{64}", value), "Invalid SHA256 digest")


def _contract_digest(contract: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Receipt contract is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _artifact(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "Invalid artifact record")
    _require(_matches(r"batch-[0-9]{6,}\.parquet", value.get("path")), "Unsafe artifact path")
    _digest(value.get("sha256"))
    _count(value.get("rows"))
    return value


def _validate_artifact_order(records: list[dict[str, Any]]) -> None:
    _require(
        [record["path"] for record in records]
        == [f"batch-{index:06d}.parquet" for index in range(len(records))],
        "Artifact order is not contiguous",
    )


def _index_artifacts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {value["path"]: value for value in records}
    _require(len(result) == len(records), "Duplicate artifact paths")
    return result


def _artifacts(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = receipt.get("artifacts")
    _require(isinstance(values, list), "At least one batch is required")
    assert isinstance(values, list)
    _require(bool(values), "At least one batch is required")
    records = [_artifact(value) for value in values]
    _validate_artifact_order(records)
    return _index_artifacts(records)


def _receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), "Receipt must be an object")
    _require(isinstance(value.get("contract"), dict), "Receipt contract must be an object")
    contract_id = value.get("contract_id")
    _digest(contract_id)
    assert isinstance(contract_id, str)
    _require(contract_id.lower() == _contract_digest(value["contract"]), "contract_id mismatch")
    _digest(value.get("source_sha256"))
    _require(value.get("status") in ("completed", "paused"), "Receipt is not publishable")
    processed_rows = _count(value.get("processed_rows"))
    source_rows = _count(value.get("source_rows"))
    batch_size = _count(value.get("batch_size"))
    _require(batch_size > 0, "Batch size must be positive")
    _require(
        processed_rows <= source_rows,
        "Processed rows exceed source rows",
    )
    if value["status"] == "completed":
        _require(processed_rows == source_rows, "completed receipt must include every source row")
    artifacts = _artifacts(value)
    _require(
        sum(artifact["rows"] for artifact in artifacts.values()) == value["processed_rows"],
        "Processed rows differ from completed batch rows",
    )
    return value


def _copy(source: Path, target: Path) -> None:
    _require(not source.is_symlink(), f"Symlink is not a publishable artifact: {source.name}")
    atomic_copy_file(source, target)


def _verify_hash(path: Path, expected: str) -> None:
    _require(sha256_file(path) == expected.lower(), f"SHA256 mismatch: {path.name}")


def _snapshot(output_dir: Path, stage: Path) -> dict[str, Any]:
    _copy(output_dir / "receipt.json", stage / "receipt.json")
    receipt = _receipt(stage / "receipt.json")
    for name, artifact in _artifacts(receipt).items():
        target = stage / name
        _copy(output_dir / name, target)
        _verify_hash(target, artifact["sha256"])
        _require(
            pq.read_metadata(target).num_rows == artifact["rows"], f"Row count mismatch: {name}"
        )
    return receipt


def _extension(old: dict[str, Any], new: dict[str, Any]) -> None:
    for key in ("contract", "contract_id", "source_sha256", "source_rows", "batch_size"):
        _require(old[key] == new[key], f"Run namespace identity differs: {key}")
    _require(old["processed_rows"] <= new["processed_rows"], "Processed rows must not regress")
    if old["status"] == "completed":
        _require(new["status"] == "completed", "completed namespace cannot regress")
    current = _artifacts(new)
    for name, artifact in _artifacts(old).items():
        _require(current.get(name) == artifact, f"Existing artifact changed or removed: {name}")


@dataclass
class _Hub:
    api: HfApi
    repo_id: str
    token: str | None
    downloads: Path

    def files(self, revision: str) -> set[str]:
        return set(self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=revision))

    def download(self, name: str, revision: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo_id,
                name,
                repo_type="dataset",
                revision=revision,
                token=self.token,
                local_dir=self.downloads,
                force_download=True,
            )
        )

    def hash(self, name: str, revision: str) -> str:
        return sha256_file(self.download(name, revision))

    def readme(self, revision: str, files: set[str]) -> str | None:
        if "README.md" not in files:
            return None
        return self.hash("README.md", revision)


def _existing(
    hub: _Hub, parent: str, prefix: str, files: set[str], receipt: dict[str, Any]
) -> dict[str, str]:
    namespace = {name for name in files if name.startswith(prefix)}
    if not namespace:
        return {}
    _require(prefix + "receipt.json" in namespace, "Existing namespace has no receipt")
    path = hub.download(prefix + "receipt.json", parent)
    old = _receipt(path)
    receipt_hash = sha256_file(path)
    _extension(old, receipt)
    expected = {
        prefix + name: artifact["sha256"].lower() for name, artifact in _artifacts(old).items()
    }
    expected[prefix + "receipt.json"] = receipt_hash
    _require(namespace == set(expected), "Existing namespace differs from its receipt")
    _verify_remote(hub, parent, expected)
    return expected


def _verify_remote(hub: _Hub, revision: str, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        _require(hub.hash(name, revision) == digest, f"Remote SHA256 mismatch: {name}")


def _commit(
    hub: _Hub,
    parent: str,
    prefix: str,
    stage: Path,
    expected: dict[str, str],
    existing: dict[str, str],
) -> str:
    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / name.removeprefix(prefix))
        for name, digest in expected.items()
        if existing.get(name) != digest
    ]
    if not operations:
        return parent
    result = hub.api.create_commit(
        hub.repo_id,
        repo_type="dataset",
        revision="main",
        parent_commit=parent,
        operations=operations,
        commit_message=f"Add geographic NER sidecars: {prefix}",
        num_threads=1,
    )
    return result.oid


def _parent_commit(hub: _Hub) -> str:
    parent = hub.api.repo_info(hub.repo_id, repo_type="dataset", revision="main").sha
    _require(isinstance(parent, str), "Dataset has no parent commit")
    assert isinstance(parent, str)
    _require(bool(parent), "Dataset has no parent commit")
    return parent


def _expected_files(prefix: str, stage: Path, receipt: dict[str, Any]) -> dict[str, str]:
    expected = {
        prefix + name: artifact["sha256"].lower() for name, artifact in _artifacts(receipt).items()
    }
    expected[prefix + "receipt.json"] = sha256_file(stage / "receipt.json")
    return expected


def _verify_published(
    hub: _Hub,
    revision: str,
    prefix: str,
    expected: dict[str, str],
    readme: str | None,
) -> None:
    remote_files = hub.files(revision)
    _require(
        {name for name in remote_files if name.startswith(prefix)} == set(expected),
        "Published namespace differs from receipt",
    )
    _verify_remote(hub, revision, expected)
    _require(hub.readme(revision, remote_files) == readme, "README changed during publication")


def _publish(hub: _Hub, prefix: str, stage: Path, receipt: dict[str, Any]) -> str:
    parent = _parent_commit(hub)
    files = hub.files(parent)
    readme = hub.readme(parent, files)
    existing = _existing(hub, parent, prefix, files, receipt)
    expected = _expected_files(prefix, stage, receipt)
    revision = _commit(hub, parent, prefix, stage, expected, existing)
    _verify_published(hub, revision, prefix, expected, readme)
    return revision


def publish_ner_shard(
    output_dir: Path, *, repo_id: str, run_id: str, token: str | None = None
) -> str:
    """Publish validated batches under ``geographic_ner/<run_id>/`` only.

    The existing dataset must have a main commit. Receipt and Parquet bytes are
    snapshotted on the output filesystem, validated before upload, and read back
    at the returned commit with bounded-memory hashes. Identical retries return
    the inspected parent without a commit. Conflicts propagate; callers may retry
    the entire operation to inspect a fresh parent. Hub imports are lazy so tests
    can replace ``HfApi`` and ``hf_hub_download`` without network access.
    """
    _require(_matches(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id), "Unsafe run_id")
    with TemporaryDirectory(prefix=".ner-publication-", dir=output_dir) as temporary:
        stage = Path(temporary)
        receipt = _snapshot(output_dir, stage)
        from huggingface_hub import HfApi

        resolved = resolve_hf_token(token)
        hub = _Hub(HfApi(token=resolved), repo_id, resolved, stage / "remote")
        return _publish(hub, f"geographic_ner/{run_id}/", stage, receipt)
