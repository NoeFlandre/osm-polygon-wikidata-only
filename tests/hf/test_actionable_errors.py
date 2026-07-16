"""Actionable error sanitization tests.

The unified sync path emits actionable errors at five boundaries:

* Remote inventory / authentication failure
* Malformed local manifests
* Missing or schema-invalid core artifacts
* Publication assembly failure
* Background upload failure

All errors must:

* Include the affected stem and a corrective action.
* Never leak tokens, hashes, full user-home paths, or request payloads.
* Preserve exception chaining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.publication import (
    PublicationValidationError,
    load_existing_core_artifacts,
)
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory


def _seed_polygons_links_manifest(
    data_root: DataRoot,
    stem: str,
    *,
    manifest_data: dict[str, Any] | None = None,
) -> None:
    poly_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "lat": 1.0, "lon": 2.0}], schema=polygon_schema()
    )
    pq.write_table(poly_table, data_root.processed_polygons / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    links_table = pa.Table.from_pylist(
        [{"polygon_id": "1", "article_id": "a1"}], schema=polygon_article_schema()
    )
    pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]
    if manifest_data is not None:
        (data_root.processed_manifests / "processed_pbfs.json").write_text(
            json.dumps(manifest_data)
        )


def test_load_existing_core_artifacts_manifest_error_includes_stem(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"
    _seed_polygons_links_manifest(
        data_root,
        stem,
        manifest_data={
            f"{stem}.osm.pbf": {
                "source_pbf": "wrong-source.osm.pbf",
                "polygons_path": f"polygons/{stem}.parquet",
                "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            }
        },
    )
    with pytest.raises(PublicationValidationError) as excinfo:
        load_existing_core_artifacts(data_root, stem)
    message = str(excinfo.value)
    # Must include the affected stem
    assert stem in message or "mexico-latest" in message
    # Must include a corrective action phrase
    assert any(hint in message.lower() for hint in ("expected", "rebuild", "rerun"))


def test_load_existing_core_artifacts_schema_error_includes_stem(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    stem = "mexico-latest"
    _seed_polygons_links_manifest(
        data_root,
        stem,
        manifest_data={
            f"{stem}.osm.pbf": {
                "source_pbf": f"{stem}.osm.pbf",
                "polygons_path": f"polygons/{stem}.parquet",
                "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            }
        },
    )
    # Corrupt the polygons parquet so schema validation fails
    (data_root.processed_polygons / f"{stem}.parquet").write_bytes(b"not-parquet-data")
    with pytest.raises(PublicationValidationError) as excinfo:
        load_existing_core_artifacts(data_root, stem)
    message = str(excinfo.value)
    assert stem in message or "mexico-latest" in message


def test_remote_inventory_failure_raises_sanitized_error() -> None:
    """RemoteInventory.fetch failures must be raised as a project-
    specific exception with the repository id and corrective action,
    never leaking the underlying network payload."""

    class FailingHub(StubHfHub):
        def list_repo_files(self, repo_id: str, *, repo_type: str = "dataset") -> list[str]:
            # Simulate HF 401 with a body that includes sensitive-looking content
            raise RuntimeError(
                "401 Unauthorized: token=hf_secretXYZ; request_id=abc-123; "
                "body={u'error': u'invalid user token'}"
            )

    with pytest.raises(Exception) as excinfo:
        RemoteInventory.fetch(repo_id="user/secret-repo", hub=FailingHub())

    message = str(excinfo.value)
    # Must mention the repository so the user knows which one failed
    assert "user/secret-repo" in message
    # Must NOT leak token, request_id, or body content
    assert "hf_secretXYZ" not in message
    assert "abc-123" not in message
    assert "invalid user token" not in message or "auth" in message.lower()
    # Must include a corrective hint
    assert any(
        hint in message.lower() for hint in ("verify", "token", "auth", "permission", "credential")
    )


def test_remote_inventory_failure_preserves_chaining() -> None:
    """The translated error must chain the original exception."""

    class FailingHub(StubHfHub):
        def list_repo_files(self, repo_id: str, *, repo_type: str = "dataset") -> list[str]:
            raise ConnectionError("connection refused")

    with pytest.raises(Exception) as excinfo:
        RemoteInventory.fetch(repo_id="user/repo", hub=FailingHub())
    assert excinfo.value.__cause__ is not None


def test_remote_inventory_failure_message_does_not_contain_user_paths() -> None:
    """The translated RemoteInventory error must never embed a
    full filesystem path or user-home directory."""

    class FailingHub(StubHfHub):
        def list_repo_files(self, repo_id: str, *, repo_type: str = "dataset") -> list[str]:
            raise RuntimeError(
                "OSError: [Errno 2] No such file or directory: '/Users/alice/private/x.json'"
            )

    with pytest.raises(Exception) as excinfo:
        RemoteInventory.fetch(repo_id="user/repo", hub=FailingHub())
    message = str(excinfo.value)
    assert "/Users/" not in message
    assert "private" not in message
