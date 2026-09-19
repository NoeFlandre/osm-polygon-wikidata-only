"""Hugging Face publication and byte-level verification for sentence batches."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from osm_polygon_wikidata_only.hf.remote_inventory import RemoteFileInfo, RemoteInventory
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.config import V2_ADDED_WIKIPEDIA_TAG_MAP_PATH
from osm_polygon_wikidata_only.v2.publication import sentence_publication_ops

from .sentence_controller_policy import ControllerRunError

_MAX_SENTENCE_UPLOAD_THREADS = 4


class HubPublisher(Protocol):
    """Local Hugging Face publication and verification boundary."""

    def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str:
        """Publish one atomic sentence batch and return its commit reference."""

    def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None:
        """Verify the uploaded sentence files and protected card assets."""


def download_hf_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None,
    local_dir: Path,
) -> Path:
    try:
        # Keep Hub verification optional for local controller construction.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - package is a runtime dependency
        raise ControllerRunError("huggingface_hub is required for HF verification") from error
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir),
        )
    )


class HfHubSentencePublisher:
    """Publish sentence artifacts locally and verify exact remote bytes."""

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None,
        cache_dir: Path,
        upload_function: Callable[..., str] = upload_files,
        download_function: Callable[..., Path] = download_hf_file,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.cache_dir = Path(cache_dir)
        self._upload_function = upload_function
        self._download_function = download_function

    def publish_sentence_batch(self, processed_v2: Path, stems: Sequence[str], message: str) -> str:
        operations = sentence_publication_ops(processed_v2, stems)
        return self._upload_function(
            self.repo_id,
            ops=operations,
            token=self.token,
            commit_message=message,
            num_threads=min(_MAX_SENTENCE_UPLOAD_THREADS, len(operations)),
        )

    def verify_sentence_batch(self, processed_v2: Path, stems: Sequence[str]) -> None:
        expected = expected_sentence_files(processed_v2, stems)
        validate_expected_sentence_files(expected)
        inventory = RemoteInventory.fetch_paths(
            self.repo_id,
            paths=[remote for _, remote in expected],
            token=self.token,
        )
        missing = missing_remote_files(expected, inventory)
        if missing:
            raise ControllerRunError(f"HF sentence publication is missing files: {missing}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        verify_expected_sentence_files(
            self.repo_id,
            expected,
            inventory=inventory,
            token=self.token,
            cache_dir=self.cache_dir,
            download_function=self._download_function,
        )


def expected_sentence_files(
    processed_v2: Path,
    stems: Sequence[str],
) -> list[tuple[Path | None, str]]:
    operations = sentence_publication_ops(processed_v2, stems)
    expected = [(operation.local_path, operation.path_in_repo) for operation in operations]
    expected.append(
        (processed_v2 / V2_ADDED_WIKIPEDIA_TAG_MAP_PATH, V2_ADDED_WIKIPEDIA_TAG_MAP_PATH)
    )
    return expected


def validate_expected_sentence_files(expected: Sequence[tuple[Path | None, str]]) -> None:
    if any(local is None for local, _ in expected):
        raise ControllerRunError("HF verification received an incomplete publication plan")


def missing_remote_files(
    expected: Sequence[tuple[Path | None, str]],
    inventory: RemoteInventory,
) -> list[str]:
    return [remote for _, remote in expected if not inventory.contains(remote)]


def verify_expected_sentence_files(
    repo_id: str,
    expected: Sequence[tuple[Path | None, str]],
    *,
    inventory: RemoteInventory,
    token: str | None,
    cache_dir: Path,
    download_function: Callable[..., Path] = download_hf_file,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hf-sentence-verify-", dir=cache_dir) as temporary:
        for local, remote in expected:
            assert local is not None
            verify_sentence_file(
                repo_id,
                local,
                remote,
                inventory=inventory,
                token=token,
                local_dir=Path(temporary),
                download_function=download_function,
            )


def verify_sentence_file(
    repo_id: str,
    local: Path,
    remote: str,
    *,
    inventory: RemoteInventory,
    token: str | None,
    local_dir: Path,
    download_function: Callable[..., Path] = download_hf_file,
) -> None:
    info = inventory.metadata(remote)
    if info is not None and info.sha256 is not None:
        verify_known_sentence_file(local, remote, info)
        return
    downloaded = download_function(repo_id, remote, token=token, local_dir=local_dir)
    if sha256_file(downloaded) != sha256_file(local):
        raise ControllerRunError(f"HF sentence publication hash mismatch: {remote}")


def verify_known_sentence_file(local: Path, remote: str, info: RemoteFileInfo) -> None:
    if info.size != local.stat().st_size or info.sha256 != sha256_file(local):
        raise ControllerRunError(f"HF sentence publication hash mismatch: {remote}")


__all__ = [
    "HfHubSentencePublisher",
    "HubPublisher",
    "download_hf_file",
    "expected_sentence_files",
    "missing_remote_files",
    "validate_expected_sentence_files",
    "verify_expected_sentence_files",
    "verify_known_sentence_file",
    "verify_sentence_file",
]
