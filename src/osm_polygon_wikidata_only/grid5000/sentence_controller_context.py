"""Typed collaboration context shared by the Grid5000 controller mixins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from osm_polygon_wikidata_only.config.paths import DataRoot

from .sentence_controller_policy import ControllerLimits
from .sentence_publication import HubPublisher
from .sentence_transport import Grid5000Transport


class SentenceControllerContext(Protocol):
    """Attributes and phase methods supplied by the concrete controller.

    The implementation is deliberately split into mixins, so a normal class
    hierarchy cannot express the cooperative ``self`` contract to static type
    checkers. This protocol records that contract without changing runtime MRO
    or the public controller API.
    """

    data_root: DataRoot
    site: str
    queue: str
    gpu_model: str
    repo_id: str
    transport: Grid5000Transport
    publisher: HubPublisher
    run_id: str | None
    repo_root: Path
    source_commit: str
    limits: ControllerLimits
    _sleep: Callable[[float], None]
    poll_interval_s: float
    _ledger: dict[str, Any] | None
    _current_batch: dict[str, Any] | None
    ledger_path: Path
    remote_run_root: str

    def initialize(self) -> dict[str, Any]: ...

    def _process_batch(self, batch: dict[str, Any]) -> None: ...

    def _publish_batch(self, batch: dict[str, Any]) -> None: ...

    def _publish_if_ready(self, batch: dict[str, Any]) -> None: ...

    def _submit_batch(self, batch: dict[str, Any]) -> None: ...

    def _ensure_remote_namespace(self) -> None: ...

    def _stage_batch(self, staging: Path, batch: Mapping[str, object]) -> None: ...

    def _stage_checkpoint_trees(self, staged_data: Path, stem: str) -> None: ...

    def _remote_job_command(self, batch: Mapping[str, object]) -> str: ...

    def _reconcile_batch(self, batch: dict[str, Any]) -> None: ...

    def _retrieve_batch(
        self,
        batch: dict[str, Any],
        *,
        state: str,
        exit_code: int | None,
    ) -> None: ...

    def _read_valid_receipt(
        self,
        batch: dict[str, Any],
        received: Path,
        received_data: DataRoot,
    ) -> dict[str, Any]: ...

    def _mark_batch_failed(
        self,
        batch: dict[str, Any],
        received_data: DataRoot,
        receipt: Mapping[str, object],
    ) -> None: ...

    def _import_success(
        self,
        batch: dict[str, Any],
        received_data: DataRoot,
        receipt: Mapping[str, object],
    ) -> None: ...

    def _cleanup_remote_job(self, batch: dict[str, Any]) -> None: ...

    def _assert_baseline(self) -> None: ...

    def _policy_check(self) -> None: ...

    def _run_frontend(self, args: tuple[str, ...], *, allow_failure: bool = False) -> Any: ...

    def _finalize_run(self) -> None: ...

    def _handle_interrupt(self) -> None: ...

    def _load_existing_ledger(self) -> dict[str, Any]: ...

    def _create_ledger(self) -> dict[str, Any]: ...

    def _refresh_resumable_source_commit(self, ledger: dict[str, Any]) -> None: ...

    def _new_ledger(self) -> dict[str, Any]: ...

    def _new_batch_record(self, batch: Any) -> dict[str, Any]: ...

    def _validate_immutable_ledger(self, ledger: Mapping[str, object]) -> None: ...

    def _immutable_ledger_fields(self) -> dict[str, object]: ...

    def _write_ledger(self, ledger: dict[str, Any] | None = None) -> None: ...

    def _validate_receipt(
        self, batch: Mapping[str, object], receipt: Mapping[str, object]
    ) -> None: ...

    def _import_partial(self, batch: Mapping[str, object], received_data: DataRoot) -> None: ...

    def _import_checkpoints(self, batch: Mapping[str, object], received_data: DataRoot) -> None: ...

    def _import_checkpoint(self, stem: str, project: str, received_data: DataRoot) -> None: ...
