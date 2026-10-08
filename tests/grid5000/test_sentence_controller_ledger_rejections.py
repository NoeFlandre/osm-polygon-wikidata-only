"""Rejection branches of the Grid5000 sentence-controller ledger lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000 import (
    sentence_controller,
    sentence_controller_ledger,
    sentence_controller_policy,
)
from tests.grid5000.test_sentence_controller import (
    _controller,
    _data_root,
    _FakePublisher,
    _FakeTransport,
)

ControllerRunError = sentence_controller.ControllerRunError


def _fresh(tmp_path: Path, **kwargs: object) -> sentence_controller.Grid5000SentenceController:
    return _controller(_data_root(tmp_path), _FakeTransport(tmp_path), _FakePublisher(), **kwargs)  # ty: ignore[invalid-argument-type]


def _rewrite_ledger(controller: sentence_controller.Grid5000SentenceController, **fields) -> None:
    ledger = json.loads(controller.ledger_path.read_text(encoding="utf-8"))
    ledger.update(fields)
    controller.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")


def _resume(tmp_path: Path, **kwargs: object) -> sentence_controller.Grid5000SentenceController:
    return _controller(DataRoot(tmp_path), _FakeTransport(tmp_path), _FakePublisher(), **kwargs)  # ty: ignore[invalid-argument-type]


def test_initialize_is_idempotent_within_one_controller(tmp_path: Path) -> None:
    controller = _fresh(tmp_path)
    assert controller.initialize() is controller.initialize()


def test_missing_run_id_generates_a_safe_identifier(tmp_path: Path) -> None:
    controller = _fresh(tmp_path, run_id=None)
    ledger = controller.initialize()
    assert isinstance(ledger["run_id"], str)
    assert controller.run_id == ledger["run_id"]


def test_unsafe_run_id_is_rejected_before_writing(tmp_path: Path) -> None:
    controller = _fresh(tmp_path, run_id="../escape")
    with pytest.raises(ControllerRunError, match="Unsafe Grid5000 run_id"):
        controller.initialize()
    assert not controller.ledger_path.exists()


def test_existing_ledger_without_run_id_is_rejected(tmp_path: Path) -> None:
    first = _fresh(tmp_path)
    first.initialize()
    _rewrite_ledger(first, run_id=7)
    with pytest.raises(ControllerRunError, match="no valid run_id"):
        _resume(tmp_path).initialize()


def test_existing_ledger_rejects_a_different_requested_run_id(tmp_path: Path) -> None:
    _fresh(tmp_path).initialize()
    with pytest.raises(ControllerRunError, match="does not match"):
        _resume(tmp_path, run_id="run-other").initialize()


def test_resume_adopts_the_stored_run_id_when_none_is_requested(tmp_path: Path) -> None:
    stored = _fresh(tmp_path).initialize()["run_id"]
    resumed = _resume(tmp_path, run_id=None)
    assert resumed.initialize()["run_id"] == stored
    assert resumed.run_id == stored


def test_non_string_stored_commit_is_not_migrated_and_fails_immutability(
    tmp_path: Path,
) -> None:
    first = _fresh(tmp_path)
    first.initialize()
    _rewrite_ledger(first, source_commit=None)
    with pytest.raises(ControllerRunError, match="immutable field changed: source_commit"):
        _resume(tmp_path).initialize()


def test_source_commit_updates_must_be_a_list(tmp_path: Path) -> None:
    first = _fresh(tmp_path)
    first.initialize()
    _rewrite_ledger(first, source_commit="old123", source_commit_updates={"bad": True})
    with pytest.raises(ControllerRunError, match="source_commit_updates must be a list"):
        _resume(tmp_path).initialize()


def test_changed_immutable_site_is_rejected(tmp_path: Path) -> None:
    first = _fresh(tmp_path)
    first.initialize()
    _rewrite_ledger(first, site="nancy")
    with pytest.raises(ControllerRunError, match="immutable field changed: site"):
        _resume(tmp_path).initialize()


def test_uninitialized_ledger_cannot_be_written(tmp_path: Path) -> None:
    controller = _fresh(tmp_path)
    with pytest.raises(ControllerRunError, match="uninitialized sentence ledger"):
        controller._write_ledger()
    assert not controller.ledger_path.exists()


def test_sentence_ledger_creation_and_immutable_validation() -> None:
    mixin = sentence_controller_ledger.SentenceControllerLedgerMixin
    written: list[object] = []
    controller = SimpleNamespace(
        run_id=None,
        ledger_path=Path("missing-ledger.json"),
        _ledger=None,
        _create_ledger=lambda: cast(Any, mixin._create_ledger)(controller),
        _new_ledger=lambda: cast(
            sentence_controller_policy.LedgerDict, {"run_id": controller.run_id}
        ),
        _write_ledger=lambda ledger=None: written.append(ledger),
    )
    ledger = cast(Any, mixin.initialize)(controller)
    assert ledger["run_id"] == controller.run_id
    assert written == [ledger]

    controller.run_id = "../unsafe"
    with pytest.raises(
        sentence_controller_policy.ControllerRunError, match="Unsafe Grid5000 run_id"
    ):
        cast(Any, mixin._create_ledger)(controller)

    validator = SimpleNamespace(_immutable_ledger_fields=lambda: {"repo_id": "expected"})
    with pytest.raises(sentence_controller_policy.ControllerRunError, match="repo_id"):
        cast(Any, mixin._validate_immutable_ledger)(validator, {"repo_id": "other"})
