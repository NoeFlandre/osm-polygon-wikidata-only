from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import osm_polygon_wikidata_only.hf._upload_state as state
from osm_polygon_wikidata_only.hf._upload_state import UploadStateStore
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp


def _add_op(path: Path, *, remote: str = "data.parquet") -> PublicationOp:
    return PublicationOp(action="add", path_in_repo=remote, local_path=path)


def _delete_op(*, remote: str = "data.parquet") -> PublicationOp:
    return PublicationOp(action="delete", path_in_repo=remote)


def test_persist_creates_an_immutable_snapshot_and_current_envelope(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"payload")

    stored = UploadStateStore(state_dir).persist(
        [PublicationOp(action="add", path_in_repo="data.parquet", local_path=canonical)],
        "commit message",
    )

    assert stored.state_path == state_dir / "000001.json"
    assert stored.snapshot_dir == state_dir / "snapshots" / "000001"
    assert stored.ops[0].snapshot_path is not None
    assert stored.ops[0].snapshot_path.read_bytes() == b"payload"
    assert stored.ops[0].snapshot_path.stat().st_ino != canonical.stat().st_ino
    assert stored.op_shas == (hashlib.sha256(b"payload").hexdigest(),)
    assert json.loads(stored.state_path.read_text()) == {
        "contract_version": "bg-upload-v1",
        "message": "commit message",
        "ops": [
            {
                "action": "add",
                "local_path": str(canonical),
                "path_in_repo": "data.parquet",
                "sha256": hashlib.sha256(b"payload").hexdigest(),
                "snapshot_path": str(stored.ops[0].snapshot_path),
            }
        ],
        "sequence": 1,
    }


def test_persist_preserves_message_and_canonical_json_format(tmp_path: Path) -> None:
    stored = UploadStateStore(tmp_path / "state").persist(
        [_delete_op()],
        "commit message",
    )

    assert stored.message == "commit message"
    envelope = {
        "contract_version": "bg-upload-v1",
        "sequence": 1,
        "message": "commit message",
        "ops": [
            {
                "action": "delete",
                "path_in_repo": "data.parquet",
                "local_path": None,
            }
        ],
    }
    assert stored.state_path.read_text(encoding="utf-8") == (
        json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    )


def test_highwater_accepts_zero_rejects_negative_and_reads_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    highwater = tmp_path / ".highwater"
    highwater.write_text("0", encoding="utf-8")
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert state._read_highwater(tmp_path) == 0
    assert encodings == ["utf-8"]

    highwater.write_text("-1", encoding="utf-8")
    assert state._read_highwater(tmp_path) == 0


def test_scanned_sequences_only_include_json_and_support_envelope_sequences(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    numbered = state_dir / "000007.json"
    numbered.write_text("{}", encoding="utf-8")
    (state_dir / "000009.JSON").write_text("{}", encoding="utf-8")
    legacy_named = state_dir / "legacy-id.json"
    legacy_named.write_text(
        json.dumps({"contract_version": "bg-upload-v1", "sequence": 11}),
        encoding="utf-8",
    )

    assert state._sequence_from_state_path(numbered) == 7
    assert state._sequence_from_state_path(legacy_named) == 11
    assert state._scanned_sequence(state_dir) == 11


@pytest.mark.parametrize("sequence", [0, -1, True, "1"])
def test_sequence_from_non_numeric_envelope_returns_zero(tmp_path: Path, sequence: object) -> None:
    path = tmp_path / "legacy-id.json"
    path.write_text(
        json.dumps({"contract_version": "bg-upload-v1", "sequence": sequence}),
        encoding="utf-8",
    )

    assert state._sequence_from_state_path(path) == 0


def test_sequence_from_envelope_accepts_sequence_one(tmp_path: Path) -> None:
    path = tmp_path / "legacy-id.json"
    path.write_text(
        json.dumps({"contract_version": "bg-upload-v1", "sequence": 1}),
        encoding="utf-8",
    )

    assert state._sequence_from_state_path(path) == 1


def test_sha256_reads_fixed_size_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"ignored")
    reads: list[int | None] = []

    class Reader:
        def __enter__(self) -> Reader:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int | None) -> bytes:
            reads.append(size)
            return b"chunk" if len(reads) == 1 else b""

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Reader())
    assert state._sha256_file(payload) == hashlib.sha256(b"chunk").hexdigest()
    assert reads == [65536, 65536]


def test_independent_copy_creates_missing_parent_without_sharing_inode(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "nested" / "deeper" / "target.bin"
    source.write_bytes(b"payload")

    state._independent_copy(source, target)

    assert target.read_bytes() == b"payload"
    assert target.stat().st_ino != source.stat().st_ino


def test_json_reader_requires_utf8_and_rejects_non_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[]", encoding="utf-8")
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert state._read_json_object(path) is None
    assert encodings == ["utf-8"]


@pytest.mark.parametrize("sequence", [None, "1", 0, -1, True])
def test_current_sequence_rejects_invalid_values(tmp_path: Path, sequence: object) -> None:
    with pytest.raises(ValueError, match="invalid sequence"):
        state._current_sequence({"sequence": sequence}, tmp_path / "pending.json")


def test_current_sequence_accepts_positive_integer(tmp_path: Path) -> None:
    assert state._current_sequence({"sequence": 1}, tmp_path / "pending.json") == 1


def test_legacy_envelope_requires_message_and_ops_and_excludes_current_contract() -> None:
    assert state._is_legacy_envelope({"message": "m", "ops": []})
    assert not state._is_legacy_envelope({"message": "m"})
    assert not state._is_legacy_envelope({"ops": []})
    assert not state._is_legacy_envelope(
        {"contract_version": "bg-upload-v1", "message": "m", "ops": []}
    )


def test_path_containment_uses_non_strict_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    child = parent / "missing" / "child.bin"
    calls: list[bool | None] = []
    original_resolve = Path.resolve

    def resolve(path: Path, *, strict: bool = False) -> Path:
        if path in {child, parent}:
            calls.append(strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert state._is_inside(child, parent)
    assert calls == [False, False]


def test_remove_failed_upgrade_is_best_effort_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    snapshot_root = state_dir / "snapshots"
    snapshot_dir = snapshot_root / "000001"
    snapshot_dir.mkdir(parents=True)
    (state_dir / "000001.json").write_text("{}", encoding="utf-8")
    removed: list[tuple[Path, bool]] = []

    def rmtree(path: Path, *, ignore_errors: bool) -> None:
        removed.append((path, ignore_errors))

    monkeypatch.setattr(state.shutil, "rmtree", rmtree)
    state._remove_failed_upgrade(state_dir, 1, snapshot_dir)
    assert removed == [(snapshot_dir, True)]

    removed.clear()
    state._remove_failed_upgrade(state_dir, 1, snapshot_root / "missing")
    assert removed == []

    # A missing envelope is a normal recovery case.
    state._remove_failed_upgrade(state_dir, 1, snapshot_dir)


def test_cleanup_failed_submission_is_idempotent_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = UploadStateStore(tmp_path / "state")
    snapshot_dir = store.state_dir / "snapshots" / "000001"
    snapshot_dir.mkdir(parents=True)
    calls: list[tuple[Path, bool]] = []

    def rmtree(path: Path, *, ignore_errors: bool) -> None:
        calls.append((path, ignore_errors))

    monkeypatch.setattr(state.shutil, "rmtree", rmtree)
    store.cleanup_failed_submission(store.state_dir / "missing.json", snapshot_dir)
    store.cleanup_failed_submission(store.state_dir / "missing.json", None)

    assert calls == [(snapshot_dir, True)]


def test_store_creates_nested_state_dir_and_allocates_consecutive_sequences(
    tmp_path: Path,
) -> None:
    store = UploadStateStore(tmp_path / "one" / "two" / "state")

    assert store._allocate_sequence() == 1
    assert store._allocate_sequence() == 2
    assert (store.state_dir / ".highwater").read_text(encoding="utf-8") == "2\n"


def test_submission_snapshot_dir_is_idempotent_for_adds(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"payload")
    op = _add_op(canonical)

    first = store._submission_snapshot_dir([op], 1)
    second = store._submission_snapshot_dir([op], 1)

    assert first == second == tmp_path / "state" / "snapshots" / "000001"


def test_snapshot_operation_rejects_an_add_without_a_local_path_from_legacy_input(
    tmp_path: Path,
) -> None:
    store = UploadStateStore(tmp_path / "state")

    with pytest.raises(ValueError, match="requires a local_path"):
        store._upgrade_legacy_operation(
            {"action": "add", "path_in_repo": "data.parquet"},
            0,
            tmp_path / "state" / "snapshots" / "000001",
        )


def test_snapshot_add_operation_creates_its_parent_and_preserves_operation_fields(
    tmp_path: Path,
) -> None:
    store = UploadStateStore(tmp_path / "state")
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"payload")
    snapshot_dir = tmp_path / "new" / "snapshot"
    operation, entry = store._snapshot_add_operation(
        _add_op(canonical, remote="remote/data.bin"),
        3,
        snapshot_dir,
        {"action": "add", "path_in_repo": "remote/data.bin"},
    )
    repeated_operation, repeated_entry = store._snapshot_add_operation(
        _add_op(canonical, remote="remote/data.bin"),
        3,
        snapshot_dir,
        {"action": "add", "path_in_repo": "remote/data.bin"},
    )

    assert operation.path_in_repo == "remote/data.bin"
    assert operation.local_path == canonical
    assert operation.snapshot_path == snapshot_dir / "003" / "canonical.bin"
    assert entry["sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert repeated_operation == operation
    assert repeated_entry == entry


def test_resume_helpers_preserve_defaults_and_error_context(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    state_path = store.state_dir / "000001.json"
    snapshot_dir = store.state_dir / "snapshots" / "000001"

    assert store._resume_operations({}, state_path, snapshot_dir) == ([], ())
    with pytest.raises(ValueError, match=rf"Add op in {state_path}.*missing snapshot_path"):
        store._resume_operations(
            {"ops": [{"action": "add", "path_in_repo": "data.parquet"}]},
            state_path,
            snapshot_dir,
        )
    with pytest.raises(ValueError, match=r"Add op in .*000001\.json.*missing snapshot_path"):
        store._resume_operation(
            {"action": "add", "path_in_repo": "data.parquet"},
            state_path,
            snapshot_dir,
        )


def test_classify_current_envelope_reports_path_and_returns_current_tag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pending.json"
    payload = {"contract_version": "bg-upload-v1", "sequence": 1}
    seen: set[int] = set()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert state.UploadStateStore._classify_pending_envelope(path, seen) == (
        "current",
        payload,
        1,
    )
    with pytest.raises(ValueError, match=r"Duplicate sequence 1 in .*pending\.json"):
        state.UploadStateStore._classify_pending_envelope(path, seen)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"contract_version": "bg-upload-v1", "sequence": 0}),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=rf"Current envelope {invalid_path} has an invalid sequence",
    ):
        state.UploadStateStore._classify_pending_envelope(invalid_path, set())


def test_resume_orders_current_and_upgraded_jobs_by_sequence(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    current = [
        (2, Path("two.json"), {"sequence": 2}),
        (1, Path("one.json"), {"sequence": 1}),
    ]
    upgraded = [(3, Path("three.json"), {"sequence": 3}, Path("snapshots/3"))]

    jobs = store._ordered_pending_jobs(current, upgraded)
    duplicate_sequence = store._ordered_pending_jobs(
        [
            (1, Path("b.json"), {"sequence": 1}),
            (1, Path("a.json"), {"sequence": 1}),
        ],
        [],
    )

    assert [job[0] for job in jobs] == [1, 2, 3]
    assert [job[1] for job in duplicate_sequence] == [Path("b.json"), Path("a.json")]


def test_restore_pending_job_includes_state_path_in_validation_failure(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    state_path = store.state_dir / "000001.json"
    restored, failure = store._restore_pending_job(
        state_path,
        {"message": "m", "ops": [{"action": "add", "path_in_repo": "data"}]},
        store.state_dir / "snapshots" / "000001",
    )

    assert restored is None
    assert failure == (
        f"resume validation failed for {state_path.name}: "
        f"Add op in {state_path} is missing snapshot_path"
    )


def test_legacy_upgrade_persists_exact_current_envelope_and_snapshot(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"legacy")
    legacy_path = state_dir / "legacy-id.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps(
            {
                "message": "legacy message",
                "ops": [
                    {
                        "action": "add",
                        "path_in_repo": "remote/data.bin",
                        "local_path": str(canonical),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = UploadStateStore(state_dir)
    (state_dir / "snapshots" / "000001" / "000").mkdir(parents=True)

    sequence, envelope, snapshot_dir = store._upgrade_legacy_envelope(
        legacy_path,
        json.loads(legacy_path.read_text(encoding="utf-8")),
    )

    assert sequence == 1
    assert envelope["contract_version"] == "bg-upload-v1"
    assert envelope["sequence"] == 1
    assert envelope["message"] == "legacy message"
    assert envelope["ops"][0]["path_in_repo"] == "remote/data.bin"
    assert envelope["ops"][0]["sha256"] == hashlib.sha256(b"legacy").hexdigest()
    upgraded_path = state_dir / "000001.json"
    assert upgraded_path.read_text(encoding="utf-8") == (
        json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    )
    assert snapshot_dir.is_dir()
    assert not legacy_path.exists()


def test_legacy_upgrade_defaults_missing_message_to_empty_string(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    legacy_path = state_dir / "legacy-id.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps({"ops": []}), encoding="utf-8")
    store = UploadStateStore(state_dir)

    _sequence, envelope, _snapshot_dir = store._upgrade_legacy_envelope(
        legacy_path,
        {"ops": []},
    )

    assert envelope["message"] == ""


def test_legacy_upgrade_tolerates_a_missing_legacy_path(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    legacy_path = state_dir / "already-removed.json"
    store = UploadStateStore(state_dir)

    _sequence, envelope, _snapshot_dir = store._upgrade_legacy_envelope(
        legacy_path,
        {"message": "m", "ops": []},
    )

    assert envelope["message"] == "m"


def test_upgrade_legacy_operations_returns_reconstructed_delete_and_add_ops(
    tmp_path: Path,
) -> None:
    store = UploadStateStore(tmp_path / "state")
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"legacy")
    snapshot_dir = store.state_dir / "snapshots" / "000001"
    operations, entries = store._upgrade_legacy_operations(
        {
            "ops": [
                {"action": "delete", "path_in_repo": "old"},
                {
                    "action": "add",
                    "path_in_repo": "new",
                    "local_path": str(canonical),
                },
            ]
        },
        snapshot_dir,
    )

    assert [operation.action for operation in operations] == ["delete", "add"]
    assert operations[0].path_in_repo == "old"
    assert operations[1].path_in_repo == "new"
    assert operations[1].local_path == canonical
    assert operations[1].snapshot_path is not None
    assert entries[0] == {"action": "delete", "path_in_repo": "old", "local_path": None}
    assert entries[1]["sha256"] == hashlib.sha256(b"legacy").hexdigest()


def test_upgrade_legacy_operations_defaults_to_empty_ops(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    assert store._upgrade_legacy_operations({}, tmp_path / "snapshots") == ([], [])


def test_upgrade_legacy_operation_reports_missing_canonical_file(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")
    missing = tmp_path / "missing.bin"

    with pytest.raises(
        FileNotFoundError,
        match=rf"Cannot snapshot missing canonical file: {missing}",
    ):
        store._upgrade_legacy_operation(
            {
                "action": "add",
                "path_in_repo": "remote/data.bin",
                "local_path": str(missing),
            },
            0,
            store.state_dir / "snapshots" / "000001",
        )


def test_upgrade_legacy_operation_rejects_delete_with_local_path(tmp_path: Path) -> None:
    store = UploadStateStore(tmp_path / "state")

    with pytest.raises(ValueError, match="must not carry a local_path"):
        store._upgrade_legacy_operation(
            {
                "action": "delete",
                "path_in_repo": "remote/data.bin",
                "local_path": str(tmp_path / "unexpected.bin"),
            },
            0,
            store.state_dir / "snapshots" / "000001",
        )


def test_resume_pending_continues_after_one_legacy_upgrade_failure(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    invalid = state_dir / "a-invalid.json"
    valid = state_dir / "b-valid.json"
    invalid.write_text(
        json.dumps(
            {
                "message": "invalid",
                "ops": [
                    {
                        "action": "add",
                        "path_in_repo": "missing",
                        "local_path": str(tmp_path / "missing.bin"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    valid.write_text(
        json.dumps({"message": "valid", "ops": [{"action": "delete", "path_in_repo": "old"}]}),
        encoding="utf-8",
    )
    uploaded: list[str] = []
    store = UploadStateStore(state_dir)
    result = store.resume_pending()
    uploaded.extend(upload.message for upload in result.uploads)

    assert uploaded == ["valid"]
    assert len(result.failures) == 1


def test_legacy_upgrade_duplicate_sequence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = UploadStateStore(tmp_path / "state")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    snapshot = store.state_dir / "snapshots" / "000001"
    payload = {"message": "m", "ops": []}
    results = iter(
        [
            (1, {"contract_version": "bg-upload-v1", "sequence": 1}, snapshot),
            (1, {"contract_version": "bg-upload-v1", "sequence": 1}, snapshot),
        ]
    )
    monkeypatch.setattr(store, "_upgrade_legacy_envelope", lambda *_args: next(results))

    with pytest.raises(ValueError, match="duplicate sequence 1"):
        store._upgrade_pending_legacy(
            [(first, payload), (second, payload)],
            set(),
            [],
        )


def test_delete_is_idempotent_and_ignores_snapshot_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = UploadStateStore(tmp_path / "state")
    state_path = store.state_dir / "missing.json"
    snapshot_dir = store.state_dir / "snapshots" / "000001"
    snapshot_dir.mkdir(parents=True)
    calls: list[tuple[Path, bool]] = []

    def rmtree(path: Path, *, ignore_errors: bool) -> None:
        calls.append((path, ignore_errors))

    monkeypatch.setattr(state.shutil, "rmtree", rmtree)
    store.delete(state_path, snapshot_dir)
    store.delete(state_path, None)

    assert calls == [(snapshot_dir, True)]


def test_snapshot_mismatch_does_not_hash_when_no_sha_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"payload")
    store = UploadStateStore(tmp_path / "state")

    def unexpected_hash(_path: Path) -> str:
        raise AssertionError("an unrecorded snapshot must not be hashed")

    monkeypatch.setattr(store, "_hash_file", unexpected_hash)
    assert store.snapshot_mismatch("message", _add_op(canonical), "") is None
