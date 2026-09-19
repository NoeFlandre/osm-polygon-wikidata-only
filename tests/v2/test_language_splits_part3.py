"""Split coverage tests (part 3)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.v2.language_splits_support import *


def test_v2_install_staged_files_requires_explicit_final_path_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SortPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def as_posix(self) -> str:
            return self.value

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, SortPath) and self.value == other.value

    final_z = SortPath("language_splits/z.parquet")
    final_a = SortPath("language_splits/a.parquet")
    stale = SortPath("language_splits/stale.parquet")
    staged = {final_z: SortPath("stage/z"), final_a: SortPath("stage/a")}
    observed: dict[str, object] = {}

    def fake_previous(root_arg: Path, destination_arg: Path) -> set[SortPath]:
        observed["previous"] = (root_arg, destination_arg)
        return {stale}

    monkeypatch.setattr(language_splits, "_previous_partition_paths", fake_previous)

    def fake_backup(targets: list[SortPath], backups: dict[SortPath, SortPath]) -> None:
        observed["backup"] = targets

    def fake_install(
        staged_arg: dict[SortPath, SortPath],
        installed: list[SortPath],
        *,
        manifest_path: Path,
    ) -> None:
        observed["install"] = list(staged_arg)

    def fake_cleanup(staged_arg: object, backups: object) -> None:
        observed["cleanup"] = True

    def fake_remove(destination: Path, owned_paths: set[SortPath]) -> None:
        observed["owned"] = owned_paths

    monkeypatch.setattr(language_splits, "_backup_targets", fake_backup)
    monkeypatch.setattr(language_splits, "_install_files", fake_install)
    monkeypatch.setattr(language_splits, "_cleanup_transaction", fake_cleanup)
    monkeypatch.setattr(language_splits, "_remove_empty_output_directories", fake_remove)

    language_splits._install_staged_files(tmp_path, tmp_path / "language_splits", staged)

    assert observed["previous"] == (tmp_path, tmp_path / "language_splits")
    assert observed["backup"] == [final_a, stale, final_z]
    assert observed["install"] == [final_z, final_a]
    assert observed["owned"] == {final_a, final_z, stale}
    assert observed["cleanup"] is True


def test_v2_restore_files_is_missing_safe_and_sorts_backups_by_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_final = tmp_path / "restored/missing.parquet"
    final_z = tmp_path / "restored/z.parquet"
    final_a = tmp_path / "restored/a.parquet"
    backup_z = tmp_path / "backups/0-for-z.backup"
    backup_a = tmp_path / "backups/9-for-a.backup"
    backup_z.parent.mkdir()
    backup_z.write_bytes(b"z")
    backup_a.write_bytes(b"a")
    replacements: list[tuple[Path, Path]] = []
    original_replace = language_splits.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(language_splits.os, "replace", recording_replace)

    language_splits._restore_files(
        [missing_final],
        {final_z: backup_z, final_a: backup_a},
    )

    assert replacements == [(backup_a, final_a), (backup_z, final_z)]
    assert final_a.read_bytes() == b"a"
    assert final_z.read_bytes() == b"z"
    assert not missing_final.exists()


def test_v2_restore_files_requires_final_path_order_and_recursive_parent_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RestorePath:
        def __init__(self, value: str, events: list[object]) -> None:
            self.value = value
            self.events = events

        @property
        def parent(self) -> RestorePath:
            return self

        def as_posix(self) -> str:
            return self.value

        def unlink(self, *, missing_ok: bool) -> None:
            self.events.append(("unlink", self.value, missing_ok))

        def mkdir(self, *, parents: bool, exist_ok: bool) -> None:
            self.events.append(("mkdir", self.value, parents, exist_ok))

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, RestorePath) and self.value == other.value

    events: list[object] = []
    missing = RestorePath("restored/missing.parquet", events)
    final_z = RestorePath("restored/z.parquet", events)
    final_a = RestorePath("restored/a.parquet", events)
    backup_z = RestorePath("backup/0-for-z", events)
    backup_a = RestorePath("backup/9-for-a", events)
    replacements: list[tuple[RestorePath, RestorePath]] = []

    class ExistingBackup(RestorePath):
        def exists(self) -> bool:
            return True

    backup_z_existing = ExistingBackup(backup_z.value, events)
    backup_a_existing = ExistingBackup(backup_a.value, events)

    monkeypatch.setattr(
        language_splits.os,
        "replace",
        lambda source, destination: replacements.append((source, destination)),
    )

    language_splits._restore_files(
        [missing],
        {final_z: backup_z_existing, final_a: backup_a_existing},
    )

    assert events == [
        ("unlink", "restored/missing.parquet", True),
        ("mkdir", "restored/a.parquet", True, True),
        ("mkdir", "restored/z.parquet", True, True),
    ]
    assert replacements == [
        (backup_a_existing, final_a),
        (backup_z_existing, final_z),
    ]


def test_v2_backup_existing_keeps_hidden_backup_next_to_original(tmp_path: Path) -> None:
    path = tmp_path / "output/artifact.parquet"
    path.parent.mkdir()
    path.write_bytes(b"payload")
    backup: Path | None = None
    try:
        backup = language_splits._backup_existing(path)

        assert not path.exists()
        assert backup.parent == path.parent
        assert backup.name.startswith(f".{path.name}.")
        assert backup.name.endswith(".backup")
        assert backup.read_bytes() == b"payload"
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)


def test_v2_previous_manifest_reader_requires_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    encodings: list[object] = []

    def fake_read_text(self: Path, *, encoding: str) -> str:
        encodings.append(encoding)
        return "{}"

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert language_splits._read_previous_manifest(path) == {}
    assert encodings == ["utf-8"]


def test_v2_output_directories_are_derived_only_from_owned_paths(tmp_path: Path) -> None:
    destination = tmp_path / "language_splits"
    owned_path = destination / "configuration/lang-en/source.parquet"

    assert language_splits._output_directories_for_paths(destination, {owned_path}) == {
        destination / "configuration",
        destination / "configuration/lang-en",
    }


def test_v2_output_directories_continue_after_unowned_paths(tmp_path: Path) -> None:
    destination = tmp_path / "language_splits"
    owned_path = destination / "configuration/lang-en/source.parquet"
    unowned_path = tmp_path / "outside/source.parquet"

    directories = language_splits._output_directories_for_paths(
        destination,
        cast(set[Path], (unowned_path, owned_path)),
    )

    assert directories == {
        destination / "configuration",
        destination / "configuration/lang-en",
    }


def test_v2_remove_empty_output_directories_prunes_deepest_first(tmp_path: Path) -> None:
    destination = tmp_path / "language_splits"
    owned_path = destination / "configuration/lang-en/source.parquet"
    owned_path.parent.mkdir(parents=True)

    language_splits._remove_empty_output_directories(destination, {owned_path})

    assert not (destination / "configuration/lang-en").exists()
    assert not (destination / "configuration").exists()


def test_v2_remove_empty_output_directories_requires_depth_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Directory:
        def __init__(self, name: str, depth: int, events: list[str]) -> None:
            self.name = name
            self.parts = tuple(f"part-{index}" for index in range(depth))
            self.events = events

        def rmdir(self) -> None:
            self.events.append(self.name)

    events: list[str] = []
    outer = Directory("outer", 2, events)
    inner = Directory("inner", 3, events)
    monkeypatch.setattr(
        language_splits,
        "_output_directories_for_paths",
        lambda destination, owned_paths: {outer, inner},
    )

    language_splits._remove_empty_output_directories(tmp_path, set())

    assert events == ["inner", "outer"]


def test_v2_remove_empty_output_directories_continues_after_nonempty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Directory:
        def __init__(self, name: str, depth: int, events: list[str]) -> None:
            self.name = name
            self.parts = tuple(f"part-{index}" for index in range(depth))
            self.events = events

        def rmdir(self) -> None:
            self.events.append(self.name)
            if self.name == "inner":
                raise OSError("directory is not empty")

    events: list[str] = []
    outer = Directory("outer", 2, events)
    inner = Directory("inner", 3, events)
    monkeypatch.setattr(
        language_splits,
        "_output_directories_for_paths",
        lambda destination, owned_paths: {outer, inner},
    )

    language_splits._remove_empty_output_directories(tmp_path, set())

    assert events == ["inner", "outer"]


def test_v2_split_conservation_guard_rejects_missing_output(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    inventory = build_language_inventory(root, DatasetContract.V2)

    with pytest.raises(V2LanguageSplitError, match="row conservation failed"):
        _validate_conservation(inventory, ())


def test_v2_split_can_be_loaded_with_standard_datasets(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    french_files = sorted(
        (root / "language_splits/wikipedia_documents_by_language/lang-fr").glob("*.parquet")
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import json",
                    "import sys",
                    "from datasets import load_dataset",
                    "dataset = load_dataset(",
                    '    "parquet",',
                    "    data_files=sys.argv[1:-1],",
                    '    split="train",',
                    "    cache_dir=sys.argv[-1],",
                    ")",
                    "print(json.dumps({",
                    '    "num_rows": dataset.num_rows,',
                    '    "column_names": dataset.column_names,',
                    '    "document_id": list(dataset["document_id"]),',
                    "}))",
                )
            ),
            *(str(path) for path in french_files),
            str(tmp_path / "hf-cache"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "num_rows": 2,
        "column_names": [field.name for field in wikipedia_document_v2_schema()],
        "document_id": ["doc-fr-a", "doc-fr-z"],
    }


def test_v2_split_module_runs_as_a_local_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_v2_fixture(tmp_path)

    assert main([str(root), "--batch-size", "1"]) == 0

    captured = capsys.readouterr()
    assert str(root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH) in captured.out
