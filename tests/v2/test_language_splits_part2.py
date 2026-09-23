"""Split coverage tests (part 2)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from osm_polygon_wikidata_only.io import staged_install
from tests.v2.language_splits_support import *


def test_v2_writer_increments_shard_indices_after_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})
    created: list[Path] = []

    @contextmanager
    def fake_atomic(path: Path):
        yield path

    class FakeWriter:
        def __init__(self, path: Path, schema: pa.Schema, *, compression: str) -> None:
            created.append(path)

    monkeypatch.setattr(language_splits, "atomic_replacement", fake_atomic)
    monkeypatch.setattr(language_splits.pq, "ParquetWriter", FakeWriter)

    with ExitStack() as stack:
        shards = []
        for _ in range(3):
            shards.append(
                language_splits._writer_for_language(
                    "en",
                    tmp_path / "destination",
                    tmp_path / "stage",
                    spec,
                    10,
                    {"en": 3},
                    spec.schema_factory(),
                    state,
                    stack,
                )
            )
            state.current.pop("en")

    assert [shard.shard_index for shard in shards] == [0, 1, 2]
    assert [path.name for path in created] == [
        "part-00000-of-00003.parquet",
        "part-00001-of-00003.parquet",
        "part-00002-of-00003.parquet",
    ]


def test_v2_validated_output_checks_schema_and_records_all_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    checks: list[object] = []

    class FakeSchema:
        def equals(self, other: object, *, check_metadata: bool) -> bool:
            checks.append(check_metadata)
            return other is self and check_metadata is True

    schema = FakeSchema()

    class FakeParquetFile:
        schema_arrow = schema
        metadata = None

        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> FakeParquetFile:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(language_splits, "open_parquet", FakeParquetFile)
    monkeypatch.setattr(language_splits, "sha256_file", lambda path: "digest")
    root = tmp_path / "processed_v2"
    staged = root / "stage/file.parquet"
    final = root / "language_splits/table/lang-fr/file.parquet"

    shard = language_splits._ShardWriteState(
        language="fr",
        shard_index=0,
        writer=cast(pq.ParquetWriter, object()),
        final_path=final,
        staged_path=staged,
        row_count=0,
        source_files=["source.parquet"],
    )
    result = language_splits._validated_output_file(root, spec, shard, schema)

    assert checks == [True]
    assert result == V2LanguageSplitFile(
        table=spec.table,
        configuration=spec.configuration,
        language="fr",
        split="lang-fr",
        source_files=("source.parquet",),
        path="language_splits/table/lang-fr/file.parquet",
        row_count=0,
        sha256="digest",
    )


def test_v2_main_parses_a_path_and_default_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, int]] = []

    def fake_build(
        processed_root: Path,
        *,
        output_root: Path | None = None,
        batch_size: int,
    ) -> V2LanguageSplitResult:
        calls.append((processed_root, batch_size))
        return V2LanguageSplitResult(
            processed_root=processed_root,
            output_root=output_root or processed_root / "language_splits",
            manifest_path=processed_root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
            inventory=cast(LanguageInventory, object()),
            files=(),
        )

    monkeypatch.setattr(language_splits, "build_v2_language_splits", fake_build)

    assert main([str(tmp_path)]) == 0
    assert calls == [(tmp_path, DEFAULT_BATCH_SIZE)]
    assert capsys.readouterr().out.strip() == str(tmp_path / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH)


def test_v2_main_help_describes_the_generator(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    assert "Bounded, deterministic row-level language partitions" in capsys.readouterr().out


def test_v2_split_preserves_sorted_source_and_row_order(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root, batch_size=1)

    assert [
        path.name
        for path in sorted(
            (root / "language_splits/wikipedia_documents_by_language/lang-fr").glob("*.parquet")
        )
    ] == ["part-00000-of-00001.parquet"]
    assert [row["document_id"] for row in _rows(root, "wikipedia_documents", "fr")] == [
        "doc-fr-a",
        "doc-fr-z",
    ]
    assert [row["document_id"] for row in _rows(root, "wikipedia_documents", "de")] == ["doc-de-a"]


def test_v2_split_is_byte_stable_on_repeated_runs(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    first = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "language_splits").rglob("*.parquet")
    }
    first_manifest = (root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_bytes()

    build_v2_language_splits(root)
    second = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "language_splits").rglob("*.parquet")
    }

    assert second == first
    assert (root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_bytes() == first_manifest
    assert not list((root / "language_splits").rglob("*.tmp"))


def test_v2_split_removes_obsolete_shards_before_manifest_publication(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    stale = _shard(root, "wikipedia_documents", "de", "part-00000-of-00001")
    assert stale.is_file()
    unmanaged = root / "language_splits/operator-data/operator-owned.parquet"
    unmanaged.parent.mkdir(parents=True)
    unmanaged.write_bytes(b"keep this file")
    operator_empty = root / "language_splits/operator-empty/nested"
    operator_empty.mkdir(parents=True)

    _write_table(
        root / "wikipedia/documents/a-latest.parquet",
        [_document("doc-fr-a", "fr", "French A")],
        wikipedia_document_v2_schema(),
    )

    build_v2_language_splits(root)

    assert not stale.exists()
    assert not stale.parent.exists()
    assert unmanaged.read_bytes() == b"keep this file"
    assert operator_empty.is_dir()
    manifest = _manifest(root)
    assert all(
        file["path"]
        != "language_splits/wikipedia_documents_by_language/lang-de/part-00000-of-00001.parquet"
        for table in cast(list[dict[str, object]], manifest["tables"])
        for bucket in cast(list[dict[str, object]], table["buckets"])
        for file in cast(list[dict[str, object]], bucket["files"])
    )


def test_v2_manifest_path_helper_ignores_unowned_records(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    build_v2_language_splits(root)
    payload = _manifest(root)
    tables = cast(list[dict[str, object]], payload["tables"])
    tables.append(
        {
            "buckets": [
                {
                    "files": [
                        {"path": "polygons/a-latest.parquet"},
                        {"path": "language_splits/operator-owned.parquet"},
                        {"path": "language_splits/operator-data/lang-fr/a.txt"},
                        {"path": 42},
                    ]
                }
            ]
        }
    )

    actual = _manifest_partition_paths(payload, root, root / "language_splits")
    expected = {
        path.resolve() for path in (root / "language_splits").rglob("*.parquet") if path.is_file()
    }
    assert actual == expected


def test_v2_manifest_records_preserves_the_typed_mapping_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, object]] = []

    def fake_cast(type_arg: object, value: object) -> object:
        observed.append((type_arg, value))
        return value

    monkeypatch.setattr(language_splits, "cast", fake_cast)

    assert language_splits._manifest_records([{"path": "language_splits/file.parquet"}]) == (
        {"path": "language_splits/file.parquet"},
    )
    assert observed == [(dict[str, object], {"path": "language_splits/file.parquet"})]


def test_v2_split_does_not_route_or_copy_the_polygon_table(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    source = (root / "polygons/a-latest.parquet").read_bytes()

    build_v2_language_splits(root)

    assert (root / "polygons/a-latest.parquet").read_bytes() == source
    assert not list((root / "language_splits").rglob("*polygon*.parquet"))
    assert all(
        "polygons" not in path.parts for path in (root / "language_splits").rglob("*.parquet")
    )


def test_v2_split_rejects_a_v1_processed_root_without_writing_output(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    manifest = root / "manifests" / "processed_pbfs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "fixture.osm.pbf": {
                    "polygons_path": "polygons/fixture.parquet",
                    "polygon_articles_path": "polygon_articles/fixture.parquet",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="V2 manifest contract version mismatch"):
        build_v2_language_splits(root)

    assert not (root / "language_splits").exists()


def test_v2_split_rejects_a_nonpositive_batch_size_without_writing_output(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_v2_language_splits(root, batch_size=0)

    assert not (root / "language_splits").exists()


def test_v2_split_rejects_output_outside_processed_root(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    with pytest.raises(ValueError, match="output must be under"):
        build_v2_language_splits(root, output_root=tmp_path / "outside")

    assert not (root / "language_splits").exists()


def test_v2_split_rejects_output_root_that_overlaps_source_artifacts(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    source = (root / "polygons/a-latest.parquet").read_bytes()

    with pytest.raises(V2LanguageSplitError, match="must not overlap"):
        build_v2_language_splits(root, output_root=root)

    assert (root / "polygons/a-latest.parquet").read_bytes() == source
    assert not (root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).exists()


def test_v2_split_rolls_back_after_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_v2_fixture(tmp_path)
    build_v2_language_splits(root)
    before = _release_snapshot(root)

    source_path = root / "polygon_document_links/a-latest.parquet"
    rows = pq.read_table(source_path).to_pylist()
    rows[0]["language"] = "de"
    _write_table(source_path, rows, polygon_document_link_v2_schema())

    def fail_after_first_install(
        staged: dict[Path, Path],
        installed: list[Path],
        *,
        manifest_path: Path,
    ) -> None:
        final, temporary = min(staged.items(), key=lambda item: item[0].as_posix())
        final.parent.mkdir(parents=True, exist_ok=True)
        staged_install.os.replace(temporary, final)
        installed.append(final)
        raise RuntimeError("injected language split failure")

    monkeypatch.setattr(staged_install, "install_files", fail_after_first_install)
    with pytest.raises(RuntimeError, match="injected language split failure"):
        build_v2_language_splits(root)

    assert _release_snapshot(root) == before
    # The staging tree is deliberately retained after a failure so the next
    # run resumes from the tables that already completed. Installed output
    # is still rolled back, which the snapshot assertion above covers.
    assert [p.name for p in root.glob(".language_splits-*")] == [".language_splits-staging"]


def test_v2_split_rolls_back_if_backup_phase_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_v2_fixture(tmp_path)
    build_v2_language_splits(root)
    before = _release_snapshot(root)
    original_backup_targets = staged_install.backup_targets

    def fail_during_backup(targets: list[Path], backups: dict[Path, Path]) -> None:
        original_backup_targets(targets[:1], backups)
        raise RuntimeError("injected backup failure")

    monkeypatch.setattr(staged_install, "backup_targets", fail_during_backup)
    with pytest.raises(RuntimeError, match="injected backup failure"):
        build_v2_language_splits(root)

    assert _release_snapshot(root) == before
    # The staging tree is deliberately retained after a failure so the next
    # run resumes from the tables that already completed. Installed output
    # is still rolled back, which the snapshot assertion above covers.
    assert [p.name for p in root.glob(".language_splits-*")] == [".language_splits-staging"]


def test_v2_install_files_sorts_by_final_path_and_records_every_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_z = tmp_path / "output/z.parquet"
    final_a = tmp_path / "output/a.parquet"
    temporary_z = tmp_path / "staged/0-for-z"
    temporary_a = tmp_path / "staged/9-for-a"
    temporary_z.parent.mkdir()
    temporary_a.write_bytes(b"a")
    temporary_z.write_bytes(b"z")
    replacements: list[tuple[Path, Path]] = []
    original_replace = staged_install.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(staged_install.os, "replace", recording_replace)
    installed: list[Path] = []

    staged_install.install_files(
        {final_z: temporary_z, final_a: temporary_a},
        installed,
    )

    assert replacements == [(temporary_a, final_a), (temporary_z, final_z)]
    assert installed == [final_a, final_z]
    assert final_a.read_bytes() == b"a"
    assert final_z.read_bytes() == b"z"


def test_v2_nested_output_files_are_installed_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "processed_v2"
    destination = root / "nested/releases/language_splits"
    data_final = destination / "wikipedia_documents_by_language/lang-fr/a.parquet"
    data_stage = destination.parent / ".language_splits-stage/data.parquet"
    manifest_final = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    manifest_stage = destination.parent / ".language_splits-stage/manifests/language_splits.json"
    data_stage.parent.mkdir(parents=True)
    manifest_stage.parent.mkdir(parents=True)
    data_stage.write_bytes(b"data")
    manifest_stage.write_bytes(b"manifest")
    replacements: list[tuple[Path, Path]] = []
    original_replace = staged_install.os.replace

    def recording_replace(source: Path, target: Path) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(staged_install.os, "replace", recording_replace)

    language_splits._install_staged_files(
        root,
        destination,
        {data_final: data_stage, manifest_final: manifest_stage},
    )

    assert replacements[-1] == (manifest_stage, manifest_final)
    assert data_final.read_bytes() == b"data"
    assert manifest_final.read_bytes() == b"manifest"


def test_v2_cross_filesystem_replace_is_rejected_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_v2_fixture(tmp_path)
    build_v2_language_splits(root)
    before = _release_snapshot(root)
    manifest_path = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    original_replace = staged_install.os.replace
    failed = False

    def reject_manifest_install(source: Path, target: Path) -> None:
        nonlocal failed
        if target == manifest_path and not failed:
            failed = True
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        original_replace(source, target)

    monkeypatch.setattr(staged_install.os, "replace", reject_manifest_install)

    with pytest.raises(V2LanguageSplitError, match="EXDEV") as error:
        build_v2_language_splits(root)

    assert str(error.value).startswith(
        "V2 language split publication cannot cross filesystems (EXDEV): "
    )
    assert _release_snapshot(root) == before
    # The staging tree is deliberately retained after a failure so the next
    # run resumes from the tables that already completed. Installed output
    # is still rolled back, which the snapshot assertion above covers.
    assert [p.name for p in root.glob(".language_splits-*")] == [".language_splits-staging"]


def test_v2_install_files_requires_explicit_final_path_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InstallPath:
        def __init__(self, value: str, events: list[object]) -> None:
            self.value = value
            self.events = events

        @property
        def parent(self) -> InstallPath:
            return self

        def as_posix(self) -> str:
            return self.value

        def mkdir(self, *, parents: bool, exist_ok: bool) -> None:
            self.events.append(("mkdir", self.value, parents, exist_ok))

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, InstallPath) and self.value == other.value

    events: list[object] = []
    final_z = InstallPath("language_splits/z.parquet", events)
    final_a = InstallPath("language_splits/a.parquet", events)
    temporary_z = InstallPath("stage/z", events)
    temporary_a = InstallPath("stage/a", events)
    replacements: list[tuple[InstallPath, InstallPath]] = []

    monkeypatch.setattr(
        staged_install.os,
        "replace",
        lambda source, destination: replacements.append((source, destination)),
    )
    installed: list[InstallPath] = []

    staged_install.install_files(
        {final_z: temporary_z, final_a: temporary_a},
        installed,
    )

    assert replacements == [(temporary_a, final_a), (temporary_z, final_z)]
    assert installed == [final_a, final_z]
