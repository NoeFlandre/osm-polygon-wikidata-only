"""Split coverage tests (part 1)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.v2.language_splits_support import *


def test_v2_split_keeps_each_multilingual_row_in_its_own_partition(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)

    french_documents = _rows(root, "wikipedia_documents", "fr")
    german_documents = _rows(root, "wikipedia_documents", "de")
    french_links = _rows(root, "polygon_document_links", "fr")
    german_links = _rows(root, "polygon_document_links", "de")

    assert [row["document_id"] for row in french_documents] == ["doc-fr-a", "doc-fr-z"]
    assert [row["document_id"] for row in german_documents] == ["doc-de-a"]
    assert [(row["polygon_id"], row["document_id"]) for row in french_links] == [
        ("polygon-1", "doc-fr-a"),
        ("polygon-2", "doc-fr-z"),
    ]
    assert [(row["polygon_id"], row["document_id"]) for row in german_links] == [
        ("polygon-1", "doc-de-a")
    ]
    assert (root / "language_splits" / "polygons").exists() is False


def test_v2_split_aggregates_rows_into_deterministic_bounded_shards(
    tmp_path: Path,
) -> None:
    root = _write_v2_fixture(tmp_path)

    result = build_v2_language_splits(root, batch_size=1, max_rows_per_shard=1)

    french_dir = root / "language_splits/wikipedia_documents_by_language/lang-fr"
    assert sorted(path.name for path in french_dir.glob("*.parquet")) == [
        "part-00000-of-00002.parquet",
        "part-00001-of-00002.parquet",
    ]
    assert [
        row["document_id"]
        for path in sorted(french_dir.glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ] == ["doc-fr-a", "doc-fr-z"]

    french_files = [
        file
        for file in result.files
        if file.table is LanguageTable.WIKIPEDIA_DOCUMENTS and file.language == "fr"
    ]
    assert [file.source_files for file in french_files] == [
        ("wikipedia/documents/a-latest.parquet",),
        ("wikipedia/documents/z-latest.parquet",),
    ]
    manifest_files = [
        file
        for table in cast(list[dict[str, object]], _manifest(root)["tables"])
        for bucket in cast(list[dict[str, object]], table["buckets"])
        for file in cast(list[dict[str, object]], bucket["files"])
        if file["table"] == "wikipedia_documents" and file["language"] == "fr"
    ]
    assert all("source_files" in file and "source_file" not in file for file in manifest_files)
    assert _manifest(root)["contract_version"] == V2_LANGUAGE_SPLIT_CONTRACT_VERSION


def test_v2_split_routes_unusable_values_to_lang_unknown(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)

    unknown = _rows(root, "polygon_document_links", "unknown")
    aliases = _rows(root, "polygon_document_links", "be-tarask")

    assert [row["document_id"] for row in unknown] == [
        "doc-missing-a",
        "doc-blank-a",
        "doc-malformed-a",
        "doc-simple-a",
    ]
    assert [row["language"] for row in unknown] == [None, "  ", "en/fr", "simple"]
    assert [row["language"] for row in aliases] == ["be_x_old"]


def test_v2_split_conserves_rows_and_preserves_schema(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    build_v2_language_splits(root)
    inventory = build_language_inventory(root, DatasetContract.V2)

    table_specs = {spec.table: spec for spec in language_table_specs(DatasetContract.V2)}
    for table_inventory in inventory.tables:
        spec = table_specs[table_inventory.table]
        observed = 0
        for bucket in table_inventory.buckets:
            paths = sorted(
                (root / "language_splits" / spec.configuration / bucket.split).glob("*.parquet")
            )
            observed += sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
            for path in paths:
                assert pq.ParquetFile(path).schema_arrow.equals(
                    spec.schema_factory(), check_metadata=True
                )
        assert observed == table_inventory.row_count

    manifest = _manifest(root)
    assert manifest["dataset_contract"] == "v2"
    assert manifest["v2_contract_version"] == "wikipedia-tags-v2"
    assert manifest["source_manifest"] == "manifests/processed_pbfs.json"

    assert set(manifest) == {
        "artifact_fingerprint",
        "contract_version",
        "dataset_contract",
        "dataset_id",
        "language_split_contract_version",
        "output_root",
        "source_manifest",
        "source_manifest_sha256",
        "tables",
        "v2_contract_version",
    }
    for table in cast(list[dict[str, object]], manifest["tables"]):
        assert set(table) == {
            "buckets",
            "configuration",
            "identity_columns",
            "language_column",
            "row_count",
            "source_files",
            "table",
        }
        for bucket in cast(list[dict[str, object]], table["buckets"]):
            assert set(bucket) == {
                "blank_rows",
                "canonical_rows",
                "files",
                "language",
                "legacy_alias_rows",
                "legacy_unusable_rows",
                "malformed_rows",
                "missing_rows",
                "row_count",
                "split",
            }
            for file in cast(list[dict[str, object]], bucket["files"]):
                assert set(file) == {
                    "configuration",
                    "language",
                    "path",
                    "row_count",
                    "sha256",
                    "source_files",
                    "split",
                    "table",
                }


def test_v2_split_result_and_nested_output_parent_are_explicit(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    output_root = root / "nested/releases/language_splits"

    result = build_v2_language_splits(root, output_root=output_root)

    assert result.processed_root == root.resolve()
    assert result.output_root == output_root.resolve()
    assert result.manifest_path == root.resolve() / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    assert result.files
    assert {file.path for file in result.files} == {
        file["path"]
        for table in cast(list[dict[str, object]], _manifest(root)["tables"])
        for bucket in cast(list[dict[str, object]], table["buckets"])
        for file in cast(list[dict[str, object]], bucket["files"])
    }


def test_v2_alternating_output_roots_remove_previous_owned_shards(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)
    custom = root / "nested/releases/language_splits"

    build_v2_language_splits(root)
    default_shards = sorted((root / "language_splits").rglob("*.parquet"))
    assert default_shards

    build_v2_language_splits(root, output_root=custom)

    assert default_shards
    assert all(not path.exists() for path in default_shards)
    assert list(custom.rglob("*.parquet"))

    build_v2_language_splits(root)

    assert not list(custom.rglob("*.parquet"))
    assert list((root / "language_splits").rglob("*.parquet"))


def test_v2_split_overlap_rejection_checks_each_source_location(tmp_path: Path) -> None:
    root = _write_v2_fixture(tmp_path)

    with pytest.raises(V2LanguageSplitError) as error:
        build_v2_language_splits(root, output_root=root / "wikipedia")
    assert str(error.value).startswith(
        "V2 language split output must not overlap source artifacts:"
    )


def test_v2_sort_keys_put_unknown_after_known_languages() -> None:
    known = V2LanguageSplitFile(
        table=LanguageTable.POLYGON_ARTICLES,
        configuration="polygon_articles_by_language",
        language="en",
        split="lang-en",
        source_files=("z.parquet",),
        path="",
        row_count=1,
        sha256="",
    )
    unknown = V2LanguageSplitFile(
        table=known.table,
        configuration=known.configuration,
        language="unknown",
        split="lang-unknown",
        source_files=("a.parquet",),
        path="",
        row_count=1,
        sha256="",
    )

    assert _language_sort_key("en") < _language_sort_key("unknown")
    assert [file.language for file in sorted((unknown, known), key=_file_sort_key)] == [
        "en",
        "unknown",
    ]


def test_v2_staging_uses_destination_local_temp_and_best_effort_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "processed_v2"
    root.mkdir()
    destination = root / "nested/language_splits"
    stage_root = destination.parent / ".language_splits-stage"
    calls: dict[str, object] = {}

    def fake_mkdtemp(*, prefix: str, dir: Path) -> str:
        calls["mkdtemp"] = (prefix, dir)
        stage_root.mkdir(parents=True, exist_ok=True)
        return str(stage_root)

    def fake_stage(*args: object) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
        return [], {}

    def fake_manifest(*args: object) -> dict[str, object]:
        return {}

    def fake_install(root_arg: Path, destination_arg: Path, staged: dict[Path, Path]) -> None:
        calls["install"] = (root_arg, destination_arg, staged)

    def fake_rmtree(path: Path, *, ignore_errors: bool) -> None:
        calls["rmtree"] = (path, ignore_errors)

    monkeypatch.setattr(language_splits.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(language_splits, "_stage_v2_files", fake_stage)
    monkeypatch.setattr(language_splits, "_validate_conservation", lambda *args: None)
    monkeypatch.setattr(language_splits, "_manifest_payload", fake_manifest)
    monkeypatch.setattr(language_splits, "atomic_write_json", lambda *args: None)
    monkeypatch.setattr(language_splits, "_install_staged_files", fake_install)
    inventory = cast(LanguageInventory, object())
    monkeypatch.setattr(
        language_splits, "_verify_source_inventory", lambda root, expected: expected
    )
    monkeypatch.setattr(language_splits.shutil, "rmtree", fake_rmtree)

    result = language_splits._stage_and_install_v2_release(
        root,
        destination,
        inventory,
        1,
        100_000,
        root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    )

    assert result == (inventory, ())
    assert calls["mkdtemp"] == (".language_splits-", destination.parent)
    assert calls["rmtree"] == (stage_root, True)
    assert calls["install"] == (
        root,
        destination,
        {
            root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH: stage_root
            / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
        },
    )


def test_v2_stage_inventory_cast_keeps_the_runtime_table_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    table_inventory = LanguageTableInventory(
        table=spec.table,
        configuration=spec.configuration,
        language_column=spec.language_column,
        identity_columns=spec.identity_columns,
        source_files=(),
        row_count=0,
        buckets=(),
    )

    class FakeInventory:
        def table(self, table: LanguageTable) -> LanguageTableInventory:
            assert table is spec.table
            return table_inventory

    observed: list[tuple[object, object]] = []

    def fake_cast(type_arg: object, value: object) -> object:
        observed.append((type_arg, value))
        return value

    monkeypatch.setattr(language_splits, "language_table_specs", lambda dataset: (spec,))
    monkeypatch.setattr(language_splits, "_write_table", lambda *args: ([], {}))
    monkeypatch.setattr(language_splits, "cast", fake_cast)

    assert language_splits._stage_v2_files(
        tmp_path / "root",
        tmp_path / "destination",
        tmp_path / "stage",
        FakeInventory(),
        1,
        100_000,
    ) == ([], {})
    assert observed == [(LanguageTableInventory, table_inventory)]


def test_v2_write_batch_routes_rows_to_language_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_write_batch`` routes selected rows to the language's shard writer."""
    batch = pa.record_batch([pa.array(["en", "en"])], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})
    observed: dict[str, object] = {}

    class FakeWriter:
        def write_table(self, selected: pa.Table) -> None:
            observed["rows"] = selected.to_pylist()

    writer = FakeWriter()
    shard = SimpleNamespace(
        source_files=[], row_count=0, writer=writer, pending=[], pending_bytes=0
    )
    state.current["en"] = shard
    state.shards.append(shard)

    def fake_writer_for_language(*args: object) -> SimpleNamespace:
        observed["max_rows_per_shard"] = args[4]
        return shard

    monkeypatch.setattr(language_splits, "_writer_for_language", fake_writer_for_language)

    with ExitStack() as stack:
        language_splits._write_batch(
            batch,
            0,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source",
            10,
            {"en": 1},
            batch.schema,
            state,
            stack,
        )

    # Rows are buffered rather than written per slice, so flush before asserting.
    language_splits._flush_shard(state, shard)
    assert observed["rows"] == [{"language": "en"}, {"language": "en"}]
    assert observed["max_rows_per_shard"] == 10


def test_v2_partition_batch_returns_explicit_wide_row_indices() -> None:
    """Partition indices carry an explicit 64-bit integer type.

    ``RecordBatch.take`` is driven by these arrays. A narrow or inferred index
    type would silently truncate on batches larger than that type can address,
    so the width is part of the contract rather than an accident of inference.
    """
    batch = pa.record_batch([pa.array(["en", "fr", "en", None])], names=["language"])
    partitions = language_splits._partition_batch(batch, 0)
    assert partitions
    for indices in partitions.values():
        assert indices.type in (pa.int64(), pa.uint64())


def test_v2_partition_batch_groups_rows_in_first_occurrence_order() -> None:
    """Partitions are ordered by first appearance and indices stay ascending."""
    batch = pa.record_batch([pa.array(["fr", "en", "fr", "en", "fr"])], names=["language"])
    partitions = language_splits._partition_batch(batch, 0)
    assert list(partitions) == ["fr", "en"]
    assert partitions["fr"].to_pylist() == [0, 2, 4]
    assert partitions["en"].to_pylist() == [1, 3]


def test_v2_table_shard_counts_omit_empty_buckets() -> None:
    bucket = LanguageBucket(
        language="en",
        row_count=0,
        canonical_rows=0,
        legacy_alias_rows=0,
        missing_rows=0,
        blank_rows=0,
        malformed_rows=0,
        legacy_unusable_rows=0,
    )
    nonempty = LanguageBucket(
        language="fr",
        row_count=1,
        canonical_rows=1,
        legacy_alias_rows=0,
        missing_rows=0,
        blank_rows=0,
        malformed_rows=0,
        legacy_unusable_rows=0,
    )
    inventory = LanguageTableInventory(
        table=LanguageTable.WIKIPEDIA_DOCUMENTS,
        configuration="wikipedia_documents_by_language",
        language_column="language",
        identity_columns=("document_id",),
        source_files=(),
        row_count=1,
        buckets=(bucket, nonempty),
    )

    assert language_splits._table_shard_counts(inventory, 100_000) == {"fr": 1}


def test_v2_write_language_indices_respects_existing_shard_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"] * 8)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})

    class FakeWriter:
        def __init__(self) -> None:
            self.rows: list[list[dict[str, object]]] = []

        def write_table(self, selected: pa.Table) -> None:
            self.rows.append(selected.to_pylist())

        def close(self) -> None:
            return None

    first = SimpleNamespace(
        source_files=[], row_count=3, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    second = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    state.current["en"] = first
    writers = iter((first, second))
    monkeypatch.setattr(language_splits, "_writer_for_language", lambda *args: next(writers))

    with ExitStack() as stack:
        language_splits._write_language_indices(
            "en",
            list(range(8)),
            batch,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source",
            10,
            {"en": 2},
            batch.schema,
            state,
            stack,
        )

    # The filled shard is flushed when it closes; the partial one stays buffered
    # until the table finishes, which is what keeps writes large and sequential.
    assert [len(rows) for rows in first.writer.rows] == [7]
    assert second.writer.rows == []
    assert [b.num_rows for b in second.pending] == [1]
    language_splits._flush_shard(state, second)
    assert [len(rows) for rows in second.writer.rows] == [1]
    assert second.pending == []
    assert state.pending_bytes == 0
    assert first.row_count == 10
    assert second.row_count == 1
    assert "en" not in state.current


def test_v2_write_language_indices_advances_offset_across_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch([pa.array(["en"] * 3)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})

    class FakeWriter:
        def write_table(self, selected: pa.Table) -> None:
            return None

        def close(self) -> None:
            return None

    first = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    second = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    state.current["en"] = first
    writers = iter((first, second))

    def fake_writer_for_language(*args: object) -> SimpleNamespace:
        try:
            return next(writers)
        except StopIteration as error:
            raise AssertionError("offset did not advance across shards") from error

    monkeypatch.setattr(language_splits, "_writer_for_language", fake_writer_for_language)

    with ExitStack() as stack:
        language_splits._write_language_indices(
            "en",
            list(range(3)),
            batch,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source",
            2,
            {"en": 2},
            batch.schema,
            state,
            stack,
        )

    assert first.row_count == 2
    assert second.row_count == 1


def test_v2_record_source_file_deduplicates_and_records_transitions() -> None:
    shard = SimpleNamespace(source_files=["source-a.parquet"])

    language_splits._record_source_file(shard, "source-a.parquet")
    language_splits._record_source_file(shard, "source-b.parquet")

    assert shard.source_files == ["source-a.parquet", "source-b.parquet"]


def test_v2_writer_uses_snappy_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})
    observed: dict[str, object] = {}

    @contextmanager
    def fake_atomic(path: Path):
        observed["temporary"] = path
        yield path

    class FakeWriter:
        def __init__(self, path: Path, schema: pa.Schema, *, compression: str) -> None:
            observed["writer"] = (path, schema, compression)

    monkeypatch.setattr(language_splits, "atomic_replacement", fake_atomic)
    monkeypatch.setattr(language_splits.pq, "ParquetWriter", FakeWriter)

    with ExitStack() as stack:
        writer = language_splits._writer_for_language(
            "en",
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            10,
            {"en": 1},
            spec.schema_factory(),
            state,
            stack,
        )

    assert writer is state.current["en"]
    assert writer.shard_index == 0
    assert observed["writer"] == (
        tmp_path / "stage" / spec.configuration / "lang-en" / "part-00000-of-00001.parquet",
        spec.schema_factory(),
        "snappy",
    )
