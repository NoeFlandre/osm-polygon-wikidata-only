"""Language-split build, install, and resume tests."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageBucket,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    build_language_inventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.io import staged_install
from osm_polygon_wikidata_only.v2 import language_splits
from osm_polygon_wikidata_only.v2.language_splits import (
    DEFAULT_BATCH_SIZE,
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    V2_LANGUAGE_SPLIT_CONTRACT_VERSION,
    V2LanguageSplitError,
    V2LanguageSplitFile,
    V2LanguageSplitResult,
    _file_sort_key,
    _language_sort_key,
    _manifest_partition_paths,
    _validate_conservation,
    build_v2_language_splits,
    main,
)
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    wikipedia_document_v2_schema,
)
from tests.v2.language_splits_support import (
    _document,
    _manifest,
    _release_snapshot,
    _resume_file_record,
    _resume_inventory,
    _rows,
    _shard,
    _write_table,
    _write_v2_fixture,
)


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


def test_v2_staging_uses_deterministic_destination_local_stage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging is destination-local and deterministic, and is cleared on success."""
    root = tmp_path / "processed_v2"
    root.mkdir()
    destination = root / "nested/language_splits"
    stage_root = destination.parent / ".language_splits-staging"
    calls: dict[str, object] = {}

    def fake_stage(*args: object) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
        calls["stage_root"] = args[2]
        return [], {}

    def fake_install(root_arg: Path, destination_arg: Path, staged: dict[Path, Path]) -> None:
        calls["install"] = (root_arg, destination_arg, staged)

    def fake_rmtree(path: Path, *, ignore_errors: bool) -> None:
        calls["rmtree"] = (path, ignore_errors)

    monkeypatch.setattr(language_splits, "_stage_v2_files", fake_stage)
    monkeypatch.setattr(language_splits, "_validate_conservation", lambda *args: None)
    monkeypatch.setattr(language_splits, "_manifest_payload", lambda *args: {})
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
    assert calls["stage_root"] == stage_root
    assert stage_root.is_dir() or calls["rmtree"] == (stage_root, True)
    assert calls["rmtree"] == (stage_root, True)
    assert calls["install"] == (
        root,
        destination,
        {
            root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH: stage_root
            / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
        },
    )


def test_v2_staging_survives_failure_so_the_next_run_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed release keeps its staging tree; resuming depends on it."""
    root = tmp_path / "processed_v2"
    root.mkdir()
    destination = root / "nested/language_splits"
    stage_root = destination.parent / ".language_splits-staging"
    removed: list[Path] = []

    def boom(*args: object) -> tuple[list[V2LanguageSplitFile], dict[Path, Path]]:
        raise V2LanguageSplitError("staging failed")

    monkeypatch.setattr(language_splits, "_stage_v2_files", boom)
    monkeypatch.setattr(language_splits.shutil, "rmtree", lambda path, **_kw: removed.append(path))

    with pytest.raises(V2LanguageSplitError, match="staging failed"):
        language_splits._stage_and_install_v2_release(
            root,
            destination,
            cast(LanguageInventory, object()),
            1,
            100_000,
            root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
        )

    assert removed == []
    assert stage_root.is_dir()


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


def test_v2_writer_and_resume_helpers_keep_boundary_contracts_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the small state transitions that protect resumable writes."""
    spec = language_table_specs(DatasetContract.V2)[0]
    zero = LanguageBucket(
        language="en",
        row_count=0,
        canonical_rows=0,
        legacy_alias_rows=0,
        missing_rows=0,
        blank_rows=0,
        malformed_rows=0,
        legacy_unusable_rows=0,
    )
    one = LanguageBucket(
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
        table=spec.table,
        configuration=spec.configuration,
        language_column=spec.language_column,
        identity_columns=spec.identity_columns,
        source_files=(),
        row_count=1,
        buckets=(zero, one),
    )
    assert language_splits._table_shard_counts(inventory, 100) == {"fr": 1}

    written: list[pa.Table] = []

    class Writer:
        def write_table(self, table: pa.Table) -> None:
            written.append(table)

    state = language_splits._TableWriteState({}, defaultdict(int), [], {})
    shard = SimpleNamespace(
        source_files=[], row_count=0, writer=Writer(), pending=[], pending_bytes=0
    )
    first = pa.record_batch([pa.array(["en"] * 2)], names=["language"])
    second = pa.record_batch([pa.array(["en"] * 3)], names=["language"])
    monkeypatch.setattr(language_splits, "_SHARD_FLUSH_BYTES", 10**9)
    monkeypatch.setattr(language_splits, "_TOTAL_FLUSH_BYTES", 10**9)
    language_splits._buffer_rows(state, shard, first)
    language_splits._buffer_rows(state, shard, second)
    assert shard.pending_bytes == first.nbytes + second.nbytes
    assert state.pending_bytes == shard.pending_bytes
    language_splits._flush_shard(state, shard)
    assert [table.num_rows for table in written] == [5]
    assert shard.pending == []
    assert shard.pending_bytes == state.pending_bytes == 0

    shard.source_files = ["source-a.parquet"]
    language_splits._record_source_file(shard, "source-a.parquet")
    language_splits._record_source_file(shard, "source-b.parquet")
    assert shard.source_files == ["source-a.parquet", "source-b.parquet"]

    marker = language_splits._resume_marker_path(tmp_path, spec)
    assert marker == tmp_path / ".resume" / f"{spec.table.value}.json"
    staged = tmp_path / "staged.parquet"
    staged.write_bytes(b"staged")
    assert language_splits._resume_staged_entry("final.parquet", str(staged)) == (
        Path("final.parquet"),
        staged,
    )
    assert language_splits._resume_staged_entry(4, str(staged)) is None

    record = _resume_file_record("language_splits/lang-fr.parquet").to_dict()
    restored = language_splits._resume_file(record, spec)
    assert restored is not None
    assert restored.path == record["path"]
    assert language_splits._resume_file({**record, "source_files": "bad"}, spec) is None
    assert language_splits._resume_file({**record, "configuration": None}, spec) is None
    assert language_splits._resume_file_rows({"source_files": ["a"], "row_count": 2}) == (
        ("a",),
        2,
    )
    assert language_splits._resume_file_rows({"source_files": "a", "row_count": 2}) is None


def test_v2_write_indices_observes_capacity_offsets_and_close_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slice is written once, then the next shard receives only its remainder."""
    batch = pa.record_batch([pa.array(["en"] * 3)], names=["language"])
    spec = language_table_specs(DatasetContract.V2)[0]
    state = language_splits._TableWriteState({}, defaultdict(int), [], {})

    class FakeWriter:
        def __init__(self) -> None:
            self.rows: list[list[dict[str, object]]] = []

        def write_table(self, table: pa.Table) -> None:
            self.rows.append(table.to_pylist())

        def close(self) -> None:
            return None

    first = SimpleNamespace(
        source_files=[], row_count=9, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    second = SimpleNamespace(
        source_files=[], row_count=0, writer=FakeWriter(), pending=[], pending_bytes=0
    )
    state.current["en"] = first
    writers = iter((first, second))

    def next_writer(*_args: object) -> SimpleNamespace:
        try:
            return next(writers)
        except StopIteration as error:
            raise AssertionError("the writer loop consumed more than two shards") from error

    monkeypatch.setattr(language_splits, "_writer_for_language", next_writer)
    with ExitStack() as stack:
        language_splits._write_language_indices(
            "en",
            pa.array([0, 1, 2], type=pa.int64()),
            batch,
            tmp_path / "destination",
            tmp_path / "stage",
            spec,
            "source.parquet",
            10,
            {"en": 2},
            batch.schema,
            state,
            stack,
        )
    language_splits._flush_shard(state, second)

    assert first.row_count == 10
    assert second.row_count == 2
    assert [row for group in first.writer.rows for row in group] == [{"language": "en"}]
    assert [row for group in second.writer.rows for row in group] == [
        {"language": "en"},
        {"language": "en"},
    ]


def test_v2_staged_table_uses_the_resume_result_and_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed table is returned without rebuilding or changing its inputs."""
    spec = language_table_specs(DatasetContract.V2)[0]
    inventory = _resume_inventory()
    record = _resume_file_record("language_splits/lang-fr.parquet")
    resumed = ([record], {tmp_path / record.path: tmp_path / "staged.parquet"})
    observed: list[object] = []

    def resume(_stage: Path, _spec: object, table: object):
        observed.append(table)
        return resumed

    monkeypatch.setattr(language_splits, "_resume_completed_table", resume)
    monkeypatch.setattr(
        language_splits,
        "_write_table",
        lambda *_args: (_ for _ in ()).throw(AssertionError("completed tables are not rebuilt")),
    )

    result = language_splits._staged_table(
        tmp_path,
        tmp_path / "destination",
        tmp_path / "stage",
        spec,
        inventory,
        1,
        10,
    )

    assert result == resumed
    assert observed == [inventory]


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

    monkeypatch.setattr(staged_install, "backup_targets", fake_backup)
    monkeypatch.setattr(staged_install, "install_files", fake_install)
    monkeypatch.setattr(staged_install, "cleanup_transaction", fake_cleanup)
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
    original_replace = staged_install.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(staged_install.os, "replace", recording_replace)

    staged_install.restore_files(
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
        staged_install.os,
        "replace",
        lambda source, destination: replacements.append((source, destination)),
    )

    staged_install.restore_files(
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
        backup = staged_install.backup_existing(path)

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


def test_v2_completed_table_is_reused_instead_of_rebuilt(tmp_path: Path) -> None:
    """A table recorded as complete is resumed rather than staged again."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    inventory = _resume_inventory()
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    record = _resume_file_record("language_splits/lang-fr.parquet")

    language_splits._record_completed_table(stage_root, spec, inventory, [record], {final: staged})

    resumed = language_splits._resume_completed_table(stage_root, spec, inventory)
    assert resumed is not None
    files, staged_paths = resumed
    assert [file.to_dict() for file in files] == [record.to_dict()]
    assert staged_paths == {final: staged}


def test_v2_resume_is_rejected_when_the_sources_changed(tmp_path: Path) -> None:
    """A marker written from different sources must not be trusted."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    language_splits._record_completed_table(
        stage_root,
        spec,
        _resume_inventory(row_count=4),
        [_resume_file_record("language_splits/lang-fr.parquet")],
        {final: staged},
    )

    assert (
        language_splits._resume_completed_table(stage_root, spec, _resume_inventory(row_count=5))
        is None
    )


def test_v2_resume_is_rejected_when_a_staged_file_disappeared(tmp_path: Path) -> None:
    """Resuming must not trust a marker whose staged output is gone."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    inventory = _resume_inventory()
    staged = stage_root / "lang-fr.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"staged")
    final = tmp_path / "language_splits/lang-fr.parquet"
    language_splits._record_completed_table(
        stage_root,
        spec,
        inventory,
        [_resume_file_record("language_splits/lang-fr.parquet")],
        {final: staged},
    )
    staged.unlink()

    assert language_splits._resume_completed_table(stage_root, spec, inventory) is None


def test_v2_resume_is_rejected_when_the_marker_is_corrupt(tmp_path: Path) -> None:
    """A truncated or non-JSON marker falls back to rebuilding the table."""
    spec = language_table_specs(DatasetContract.V2)[0]
    stage_root = tmp_path / "stage"
    marker = language_splits._resume_marker_path(stage_root, spec)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{not json", encoding="utf-8")

    assert language_splits._resume_completed_table(stage_root, spec, _resume_inventory()) is None
