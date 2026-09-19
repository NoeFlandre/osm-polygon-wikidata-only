"""Split coverage tests (part 3)."""

from __future__ import annotations

# ruff: noqa: F403,F405
from tests.contracts.publication_support import *


def test_cli_commands_no_longer_implements_publication_assembly() -> None:
    """cli.commands must not contain publication assembly implementations."""
    import osm_polygon_wikidata_only.cli.commands as commands_mod

    for name in (
        "_sync_upload_files",
        "_coverage_refresh_required",
        "_generate_geographic_text_density_snapshot",
        "_write_readme_snapshot",
        "_augmentation_upload_files",
        "load_centroids_from_parquet",
        "ensure_world_land",
        "generate_coverage_map",
    ):
        assert not hasattr(commands_mod, name), f"{name} must not live in cli.commands anymore"

def test_hf_publication_no_longer_exposes_dead_types() -> None:
    """RegionUploadArtifacts and required_local_artifacts_present are gone."""
    import osm_polygon_wikidata_only.hf.publication as publication_mod

    assert not hasattr(publication_mod, "RegionUploadArtifacts")
    assert not hasattr(publication_mod, "required_local_artifacts_present")

def test_canonical_augmentation_manifest_path_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Augmentation-only publication uploads the manifest to
    ``manifests/augmentation_manifest.json`` (canonical), not the
    obsolete ``augmentation/manifests/`` path.
    """
    from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload
    from osm_polygon_wikidata_only.hf.repo_layout import (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    plan = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert all(isinstance(op, PublicationOp) for op in plan), (
        "publication assemblers must return PublicationOp records"
    )
    additions = {op.path_in_repo for op in plan if op.action == "add"}
    deletions = {op.path_in_repo for op in plan if op.action == "delete"}
    assert REMOTE_AUGMENTATION_MANIFEST_FILE in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE not in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE in deletions

def test_region_upload_publishes_augmentation_manifest_to_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unified-sync publication also places the augmentation
    manifest at the canonical ``manifests/`` path and removes the
    legacy path.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_region_upload
    from osm_polygon_wikidata_only.hf.repo_layout import (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    )

    _core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    plan = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=None,
        world_land_warning=None,
    )
    additions = {op.path_in_repo for op in plan if op.action == "add"}
    deletions = {op.path_in_repo for op in plan if op.action == "delete"}
    assert REMOTE_AUGMENTATION_MANIFEST_FILE in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE in deletions

def test_legacy_core_publication_does_not_touch_legacy_augmentation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy core publication path leaves the remote layout
    unchanged when no augmentation is involved.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_core_upload
    from osm_polygon_wikidata_only.hf.repo_layout import (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    )

    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)
    plan = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    additions = {op.path_in_repo for op in plan if op.action == "add"}
    deletions = {op.path_in_repo for op in plan if op.action == "delete"}
    assert "manifests/processed_pbfs.json" in additions
    assert REMOTE_AUGMENTATION_MANIFEST_FILE not in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE not in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE not in deletions

def test_augmentation_publication_includes_legacy_deletion_in_same_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The augmentation-only publication emits the canonical upload
    and the legacy deletion in the SAME atomic Hub commit (one plan,
    one submission)."""
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload
    from osm_polygon_wikidata_only.hf.repo_layout import (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    plan = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    additions = {op.path_in_repo for op in plan if op.action == "add"}
    deletions = {op.path_in_repo for op in plan if op.action == "delete"}

    assert REMOTE_AUGMENTATION_MANIFEST_FILE in additions
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE in deletions
    expected_additions = {
        "wikipedia/documents/monaco-latest.parquet",
        "wikipedia/sections/monaco-latest.parquet",
        "wikivoyage/documents/monaco-latest.parquet",
        "wikivoyage/sections/monaco-latest.parquet",
        "wikidata/facts/monaco-latest.parquet",
        "assets/geographic_text_presence.png",
        "assets/geographic_text_density.png",
        "assets/dataset_hero.png",
        "stats.json",
        "README.md",
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    }
    assert expected_additions == additions

def test_repeated_augmentation_publication_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second augmentation publication produces the same plan
    (canonical additions + legacy deletion). Idempotent over runs.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_augmentation_upload
    from osm_polygon_wikidata_only.hf.repo_layout import (
        LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE,
        REMOTE_AUGMENTATION_MANIFEST_FILE,
    )

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )
    plan_one = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    plan_two = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )

    def summary(plan: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
        plan_list = list(plan)  # type: ignore[arg-type]
        adds = tuple(
            sorted(op.path_in_repo for op in plan_list if op.action == "add")  # type: ignore[attr-defined]
        )
        deletes = tuple(
            sorted(op.path_in_repo for op in plan_list if op.action == "delete")  # type: ignore[attr-defined]
        )
        return adds, deletes

    adds_one, deletes_one = summary(plan_one)
    adds_two, deletes_two = summary(plan_two)
    assert adds_one == adds_two
    assert deletes_one == deletes_two
    assert REMOTE_AUGMENTATION_MANIFEST_FILE in adds_one
    assert LEGACY_REMOTE_AUGMENTATION_MANIFEST_FILE in deletes_one

def test_publication_plan_is_deterministic_and_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry in the publication plan is unique and the unified
    plan contains exactly the documented core + aug + migration ops.
    """
    from osm_polygon_wikidata_only.hf.publication import assemble_region_upload

    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_generators(monkeypatch)
    plan = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    ops = list(plan)
    additions = [op for op in ops if op.action == "add"]
    deletions = [op for op in ops if op.action == "delete"]
    paths = [op.path_in_repo for op in ops]
    assert len(paths) == len(set(paths))
    assert sum(op.path_in_repo == "manifests/augmentation_manifest.json" for op in additions) == 1
    assert (
        sum(
            op.path_in_repo == "augmentation/manifests/augmentation_manifest.json"
            for op in deletions
        )
        == 1
    )
    assert sum(op.path_in_repo == "coverage_map.png" for op in deletions) == 1
    # Three current maps are added while the root coverage map and both
    # superseded H3 maps are deleted in the same atomic publication.
    assert len(ops) == 20, f"unexpected plan length: {len(ops)} {ops}"

def test_stub_records_add_and_delete_ops(tmp_path: Path) -> None:
    """The HF stub must record both addition and deletion ops in the
    commits list, so tests can assert migration commit semantics.
    """
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.plan import (
        add_op,
        delete_op,
    )
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    canonical = tmp_path / "canonical.json"
    canonical.write_text("{}", encoding="utf-8")

    hub = StubHfHub()
    ops_mod.upload_files(
        REPO_ID,
        ops=[
            add_op(
                canonical,
                path_in_repo="manifests/augmentation_manifest.json",
            ),
            delete_op("augmentation/manifests/augmentation_manifest.json"),
        ],
        hub=hub,
        token="stub-token",
        commit_message="migrate",
    )
    # stub recorded exactly one commit
    assert len(hub.commits) == 1
    commit = hub.commits[0]
    # The commit plan includes BOTH the canonical add and the legacy
    # delete in the same plan (atomicity).
    assert set(commit["paths"]) == {
        "manifests/augmentation_manifest.json",
        "augmentation/manifests/augmentation_manifest.json",
    }
    assert "uploads_made" in commit or len(commit["operations"]) == 2, (
        "Stub commit record must expose every op"
    )

def test_upload_files_rejects_legacy_path_only_when_canonical_missing() -> None:
    """Safety net: a publication commit that deletes the legacy path
    without uploading the canonical replacement must be refused.
    """
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    hub = StubHfHub()
    with pytest.raises(UploadError, match="canonical"):
        ops_mod.upload_files(
            REPO_ID,
            ops=[delete_op("augmentation/manifests/augmentation_manifest.json")],
            hub=hub,
            token="stub-token",
            commit_message="dangling delete",
        )

def test_upload_files_deletion_is_idempotent_when_remote_missing(tmp_path: Path) -> None:
    """Deleting a remote path that doesn't exist (e.g. on a fresh
    dataset) must be a no-op, not an error.
    """
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.plan import (
        add_op,
        delete_op,
    )
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    canonical = tmp_path / "c.json"
    canonical.write_text("{}", encoding="utf-8")

    hub = StubHfHub(remote_files=set())
    # Hugging Face rejects CommitOperationDelete when the target does
    # not exist. The uploader must therefore omit the legacy delete
    # after confirming that the remote path is already absent.
    plan = [
        add_op(canonical, path_in_repo="manifests/augmentation_manifest.json"),
        delete_op("augmentation/manifests/augmentation_manifest.json"),
    ]
    ops_mod.upload_files(
        REPO_ID,
        ops=plan,
        hub=hub,
        token="stub-token",
        commit_message="migration commit",
    )
    # Exactly one commit containing only the canonical addition.
    assert len(hub.commits) == 1
    assert hub.commits[0]["operations"] == [
        {
            "action": "add",
            "path_in_repo": "manifests/augmentation_manifest.json",
        }
    ]

def test_upload_files_rejects_legacy_coverage_map_only_when_canonical_missing() -> None:
    """Safety net: a publication commit that deletes the legacy coverage map
    without uploading the canonical replacement must be refused.
    """
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    hub = StubHfHub()
    with pytest.raises(UploadError, match="canonical"):
        ops_mod.upload_files(
            REPO_ID,
            ops=[delete_op("coverage_map.png")],
            hub=hub,
            token="stub-token",
            commit_message="dangling delete",
        )

@pytest.mark.parametrize(
    "legacy_path",
    [
        "assets/geographic_wikipedia_text_coverage.png",
        "assets/geographic_polygon_count.png",
    ],
)
def test_upload_files_rejects_old_h3_map_delete_without_text_density(
    legacy_path: str,
) -> None:
    """Both retired H3 assets require the new density map in the commit."""
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    with pytest.raises(UploadError, match="canonical"):
        ops_mod.upload_files(
            REPO_ID,
            ops=[delete_op(legacy_path)],
            hub=StubHfHub(),
            token="stub-token",
            commit_message="unsafe map retirement",
        )

def test_upload_files_rejects_legacy_article_without_lossless_replacement() -> None:
    """A legacy article cannot be deleted without the same-stem document add."""
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
    from osm_polygon_wikidata_only.hf._uploader.plan import delete_op
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    with pytest.raises(UploadError, match="canonical replacement"):
        ops_mod.upload_files(
            REPO_ID,
            ops=[delete_op("articles/monaco-latest.parquet")],
            hub=StubHfHub(),
            token="stub-token",
            commit_message="unsafe retirement",
        )

def test_upload_files_pairs_article_retirement_by_exact_stem(tmp_path: Path) -> None:
    """A replacement for another stem cannot authorize deletion."""
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError
    from osm_polygon_wikidata_only.hf._uploader.plan import add_op, delete_op
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    replacement = tmp_path / "andorra-latest.parquet"
    replacement.touch()
    with pytest.raises(UploadError, match="monaco-latest"):
        ops_mod.upload_files(
            REPO_ID,
            ops=[
                add_op(
                    replacement,
                    path_in_repo="wikipedia/documents/andorra-latest.parquet",
                ),
                delete_op("articles/monaco-latest.parquet"),
            ],
            hub=StubHfHub(),
            token="stub-token",
            commit_message="wrong replacement",
        )

def test_upload_files_coverage_map_deletion_is_idempotent_when_remote_missing(
    tmp_path: Path,
) -> None:
    """Deleting a remote coverage map that doesn't exist (e.g. after migration)
    must be a no-op, not an error.
    """
    from osm_polygon_wikidata_only.hf._uploader import operations as ops_mod
    from osm_polygon_wikidata_only.hf._uploader.plan import (
        add_op,
        delete_op,
    )
    from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub

    canonical = tmp_path / "c.png"
    canonical.touch()

    hub = StubHfHub(remote_files=set())
    plan = [
        add_op(canonical, path_in_repo="assets/coverage_map.png"),
        delete_op("coverage_map.png"),
    ]
    ops_mod.upload_files(
        REPO_ID,
        ops=plan,
        hub=hub,
        token="stub-token",
        commit_message="migration commit",
    )
    # Exactly one commit containing only the canonical addition.
    assert len(hub.commits) == 1
    assert hub.commits[0]["operations"] == [
        {
            "action": "add",
            "path_in_repo": "assets/coverage_map.png",
        }
    ]

def test_core_upload_defers_repository_wide_assets_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory run publishes region data only; assets come once at the end."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_generators(monkeypatch)
    rendered: list[str] = []
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: rendered.append("README.md"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_polygon_stats_snapshot",
        lambda *a, **kw: rendered.append("stats.json"),
    )

    ops = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
        defer_metadata_assets=True,
    )

    assert [op.path_in_repo for op in ops] == [
        "polygons/monaco-latest.parquet",
        "wikipedia/documents/monaco-latest.parquet",
        "articles/monaco-latest.parquet",
        "polygon_articles/monaco-latest.parquet",
        "manifests/processed_pbfs.json",
    ]
    # Deferred assets are not rendered either: the deferred refresh scans once.
    assert rendered == []
