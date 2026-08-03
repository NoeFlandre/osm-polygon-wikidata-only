from pathlib import Path


def test_sync_dir_has_explicit_v2_selector() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", "/tmp/raw", "--dataset-version", "v2"])
    assert args.dataset_version == "v2"


def test_v2_selector_defaults_to_v1() -> None:
    from osm_polygon_wikidata_only.cli.parser import build_parser

    args = build_parser().parse_args(["sync-dir", str(Path("/tmp/raw"))])
    assert args.dataset_version == "v1"


def test_v2_selector_routes_to_isolated_executor(tmp_path: Path, monkeypatch) -> None:
    import osm_polygon_wikidata_only.v2.cli as v2_cli
    from osm_polygon_wikidata_only.cli.commands import main

    calls: list[str] = []
    monkeypatch.setattr(v2_cli, "execute_v2", lambda *args, **kwargs: calls.append("v2") or 0)
    raw = tmp_path / "raw"
    raw.mkdir()
    assert (
        main(["sync-dir", str(raw), "--data-root", str(tmp_path), "--dataset-version", "v2"]) == 0
    )
    assert calls == ["v2"]
