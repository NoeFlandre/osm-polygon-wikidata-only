"""Checkout detection used by data-root safety checks and asset lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config import paths
from osm_polygon_wikidata_only.config.paths import (
    DataRootError,
    repository_root,
    resolve_data_root,
)
from osm_polygon_wikidata_only.hf import repo_layout

CHECKOUT = Path(__file__).resolve().parents[1]


@pytest.mark.repository
def test_source_checkout_is_detected() -> None:
    assert repository_root() == CHECKOUT


def test_installed_package_has_no_repository_root(tmp_path: Path) -> None:
    module = tmp_path / "lib/python3.12/site-packages/osm_polygon_wikidata_only/config/paths.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert repository_root(module) is None


def test_src_layout_without_checkout_markers_is_not_a_checkout(tmp_path: Path) -> None:
    module = tmp_path / "src/osm_polygon_wikidata_only/config/paths.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert repository_root(module) is None
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src/osm_polygon_wikidata_only/__init__.py").write_text("")
    assert repository_root(module) == tmp_path


def test_shallow_path_is_not_a_checkout() -> None:
    assert repository_root("/paths.py") is None


def test_data_root_inside_checkout_is_still_refused(tmp_path: Path) -> None:
    inner = tmp_path / "data"
    inner.mkdir()
    with pytest.raises(DataRootError, match="inside the repository"):
        resolve_data_root(inner, repo_root=tmp_path)


def test_installed_package_skips_the_repository_containment_check(tmp_path: Path) -> None:
    assert resolve_data_root(tmp_path, repo_root=None).path == tmp_path


def test_installed_package_uses_packaged_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo_layout, "repository_root", lambda: None)
    packaged = Path(repo_layout.__file__).resolve().parents[1]
    assert repo_layout._local_asset("assets/x.png") == packaged / "assets/x.png"


@pytest.mark.repository
def test_checkout_prefers_repository_assets() -> None:
    assert repo_layout.LOCAL_DATASET_HERO_FILE == CHECKOUT / "assets/dataset_hero.png"
    assert repo_layout.LOCAL_V2_DATASET_HERO_FILE == CHECKOUT / "assets/dataset_hero_v2.png"
    assert paths.repository_root() is not None
