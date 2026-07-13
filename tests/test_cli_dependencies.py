"""Tests for construction of the shared Wikimedia runtime."""

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.cli.dependencies import build_wikimedia_runtime
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.io.cache import JsonFileCache


def test_authenticated_runtime_shares_1200_rpm_three_slot_scheduler(tmp_path) -> None:
    runtime = build_wikimedia_runtime(
        Settings(),
        data_root=DataRoot(tmp_path),
        environ={
            "WIKIMEDIA_BOT_USERNAME": "User@pipeline",
            "WIKIMEDIA_BOT_PASSWORD": "secret",
        },
    )
    augmentation = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(tmp_path / "augmentation"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )

    assert runtime.scheduler.max_in_flight == 3
    assert runtime.scheduler.current_requests_per_minute == 1_200
    assert augmentation._scheduler is runtime.scheduler
    assert augmentation._session is runtime.session
