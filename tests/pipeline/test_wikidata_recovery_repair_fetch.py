from __future__ import annotations

from typing import Any, cast

import pytest

from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity
from osm_polygon_wikidata_only.pipeline._wikidata_recovery import repair_fetch
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.progress import RecoveryProgress


def test_fetch_qid_documents_skips_existing_and_collects_new_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = WikidataEntity(
        "Q1",
        {"enwiki": "Existing", "frwiki": "New", "dewiki": "Missing"},
    )
    settings = Settings(
        languages=None,
        fetch_full_text=True,
        max_articles_per_qid=None,
        enrichment_batch_size=50,
    )
    calls: list[str] = []

    def fetch(_qid: str, _site: str, title: str, **_kwargs: Any) -> dict[str, str] | None:
        calls.append(title)
        return {"document_id": title} if title == "New" else None

    class Progress:
        advanced = 0

        def advance(self) -> None:
            self.advanced += 1

    progress = Progress()
    monkeypatch.setattr(repair_fetch, "_fetch_recovery_document", fetch)

    documents = repair_fetch._fetch_qid_documents(
        "Q1",
        entity=entity,
        existing={("Q1", "enwiki", "Existing")},
        wikipedia_client=cast(Any, object()),
        settings=settings,
        progress=cast(RecoveryProgress, progress),
    )

    assert documents == [{"document_id": "New"}]
    assert calls == ["Missing", "New"]
    assert progress.advanced == 1
