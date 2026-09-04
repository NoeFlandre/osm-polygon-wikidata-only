"""Resumable materialization of V2 sentence sidecars."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_replacement, atomic_write_json
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.sentence_checkpoints import SentenceCheckpoint
from osm_polygon_wikidata_only.v2.sentence_logic import (
    SAT_MODEL_ID,
    SAT_MODEL_NAME,
    SAT_SUPPORTED_LANGUAGES,
    SentenceSegmenter,
    is_sat_supported_language,
    sentence_schema,
    split_sections,
)
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

DEFAULT_SENTENCE_BATCH_SIZE = 256
SENTENCE_CHECKPOINT_DIRECTORY = "sentence-checkpoints"
SENTENCE_MANIFEST_RELATIVE_PATH = Path("manifests/sentence_splitting.json")
SENTENCE_SPLIT_CONTRACT_VERSION = "v2-sentence-splitting-v1"


@dataclass(frozen=True, slots=True)
class SentenceRegionSummary:
    """Deterministic accounting for one region/project output."""

    stem: str
    project: str
    sections: int
    split_sections: int
    unsplit_sections: int
    sentence_rows: int
    supported_languages: tuple[str, ...]
    unsupported_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SentenceRunResult:
    """Final manifest path and summaries emitted by one sentence run."""

    manifest_path: Path
    regions: tuple[SentenceRegionSummary, ...]


@dataclass
class _Counters:
    sections: int = 0
    split_sections: int = 0
    unsplit_sections: int = 0
    sentence_rows: int = 0
    supported_languages: set[str] = field(default_factory=set)
    unsupported_languages: set[str] = field(default_factory=set)

    def add_sections(self, sections: Sequence[dict[str, Any]]) -> None:
        for section in sections:
            language = str(section.get("language") or "")
            self.sections += 1
            if is_sat_supported_language(language):
                self.split_sections += 1
                self.supported_languages.add(language)
            else:
                self.unsplit_sections += 1
                self.unsupported_languages.add(language)

    def summary(self, *, stem: str, project: str) -> SentenceRegionSummary:
        return SentenceRegionSummary(
            stem=stem,
            project=project,
            sections=self.sections,
            split_sections=self.split_sections,
            unsplit_sections=self.unsplit_sections,
            sentence_rows=self.sentence_rows,
            supported_languages=tuple(sorted(self.supported_languages)),
            unsupported_languages=tuple(sorted(self.unsupported_languages)),
        )


@dataclass(frozen=True, slots=True)
class _ProcessedSource:
    """Typed accounting returned after one source table is processed."""

    batch_count: int
    row_count: int
    summary: SentenceRegionSummary


def run_v2_sentence_split(
    data_root: DataRoot,
    *,
    segmenter: SentenceSegmenter,
    batch_size: int = DEFAULT_SENTENCE_BATCH_SIZE,
    stems: Sequence[str] | None = None,
) -> SentenceRunResult:
    """Split all selected finalized V2 section tables with restart-safe batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if segmenter.model_id != SAT_MODEL_ID:
        raise ValueError(f"Only {SAT_MODEL_NAME} is supported by this stage")

    manifest = load_v2_manifest(data_root.processed_v2)
    selected_stems = _selected_stems(manifest, stems)
    summaries: list[SentenceRegionSummary] = []
    for stem in selected_stems:
        summaries.append(
            _run_project(
                data_root,
                stem=stem,
                project="wikipedia",
                segmenter=segmenter,
                batch_size=batch_size,
            )
        )
        wikivoyage_source = _section_path(data_root.processed_v2, stem, "wikivoyage")
        if wikivoyage_source.is_file():
            summaries.append(
                _run_project(
                    data_root,
                    stem=stem,
                    project="wikivoyage",
                    segmenter=segmenter,
                    batch_size=batch_size,
                )
            )

    manifest_path = data_root.processed_v2 / SENTENCE_MANIFEST_RELATIVE_PATH
    _write_manifest(manifest_path, segmenter=segmenter, summaries=summaries)
    return SentenceRunResult(manifest_path=manifest_path, regions=tuple(summaries))


def _selected_stems(
    manifest: dict[str, dict[str, Any]], stems: Sequence[str] | None
) -> tuple[str, ...]:
    if not manifest:
        raise FileNotFoundError("No finalized V2 regions found in manifests/processed_pbfs.json")
    available = set(manifest)
    selected = _requested_stems(manifest, stems)
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Sentence split region(s) are not finalized in V2: {missing}")
    if not selected:
        raise ValueError("At least one V2 region is required for sentence splitting")
    for stem in selected:
        _validate_stem(stem)
    return selected


def _requested_stems(
    manifest: Mapping[str, object], stems: Sequence[str] | None
) -> tuple[str, ...]:
    return tuple(sorted(set(manifest if stems is None else stems)))


def _run_project(
    data_root: DataRoot,
    *,
    stem: str,
    project: str,
    segmenter: SentenceSegmenter,
    batch_size: int,
) -> SentenceRegionSummary:
    source_path = _section_path(data_root.processed_v2, stem, project)
    if not source_path.is_file():
        raise FileNotFoundError(f"V2 section source is missing: {source_path}")
    _validate_schema(source_path, section_schema())
    input_fingerprint = sha256_file(source_path)
    checkpoint = SentenceCheckpoint(
        data_root.v2_cache / SENTENCE_CHECKPOINT_DIRECTORY,
        stem,
        project,
        input_fingerprint=input_fingerprint,
        model_id=segmenter.model_id,
        model_revision=_model_revision(segmenter),
        batch_size=batch_size,
    )
    output_path = _sentence_path(data_root.processed_v2, stem, project)
    if _recorded_output_matches(checkpoint, output_path):
        return _summary_from_checkpoint(checkpoint)

    processed = _process_source(
        source_path,
        checkpoint=checkpoint,
        segmenter=segmenter,
        batch_size=batch_size,
        stem=stem,
        project=project,
    )
    checkpoint.mark_complete(
        batch_count=processed.batch_count,
        row_count=processed.row_count,
    )
    _write_output(output_path, checkpoint, batch_count=processed.batch_count)
    summary = processed.summary
    checkpoint.finalize(
        output_path,
        output_hash=sha256_file(output_path),
        summary=asdict(summary),
    )
    return summary


def _process_source(
    source_path: Path,
    *,
    checkpoint: SentenceCheckpoint,
    segmenter: SentenceSegmenter,
    batch_size: int,
    stem: str,
    project: str,
) -> _ProcessedSource:
    counters = _Counters()
    batch_count = 0
    row_count = 0
    with pq.ParquetFile(source_path) as parquet_file:
        for batch_index, record_batch in enumerate(
            parquet_file.iter_batches(batch_size=batch_size)
        ):
            sections = record_batch.to_pylist()
            counters.add_sections(sections)
            batch_row_count = checkpoint.batch_row_count(batch_index)
            if batch_row_count is None:
                rows, _ = split_sections(
                    sections,
                    segmenter=segmenter,
                    batch_size=batch_size,
                )
                checkpoint.write_batch(batch_index, rows)
                row_count += len(rows)
            else:
                row_count += batch_row_count
            batch_count = batch_index + 1
    counters.sentence_rows = row_count
    return _ProcessedSource(
        batch_count=batch_count,
        row_count=row_count,
        summary=counters.summary(stem=stem, project=project),
    )


def _recorded_output_matches(checkpoint: SentenceCheckpoint, output_path: Path) -> bool:
    output_hash = checkpoint.metadata.get("output_hash")
    return isinstance(output_hash, str) and checkpoint.output_matches(
        output_path,
        output_hash=output_hash,
    )


def _summary_from_mapping(
    raw: Mapping[str, object],
    *,
    error: str,
) -> SentenceRegionSummary:
    try:
        return SentenceRegionSummary(
            stem=str(raw["stem"]),
            project=str(raw["project"]),
            sections=int(cast(Any, raw["sections"])),
            split_sections=int(cast(Any, raw["split_sections"])),
            unsplit_sections=int(cast(Any, raw["unsplit_sections"])),
            sentence_rows=int(cast(Any, raw["sentence_rows"])),
            supported_languages=tuple(
                str(value) for value in cast(Any, raw["supported_languages"])
            ),
            unsupported_languages=tuple(
                str(value) for value in cast(Any, raw["unsupported_languages"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc


def _manifest_metadata(segmenter: SentenceSegmenter) -> dict[str, str]:
    return {
        "contract_version": SENTENCE_SPLIT_CONTRACT_VERSION,
        "segmenter": SAT_MODEL_NAME,
        "model_id": segmenter.model_id,
        "model_revision": _model_revision(segmenter),
        "segmenter_version": segmenter.version,
    }


def _manifest_region_summary(
    region: object,
    *,
    path: Path,
) -> SentenceRegionSummary:
    if not isinstance(region, dict):
        raise ValueError(f"Invalid sentence manifest region: {path}")
    return _summary_from_mapping(
        cast(Mapping[str, object], region),
        error=f"Invalid sentence manifest region: {path}",
    )


def _manifest_payload(
    segmenter: SentenceSegmenter,
    summaries: Sequence[SentenceRegionSummary],
) -> dict[str, object]:
    unsupported_languages = sorted(
        {language for summary in summaries for language in summary.unsupported_languages}
    )
    return {
        **_manifest_metadata(segmenter),
        "supported_languages": list(SAT_SUPPORTED_LANGUAGES),
        "unsupported_languages": unsupported_languages,
        "unsupported_language_policy": "one unsplit row; never passed to SaT",
        "regions": [asdict(summary) for summary in summaries],
    }


def _load_manifest_payload(path: Path) -> dict[str, object]:
    try:
        payload = json_loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid sentence manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sentence manifest: {path}")
    return payload


def _manifest_uses_segmenter(
    payload: Mapping[str, object],
    segmenter: SentenceSegmenter,
) -> bool:
    return all(payload.get(key) == value for key, value in _manifest_metadata(segmenter).items())


def _summary_from_checkpoint(checkpoint: SentenceCheckpoint) -> SentenceRegionSummary:
    raw = checkpoint.metadata.get("summary")
    if not isinstance(raw, dict):
        raise ValueError("Completed sentence checkpoint has no summary")
    return _summary_from_mapping(
        raw,
        error="Completed sentence checkpoint has an invalid summary",
    )


def _write_output(output_path: Path, checkpoint: SentenceCheckpoint, *, batch_count: int) -> None:
    schema = sentence_schema()
    with atomic_replacement(output_path) as temporary:
        with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
            _write_checkpoint_batches(writer, checkpoint, batch_count=batch_count)
        _validate_schema(temporary, schema)


def _write_checkpoint_batches(
    writer: pq.ParquetWriter,
    checkpoint: SentenceCheckpoint,
    *,
    batch_count: int,
) -> None:
    for batch_index in range(batch_count):
        table = checkpoint.load_batch_table(batch_index)
        if table is None:
            raise ValueError(f"Invalid sentence checkpoint batch: {batch_index}")
        if table.num_rows:
            writer.write_table(table)


def _write_manifest(
    path: Path,
    *,
    segmenter: SentenceSegmenter,
    summaries: Sequence[SentenceRegionSummary],
) -> None:
    merged = {
        (summary.stem, summary.project): summary
        for summary in _load_manifest_summaries(path, segmenter=segmenter)
    }
    merged.update({(summary.stem, summary.project): summary for summary in summaries})
    ordered_summaries = [merged[key] for key in sorted(merged)]
    atomic_write_json(path, _manifest_payload(segmenter, ordered_summaries))


def _load_manifest_summaries(
    path: Path,
    *,
    segmenter: SentenceSegmenter,
) -> tuple[SentenceRegionSummary, ...]:
    if not path.is_file():
        return ()
    payload = _load_manifest_payload(path)
    if not _manifest_uses_segmenter(payload, segmenter):
        raise ValueError(f"Sentence manifest uses a different segmenter: {path}")
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ValueError(f"Invalid sentence manifest regions: {path}")
    return tuple(_manifest_region_summary(region, path=path) for region in regions)


def _model_revision(segmenter: SentenceSegmenter) -> str:
    return str(getattr(segmenter, "revision", "unknown"))


def _section_path(processed_v2: Path, stem: str, project: str) -> Path:
    return processed_v2 / project / "sections" / f"{stem}.parquet"


def _sentence_path(processed_v2: Path, stem: str, project: str) -> Path:
    return processed_v2 / project / "sentences" / f"{stem}.parquet"


def _validate_schema(path: Path, schema: pa.Schema) -> None:
    with pq.ParquetFile(path) as parquet_file:
        if not parquet_file.schema_arrow.equals(schema, check_metadata=True):
            raise ValueError(f"Unexpected Parquet schema: {path}")


def _validate_stem(stem: str) -> None:
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid V2 stem: {stem!r}")


__all__ = [
    "DEFAULT_SENTENCE_BATCH_SIZE",
    "SENTENCE_CHECKPOINT_DIRECTORY",
    "SENTENCE_MANIFEST_RELATIVE_PATH",
    "SENTENCE_SPLIT_CONTRACT_VERSION",
    "SentenceRegionSummary",
    "SentenceRunResult",
    "run_v2_sentence_split",
]
