"""Prepare a deterministic, link-backed geographic NER pilot shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_json, atomic_write_parquet
from osm_polygon_wikidata_only.ner.pipeline import INPUT_COLUMNS, Contract

TARGET_LANGUAGES = ("ar", "de", "en", "es", "fr", "hy", "ja", "pt", "ru", "zh")
_PROJECTS = ("wikipedia", "wikivoyage")
_EXTRA_LANGUAGE_ORDER = ("de", "en", "es", "fr", "ru", "zh", "ar", "hy", "ja", "pt")
_REDISTRIBUTION_LANGUAGE_ORDER = ("en", "fr", "de", "es", "ru", "zh", "ja", "pt")
_DEFAULT_SAMPLE_SIZE = 100
_DEFAULT_SEED = "geographic-ner-pilot-v1"
_DEFAULT_MAX_CANDIDATE_TABLES = 256
_DEFAULT_MAX_CANDIDATE_BATCHES = 512
_MIN_PILOT_REGION_STEMS = 8
_MAX_PILOT_REGION_STEMS = 16
_MIN_DOCUMENTS_PER_GROUP = 2
_OPTIONAL_POSITION_COLUMNS = ("section_index", "sentence_index")


def build_contract() -> Contract:
    """Return the pinned, explicitly unvalidated pilot contract."""
    return Contract(languages=TARGET_LANGUAGES)


def select_rows(
    data_root: Path,
    *,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    seed: str = _DEFAULT_SEED,
    max_candidate_tables: int = _DEFAULT_MAX_CANDIDATE_TABLES,
    max_candidate_batches: int = _DEFAULT_MAX_CANDIDATE_BATCHES,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select a deterministic, balanced sample from completed split sentences."""
    _validate_selection_inputs(sample_size, seed, max_candidate_tables, max_candidate_batches)
    root = _processed_root(Path(data_root))
    sentence_files = tuple(islice(_sentence_files(root), max_candidate_tables))
    link_pairs, link_triples = _read_link_keys(root, sentence_files)
    candidates, capacities, source_files, split_rows, split_projects = _read_candidates(
        root,
        sentence_files,
        link_pairs,
        link_triples,
        sample_size=sample_size,
        seed=seed,
        max_candidate_tables=max_candidate_tables,
        max_candidate_batches=max_candidate_batches,
    )
    _validate_split_coverage(split_rows, split_projects)
    if not candidates:
        raise ValueError("No link-backed pilot rows are available")
    quotas = _quotas(candidates, capacities, sample_size=sample_size, seed=seed)
    selected = _select_group_rows(candidates, quotas)
    rows = [_input_row(row) for row in _ordered_rows(selected, seed)]
    if len(rows) != sample_size:
        raise ValueError("Pilot selection did not produce the requested sample size")
    return rows, _selection_manifest(
        selected,
        rows,
        source_files=source_files,
        sample_size=sample_size,
        seed=seed,
    )


def _validate_selection_inputs(
    sample_size: int, seed: str, max_candidate_tables: int, max_candidate_batches: int
) -> None:
    _validate_positive_integer(sample_size, "Pilot sample_size must be a positive integer")
    if not isinstance(seed, str) or not seed:
        raise ValueError("Pilot seed must be a non-empty string")
    _validate_positive_integer(
        max_candidate_tables, "Pilot max_candidate_tables must be a positive integer"
    )
    _validate_positive_integer(
        max_candidate_batches, "Pilot max_candidate_batches must be a positive integer"
    )


def _validate_positive_integer(value: object, message: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(message)


def _validate_split_coverage(split_rows: int, split_projects: set[str]) -> None:
    if split_rows == 0 or set(split_projects) != set(_PROJECTS):
        raise ValueError("No completed split rows are available for the pilot")


def _processed_root(data_root: Path) -> Path:
    candidate = data_root / "processed_v2"
    if candidate.is_dir():
        return candidate
    if data_root.name == "processed_v2" and data_root.is_dir():
        return data_root
    raise ValueError(f"processed_v2 directory is missing: {candidate}")


def _read_link_keys(
    root: Path, sentence_files: Sequence[tuple[str, Path]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    paths = _selected_link_paths(root, sentence_files)
    if not paths:
        raise ValueError("No polygon-document link files are available")
    pairs: set[tuple[str, str]] = set()
    triples: set[tuple[str, str, str]] = set()
    for path in paths:
        _read_link_file(path, pairs, triples)
    return pairs, triples


def _selected_link_paths(root: Path, sentence_files: Sequence[tuple[str, Path]]) -> list[Path]:
    paths = {root / "polygon_document_links" / f"{path.stem}.parquet" for _, path in sentence_files}
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise ValueError(f"Selected sentence table has no link file: {missing[0]}")
    return sorted(paths)


def _read_link_file(
    path: Path, pairs: set[tuple[str, str]], triples: set[tuple[str, str, str]]
) -> None:
    columns = _available_columns(path, ("document_id", "project", "language"))
    required = {"document_id", "project"}
    if not required.issubset(columns):
        raise ValueError(f"Link file is missing document_id/project columns: {path}")
    selected = sorted(required | ({"language"} & columns))
    for batch in _batches(path, selected):
        for row in batch.to_pylist():
            _add_link_row(row, pairs, triples)


def _add_link_row(
    row: dict[str, Any],
    pairs: set[tuple[str, str]],
    triples: set[tuple[str, str, str]],
) -> None:
    document_id = row.get("document_id")
    project = row.get("project")
    if not isinstance(document_id, str) or not isinstance(project, str):
        return
    pairs.add((document_id, project))
    language = row.get("language")
    if isinstance(language, str):
        triples.add((document_id, project, language))


def _read_candidates(
    root: Path,
    sentence_files: Sequence[tuple[str, Path]],
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
    *,
    sample_size: int,
    seed: str,
    max_candidate_tables: int,
    max_candidate_batches: int,
) -> tuple[
    dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    dict[tuple[str, str], int],
    list[str],
    int,
    set[str],
]:
    state = _CandidateState()
    requested = set((*INPUT_COLUMNS, *_OPTIONAL_POSITION_COLUMNS))
    for project, path in sentence_files:
        _read_candidate_table(
            root,
            project,
            path,
            state,
            requested,
            link_pairs,
            link_triples,
            sample_size=sample_size,
            seed=seed,
            max_candidate_tables=max_candidate_tables,
            max_candidate_batches=max_candidate_batches,
        )
        if _candidate_scan_ready(state.candidates, state.capacities, sample_size, seed):
            break
    if not _candidate_scan_ready(state.candidates, state.capacities, sample_size, seed):
        _raise_candidate_scan_cap(state, max_candidate_tables, max_candidate_batches)
    return _candidate_result(state)


def _raise_candidate_scan_cap(
    state: _CandidateState, max_candidate_tables: int, max_candidate_batches: int
) -> None:
    if state.tables_read >= max_candidate_tables or state.batches_read >= max_candidate_batches:
        raise ValueError("Pilot candidate scan cap reached before quotas were filled")


@dataclass
class _CandidateState:
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    capacities: Counter[tuple[str, str]] = field(default_factory=Counter)
    source_files: list[str] = field(default_factory=list)
    seen: set[tuple[str, str, str, str]] = field(default_factory=set)
    split_rows: int = 0
    split_projects: set[str] = field(default_factory=set)
    tables_read: int = 0
    batches_read: int = 0


def _read_candidate_table(
    root: Path,
    project: str,
    path: Path,
    state: _CandidateState,
    requested: set[str],
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
    *,
    sample_size: int,
    seed: str,
    max_candidate_tables: int,
    max_candidate_batches: int,
) -> None:
    if state.tables_read >= max_candidate_tables:
        raise ValueError("Pilot candidate scan cap reached before quotas were filled")
    state.tables_read += 1
    state.source_files.append(path.relative_to(root).as_posix())
    columns = _available_columns(path, tuple(requested))
    _require_candidate_columns(path, columns)
    _read_candidate_batches(
        path,
        sorted(columns),
        project,
        state,
        link_pairs,
        link_triples,
        sample_size=sample_size,
        seed=seed,
        max_candidate_batches=max_candidate_batches,
    )


def _require_candidate_columns(path: Path, columns: set[str]) -> None:
    missing = set(INPUT_COLUMNS) - columns
    if missing:
        raise ValueError(f"Sentence file is missing required columns: {path}")


def _read_candidate_batches(
    path: Path,
    columns: Sequence[str],
    project: str,
    state: _CandidateState,
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
    *,
    sample_size: int,
    seed: str,
    max_candidate_batches: int,
) -> None:
    for batch in _batches(path, columns):
        if state.batches_read >= max_candidate_batches:
            raise ValueError("Pilot candidate scan cap reached before quotas were filled")
        state.batches_read += 1
        _read_candidate_batch(
            batch,
            project,
            state,
            link_pairs,
            link_triples,
            sample_size=sample_size,
            seed=seed,
        )
        if _candidate_scan_ready(state.candidates, state.capacities, sample_size, seed):
            return


def _read_candidate_batch(
    batch: pa.RecordBatch,
    project: str,
    state: _CandidateState,
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
    *,
    sample_size: int,
    seed: str,
) -> None:
    for row in batch.to_pylist():
        _read_candidate_row(
            row,
            project,
            state,
            link_pairs,
            link_triples,
            sample_size=sample_size,
            seed=seed,
        )


def _read_candidate_row(
    row: dict[str, Any],
    project: str,
    state: _CandidateState,
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
    *,
    sample_size: int,
    seed: str,
) -> None:
    if row.get("segmentation_status") != "split":
        return
    state.split_rows += 1
    state.split_projects.add(project)
    if not _candidate_row(row, project, link_pairs, link_triples):
        return
    key = _row_key(row)
    if key in state.seen:
        return
    state.seen.add(key)
    group = (project, str(row["language"]))
    state.capacities[group] += 1
    _retain_candidate(state.candidates[group], _score(seed, _identity(row)), row, sample_size)


def _candidate_result(
    state: _CandidateState,
) -> tuple[
    dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    dict[tuple[str, str], int],
    list[str],
    int,
    set[str],
]:
    return (
        dict(state.candidates),
        dict(state.capacities),
        state.source_files,
        state.split_rows,
        state.split_projects,
    )


def _candidate_scan_ready(
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    capacities: dict[tuple[str, str], int],
    sample_size: int,
    seed: str,
) -> bool:
    try:
        quotas = _quotas(candidates, capacities, sample_size=sample_size, seed=seed)
    except ValueError:
        return False
    return _quotas_are_diverse(candidates, quotas)


def _quotas_are_diverse(
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    quotas: dict[tuple[str, str], int],
) -> bool:
    for group, quota in quotas.items():
        rows = candidates.get(group, [])
        if len(rows) < quota:
            return False
        documents = {row["document_id"] for _, row in rows[:quota]}
        if len(documents) < min(quota, _MIN_DOCUMENTS_PER_GROUP):
            return False
    return True


def _sentence_files(root: Path) -> Iterable[tuple[str, Path]]:
    manifest_path = root / "manifests" / "sentence_splitting.json"
    if manifest_path.is_file():
        yield from _manifest_sentence_files(root, manifest_path)
        return
    yield from _fallback_sentence_files(root)


def _manifest_sentence_files(root: Path, manifest_path: Path) -> Iterable[tuple[str, Path]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    regions = raw.get("regions") if isinstance(raw, dict) else None
    if not isinstance(regions, list):
        raise ValueError(f"Sentence manifest has no regions: {manifest_path}")
    by_stem = _manifest_regions_by_stem(root, regions)
    for stem in _select_pilot_stems(by_stem):
        yield from _stem_files(by_stem[stem])


def _stem_files(regions: dict[str, tuple[str, Path, frozenset[str]]]) -> Iterable[tuple[str, Path]]:
    for project in _PROJECTS:
        if project in regions:
            yield regions[project][:2]


def _manifest_regions_by_stem(
    root: Path, regions: list[object]
) -> dict[str, dict[str, tuple[str, Path, frozenset[str]]]]:
    by_stem: dict[str, dict[str, tuple[str, Path, frozenset[str]]]] = defaultdict(dict)
    for region in regions:
        project, stem = _manifest_region_identity(region)
        if not _region_supports_target_languages(region):
            continue
        by_stem[stem][project] = (
            *(_manifest_region_path(root, region)),
            _region_target_languages(region),
        )
    return by_stem


def _region_supports_target_languages(region: object) -> bool:
    return bool(_region_target_languages(region))


def _region_target_languages(region: object) -> frozenset[str]:
    supported = _supported_region_languages(region)
    if supported is None:
        return frozenset(TARGET_LANGUAGES)
    return frozenset(
        value for value in supported if isinstance(value, str) and value in TARGET_LANGUAGES
    )


def _supported_region_languages(region: object) -> list[object] | None:
    if not isinstance(region, dict):
        return None
    supported = region.get("supported_languages")
    return cast(list[object], supported) if isinstance(supported, list) else None


def _select_pilot_stems(
    by_stem: dict[str, dict[str, tuple[str, Path, frozenset[str]]]],
) -> tuple[str, ...]:
    coverage = {stem: _stem_language_coverage(regions) for stem, regions in by_stem.items()}
    return _greedy_pilot_stems(coverage)


def _greedy_pilot_stems(coverage: dict[str, set[tuple[str, str]]]) -> tuple[str, ...]:
    required = set().union(*coverage.values()) if coverage else set()
    selected: list[str] = []
    remaining = set(coverage)
    while remaining and len(selected) < _MAX_PILOT_REGION_STEMS:
        stem = max(remaining, key=lambda value: (len(coverage[value] & required), value))
        selected.append(stem)
        required -= coverage[stem]
        remaining.remove(stem)
        if _pilot_stem_selection_is_complete(required, len(selected), len(coverage)):
            break
    return tuple(selected)


def _pilot_stem_selection_is_complete(
    required: set[tuple[str, str]], selected: int, total: int
) -> bool:
    return not required and selected >= min(_MIN_PILOT_REGION_STEMS, total)


def _stem_language_coverage(
    regions: dict[str, tuple[str, Path, frozenset[str]]],
) -> set[tuple[str, str]]:
    return {
        (project, language) for project, _, languages in regions.values() for language in languages
    }


def _manifest_region_path(root: Path, region: object) -> tuple[str, Path]:
    project, stem = _manifest_region_identity(region)
    path = root / project / "sentences" / f"{stem}.parquet"
    if not path.is_file():
        raise ValueError(f"Sentence manifest file is missing: {path}")
    return project, path


def _manifest_region_identity(region: object) -> tuple[str, str]:
    if not isinstance(region, dict):
        raise ValueError("Sentence manifest region is invalid")
    project = region.get("project")
    stem = region.get("stem")
    if not isinstance(project, str) or project not in _PROJECTS or not isinstance(stem, str):
        raise ValueError("Sentence manifest region has invalid project or stem")
    return project, stem


def _fallback_sentence_files(root: Path) -> Iterable[tuple[str, Path]]:
    for project in _PROJECTS:
        for path in sorted((root / project / "sentences").glob("*.parquet")):
            yield project, path


def _available_columns(path: Path, requested: Sequence[str]) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names) & set(requested)


def _batches(path: Path, columns: Sequence[str]) -> Iterable[pa.RecordBatch]:
    yield from pq.ParquetFile(path).iter_batches(batch_size=8192, columns=list(columns))


def _candidate_row(
    row: dict[str, Any],
    project: str,
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
) -> bool:
    if not _candidate_identity_matches(row, project):
        return False
    if not _candidate_fields_are_strings(row):
        return False
    document_id = row["document_id"]
    language = row["language"]
    return _candidate_is_linked(document_id, project, language, link_pairs, link_triples)


def _candidate_identity_matches(row: dict[str, Any], project: str) -> bool:
    return row.get("project") == project and row.get("language") in TARGET_LANGUAGES


def _candidate_fields_are_strings(row: dict[str, Any]) -> bool:
    return all(isinstance(row.get(column), str) for column in INPUT_COLUMNS)


def _candidate_is_linked(
    document_id: str,
    project: str,
    language: str,
    link_pairs: set[tuple[str, str]],
    link_triples: set[tuple[str, str, str]],
) -> bool:
    return (document_id, project) in link_pairs or (document_id, project, language) in link_triples


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["sentence_id"]),
        str(row["document_id"]),
        str(row["project"]),
        str(row["language"]),
    )


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return _row_key(row)


def _score(seed: str, identity: object) -> str:
    return hashlib.sha256(f"{seed}\0{identity!r}".encode()).hexdigest()


def _retain_candidate(
    retained: list[tuple[str, dict[str, Any]]],
    score: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    retained.append((score, row))
    retained.sort(key=lambda item: (item[0], _identity(item[1])))
    if len(retained) > limit:
        retained.pop()


def _quotas(
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    capacities: dict[tuple[str, str], int],
    *,
    sample_size: int,
    seed: str,
) -> dict[tuple[str, str], int]:
    if sum(capacities.values()) < sample_size:
        raise ValueError("Not enough link-backed rows for the requested pilot sample")
    groups_by_language: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for project, language in sorted(candidates):
        groups_by_language[language].append((project, language))
    targets = _language_targets(groups_by_language, capacities, sample_size)
    quotas = _stratified_quotas(targets, groups_by_language, capacities)
    _balance_projects(quotas, capacities, sample_size, seed)
    return quotas


def _language_targets(
    groups_by_language: dict[str, list[tuple[str, str]]],
    capacities: dict[tuple[str, str], int],
    sample_size: int,
) -> dict[str, int]:
    if set(groups_by_language) != set(TARGET_LANGUAGES):
        raise ValueError("Pilot source is missing one or more target languages")
    targets = _base_language_targets(sample_size)
    _cap_language_targets(targets, groups_by_language, capacities)
    _reduce_single_project_targets(targets, groups_by_language)
    _fill_language_shortfall(targets, groups_by_language, capacities, sample_size)
    return targets


def _base_language_targets(sample_size: int) -> dict[str, int]:
    base, remainder = divmod(sample_size, len(TARGET_LANGUAGES))
    targets = {language: base for language in TARGET_LANGUAGES}
    for language in _EXTRA_LANGUAGE_ORDER[:remainder]:
        targets[language] += 1
    return targets


def _cap_language_targets(
    targets: dict[str, int],
    groups_by_language: dict[str, list[tuple[str, str]]],
    capacities: dict[tuple[str, str], int],
) -> None:
    for language in TARGET_LANGUAGES:
        capacity = sum(capacities[group] for group in groups_by_language[language])
        targets[language] = min(targets[language], capacity)


def _reduce_single_project_targets(
    targets: dict[str, int], groups_by_language: dict[str, list[tuple[str, str]]]
) -> None:
    single_project_languages = [
        language
        for language in TARGET_LANGUAGES
        if len(groups_by_language[language]) == 1 and targets[language] > 1
    ]
    for language in single_project_languages:
        targets[language] -= 1
    _redistribute_language_carry(targets, len(single_project_languages))


def _redistribute_language_carry(targets: dict[str, int], carry: int) -> None:
    for language in _REDISTRIBUTION_LANGUAGE_ORDER:
        if carry == 0:
            break
        targets[language] += 1
        carry -= 1
    if carry:
        raise ValueError("Pilot language quotas cannot be redistributed")


def _fill_language_shortfall(
    targets: dict[str, int],
    groups_by_language: dict[str, list[tuple[str, str]]],
    capacities: dict[tuple[str, str], int],
    sample_size: int,
) -> None:
    for assigned in range(sum(targets.values()), sample_size):
        choices = _languages_with_capacity(targets, groups_by_language, capacities)
        if not choices:
            raise ValueError("Not enough link-backed rows for the requested pilot sample")
        targets[choices[assigned % len(choices)]] += 1


def _languages_with_capacity(
    targets: dict[str, int],
    groups_by_language: dict[str, list[tuple[str, str]]],
    capacities: dict[tuple[str, str], int],
) -> list[str]:
    return [
        language
        for language in TARGET_LANGUAGES
        if targets[language] < sum(capacities[group] for group in groups_by_language[language])
    ]


def _stratified_quotas(
    language_targets: dict[str, int],
    groups_by_language: dict[str, list[tuple[str, str]]],
    capacities: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    for language, target in language_targets.items():
        quotas.update(_language_quotas(target, groups_by_language[language], capacities))
    if sum(quotas.values()) != sum(language_targets.values()):
        raise ValueError("Pilot language quotas cannot fit the available project strata")
    return quotas


def _language_quotas(
    target: int,
    groups: Sequence[tuple[str, str]],
    capacities: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    base, remainder = divmod(target, len(groups))
    quotas = {group: min(base, capacities[group]) for group in groups}
    for group in groups[-remainder:] if remainder else ():
        if quotas[group] < capacities[group]:
            quotas[group] += 1
    return quotas


def _balance_projects(
    quotas: dict[tuple[str, str], int],
    capacities: dict[tuple[str, str], int],
    sample_size: int,
    seed: str,
) -> None:
    targets = {"wikipedia": (sample_size + 1) // 2, "wikivoyage": sample_size // 2}
    counts = Counter({project: 0 for project in _PROJECTS})
    for (project, _), quota in quotas.items():
        counts[project] += quota
    for project, other in (("wikipedia", "wikivoyage"), ("wikivoyage", "wikipedia")):
        _rebalance_project(quotas, capacities, counts, targets, project, other, seed)


def _rebalance_project(
    quotas: dict[tuple[str, str], int],
    capacities: dict[tuple[str, str], int],
    counts: Counter[str],
    targets: dict[str, int],
    project: str,
    other: str,
    seed: str,
) -> None:
    while counts[project] > targets[project]:
        language = _movable_language(quotas, capacities, project, other, seed)
        quotas[(project, language)] -= 1
        quotas[(other, language)] += 1
        counts[project] -= 1
        counts[other] += 1


def _movable_language(
    quotas: dict[tuple[str, str], int],
    capacities: dict[tuple[str, str], int],
    project: str,
    other: str,
    seed: str,
) -> str:
    choices = [
        language
        for candidate_project, language in quotas
        if _can_move_language(quotas, capacities, candidate_project, language, project, other)
    ]
    if not choices:
        raise ValueError("Pilot sample cannot be balanced across projects")
    return min(choices, key=lambda value: _score(seed + "-balance", value))


def _can_move_language(
    quotas: dict[tuple[str, str], int],
    capacities: dict[tuple[str, str], int],
    candidate_project: str,
    language: str,
    project: str,
    other: str,
) -> bool:
    return (
        candidate_project == project
        and (other, language) in quotas
        and quotas[(project, language)] > 0
        and quotas[(other, language)] < capacities[(other, language)]
    )


def _select_group_rows(
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group, quota in quotas.items():
        if quota > len(candidates[group]):
            raise ValueError(f"Pilot group has too few retained rows: {group}")
        selected.extend(_diverse_rows(candidates[group], quota))
    return selected


def _diverse_rows(
    candidates: Sequence[tuple[str, dict[str, Any]]], quota: int
) -> list[dict[str, Any]]:
    selected, selected_keys = _unique_document_rows(candidates, quota)
    if len(selected) == quota:
        return selected
    return _fill_diverse_rows(candidates, selected, selected_keys, quota)


def _unique_document_rows(
    candidates: Sequence[tuple[str, dict[str, Any]]], quota: int
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str]]]:
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    documents: set[str] = set()
    for _, row in candidates:
        if len(selected) == quota:
            break
        if row["document_id"] in documents:
            continue
        selected.append(row)
        selected_keys.add(_identity(row))
        documents.add(row["document_id"])
    return selected, selected_keys


def _fill_diverse_rows(
    candidates: Sequence[tuple[str, dict[str, Any]]],
    selected: list[dict[str, Any]],
    selected_keys: set[tuple[str, str, str, str]],
    quota: int,
) -> list[dict[str, Any]]:
    for _, row in candidates:
        if _identity(row) in selected_keys:
            continue
        selected.append(row)
        if len(selected) == quota:
            return selected
    raise ValueError("Pilot group has too few retained rows")


def _ordered_rows(rows: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_score(seed + "-order", _identity(row)), _identity(row)))


def _input_row(row: dict[str, Any]) -> dict[str, str]:
    return {column: row[column] for column in INPUT_COLUMNS}


def _section_position(row: dict[str, Any]) -> Any:
    section_index = row.get("section_index")
    return row["sentence_id"] if section_index is None else section_index


def _selection_manifest(
    selected: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    *,
    source_files: Sequence[str],
    sample_size: int,
    seed: str,
) -> dict[str, Any]:
    project_counts = Counter(row["project"] for row in rows)
    language_counts = Counter(row["language"] for row in rows)
    documents = {(row["project"], row["document_id"]) for row in rows}
    positions = {
        (
            row["project"],
            row["document_id"],
            _section_position(row),
        )
        for row in selected
    }
    return {
        "seed": seed,
        "sample_size": sample_size,
        "target_languages": list(TARGET_LANGUAGES),
        "source_files": list(source_files),
        "project_counts": dict(sorted(project_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "unique_documents": len(documents),
        "unique_section_positions": len(positions),
        "link_relationship": {
            "all_selected_rows_link_backed": True,
            "selected_rows": len(rows),
        },
        "contract_id": build_contract().identity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", default=_DEFAULT_SEED)
    parser.add_argument("--max-candidate-tables", type=int, default=_DEFAULT_MAX_CANDIDATE_TABLES)
    parser.add_argument("--max-candidate-batches", type=int, default=_DEFAULT_MAX_CANDIDATE_BATCHES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows, manifest = select_rows(
        args.data_root,
        sample_size=args.sample_size,
        seed=args.seed,
        max_candidate_tables=args.max_candidate_tables,
        max_candidate_batches=args.max_candidate_batches,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(args.output_dir / "input.parquet", pa.Table.from_pylist(rows))
    atomic_write_json(args.output_dir / "contract.json", asdict(build_contract()))
    atomic_write_json(args.output_dir / "selection.json", manifest)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
