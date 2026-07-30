"""Deterministic JSON codec for cached augmentation file summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .models import PerFileSummary


def summary_to_json(summary: PerFileSummary) -> dict[str, Any]:
    """Serialize one summary using stable ordering for set-like fields."""
    return {
        "relative_path": summary.relative_path,
        "fingerprint": summary.fingerprint,
        "file_size_bytes": summary.file_size_bytes,
        "kind": summary.kind,
        "scan_failed": summary.scan_failed,
        "rows": summary.rows,
        "non_empty": summary.non_empty,
        "empty_or_null": summary.empty_or_null,
        "total_chars": summary.total_chars,
        "total_words": summary.total_words,
        "total_tokens_estimate": summary.total_tokens_estimate,
        "document_ids": sorted(summary.document_ids),
        "section_ids": sorted(summary.section_ids),
        "qids": sorted(summary.qids),
        "languages": dict(sorted(summary.languages.items())),
        "fact_rows": summary.fact_rows,
        "fact_ids": sorted(summary.fact_ids),
        "subject_qids": sorted(summary.subject_qids),
        "property_ids": sorted(summary.property_ids),
        "property_labels": dict(sorted(summary.property_labels.items())),
        "property_counts": dict(sorted(summary.property_counts.items())),
        "with_property_en_label": summary.with_property_en_label,
        "with_value_en_label": summary.with_value_en_label,
        "with_qualifiers": summary.with_qualifiers,
        "with_references": summary.with_references,
        "unavailable_qualifiers": summary.unavailable_qualifiers,
        "unavailable_references": summary.unavailable_references,
        "value_type_counts": dict(sorted(summary.value_type_counts.items())),
    }


def summary_from_json(blob: Mapping[str, object]) -> PerFileSummary | None:
    """Decode a compatible cache entry or return ``None``."""
    required = ("relative_path", "fingerprint", "file_size_bytes", "kind")
    if not all(key in blob for key in required):
        return None

    def strings(key: str) -> frozenset[str]:
        value = blob.get(key)
        return frozenset(str(item) for item in value) if isinstance(value, list) else frozenset()

    def string_map(key: str) -> dict[str, str]:
        value = blob.get(key)
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in cast(dict[object, object], value).items()}

    def integer_map(key: str) -> dict[str, int]:
        value = blob.get(key)
        if not isinstance(value, dict):
            return {}
        return {str(k): int(cast(Any, v)) for k, v in cast(dict[object, object], value).items()}

    def integer(key: str) -> int:
        value = blob.get(key)
        return int(value) if isinstance(value, (int, float, str, bytes)) else 0

    def boolean(key: str) -> bool:
        value = blob.get(key)
        return bool(value) if value is not None else False

    def string(key: str) -> str:
        value = blob.get(key)
        return str(value) if value is not None else ""

    return PerFileSummary(
        relative_path=string("relative_path"),
        fingerprint=string("fingerprint"),
        file_size_bytes=integer("file_size_bytes"),
        kind=string("kind"),
        scan_failed=boolean("scan_failed"),
        rows=integer("rows"),
        non_empty=integer("non_empty"),
        empty_or_null=integer("empty_or_null"),
        total_chars=integer("total_chars"),
        total_words=integer("total_words"),
        total_tokens_estimate=integer("total_tokens_estimate"),
        document_ids=strings("document_ids"),
        section_ids=strings("section_ids"),
        qids=strings("qids"),
        languages=integer_map("languages"),
        fact_rows=integer("fact_rows"),
        fact_ids=strings("fact_ids"),
        subject_qids=strings("subject_qids"),
        property_ids=strings("property_ids"),
        property_labels=string_map("property_labels"),
        property_counts=integer_map("property_counts"),
        with_property_en_label=integer("with_property_en_label"),
        with_value_en_label=integer("with_value_en_label"),
        with_qualifiers=integer("with_qualifiers"),
        with_references=integer("with_references"),
        unavailable_qualifiers=integer("unavailable_qualifiers"),
        unavailable_references=integer("unavailable_references"),
        value_type_counts=integer_map("value_type_counts"),
    )


__all__ = ["summary_from_json", "summary_to_json"]
