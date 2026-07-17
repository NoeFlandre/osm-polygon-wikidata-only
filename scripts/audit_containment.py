"""Read-only audit of configured whole-file containment retirements."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from osm_polygon_wikidata_only.pipeline.containment_migration import audit_rule
from osm_polygon_wikidata_only.pipeline.containment_policy import CONTAINMENT_RULES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [audit_rule(args.data_root / "processed", rule) for rule in CONTAINMENT_RULES]
    payload = {
        "safe_parents": [report.parent for report in reports if report.safe_to_stage],
        "blocked_parents": [report.parent for report in reports if not report.safe_to_stage],
        "reports": [asdict(report) | {"safe_to_stage": report.safe_to_stage} for report in reports],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if not payload["blocked_parents"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
