"""Phase 2 / Amendment 7: Pending-publication envelope preservation.

The pending-publication functions ``save_pending_publications``,
``add_pending_publications``, ``remove_pending_publications`` MUST
merge with the existing envelope (preserving every other field) --
especially ``metadata_refresh``.

Tests:

1. ``add_pending_publications`` preserves an existing
   ``metadata_refresh`` field.
2. ``remove_pending_publications`` preserves an existing
   ``metadata_refresh`` field.
3. The marker+stems interaction works across multiple add/remove ops.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _import_module():
    from osm_polygon_wikidata_only.pipeline import pending_publications as mod

    return mod


def _sha256(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def test_add_pending_publications_preserves_metadata_refresh(tmp_path: Path) -> None:
    """Adding stems must NOT erase the ``metadata_refresh`` field."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()

    # Pre-seed the envelope with a metadata_refresh field.
    path = dr.processed_manifests / mod.FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_existing_marker = {
        "stems": ["monaco-latest"],
        "fingerprint_hashes": {"monaco-latest": _sha256("monaco")},
    }
    initial_envelope = {
        "contract_version": mod.CONTRACT_VERSION,
        "stems": [],
        "metadata_refresh": pre_existing_marker,
    }
    path.write_text(json.dumps(initial_envelope, indent=2, sort_keys=True) + "\n")

    mod.add_pending_publications(dr, {"alpha-latest", "beta-latest"})

    envelope = json.loads(path.read_text())
    assert envelope["metadata_refresh"] == pre_existing_marker, (
        f"add_pending_publications erased metadata_refresh: {envelope}"
    )
    assert set(envelope["stems"]) == {"alpha-latest", "beta-latest"}


def test_remove_pending_publications_preserves_metadata_refresh(tmp_path: Path) -> None:
    """Removing stems must NOT erase the ``metadata_refresh`` field."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()

    path = dr.processed_manifests / mod.FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_existing_marker = {
        "stems": ["monaco-latest"],
        "fingerprint_hashes": {"monaco-latest": _sha256("monaco")},
    }
    initial_envelope = {
        "contract_version": mod.CONTRACT_VERSION,
        "stems": ["alpha-latest", "beta-latest"],
        "metadata_refresh": pre_existing_marker,
    }
    path.write_text(json.dumps(initial_envelope, indent=2, sort_keys=True) + "\n")

    mod.remove_pending_publications(dr, {"alpha-latest"})

    envelope = json.loads(path.read_text())
    assert envelope["metadata_refresh"] == pre_existing_marker, (
        f"remove_pending_publications erased metadata_refresh: {envelope}"
    )
    assert envelope["stems"] == ["beta-latest"]


def test_marker_and_stems_round_trip(tmp_path: Path) -> None:
    """Setting a marker then adding stems then removing stems must keep
    the marker intact at every step."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()

    mod.add_pending_publications(dr, {"alpha-latest", "beta-latest"})
    mod.set_metadata_refresh_marker(dr, ["alpha-latest"], {"alpha-latest": _sha256("alpha")})

    # Save the bytes once the marker is set.
    path = dr.processed_manifests / mod.FILENAME
    payload_with_marker = json.loads(path.read_text())

    # Add more stems; the marker must survive.
    mod.add_pending_publications(dr, {"gamma-latest"})
    payload_after_add = json.loads(path.read_text())
    assert payload_after_add["metadata_refresh"] == payload_with_marker["metadata_refresh"]

    # Remove stems; the marker must survive.
    mod.remove_pending_publications(dr, {"alpha-latest"})
    payload_after_remove = json.loads(path.read_text())
    assert payload_after_remove["metadata_refresh"] == payload_with_marker["metadata_refresh"]


def test_save_pending_publications_preserves_other_fields(tmp_path: Path) -> None:
    """``save_pending_publications`` itself (not just add/remove) must
    merge with other envelope fields."""
    mod = _import_module()
    from osm_polygon_wikidata_only.config.paths import DataRoot

    dr = DataRoot(tmp_path)
    dr.ensure()

    path = dr.processed_manifests / mod.FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_existing_marker = {
        "stems": ["monaco-latest"],
        "fingerprint_hashes": {"monaco-latest": _sha256("monaco")},
    }
    initial_envelope = {
        "contract_version": mod.CONTRACT_VERSION,
        "stems": ["alpha-latest"],
        "metadata_refresh": pre_existing_marker,
        "extra_field": {"key": "value"},
    }
    path.write_text(json.dumps(initial_envelope, indent=2, sort_keys=True) + "\n")

    mod.save_pending_publications(dr, {"beta-latest"})

    envelope = json.loads(path.read_text())
    assert envelope["metadata_refresh"] == pre_existing_marker
    assert envelope.get("extra_field") == {"key": "value"}
    assert envelope["stems"] == ["beta-latest"]
