from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ov_probe.io import InputValidationError
from ov_probe.openai_clip_visual_anchor import (
    build_visual_prototypes,
    fuse_scores,
    join_split_to_partitions,
)


def _partitions() -> list[dict[str, object]]:
    return [
        {"row_index": 0, "image_id": "dev-a", "candidate_index": 0, "partition": "development"},
        {"row_index": 1, "image_id": "dev-b", "candidate_index": 0, "partition": "development"},
        {"row_index": 2, "image_id": "hold-a", "candidate_index": 0, "partition": "heldout"},
        {"row_index": 3, "image_id": "hold-b", "candidate_index": 0, "partition": "heldout"},
    ]


def _records(indices: list[int], labels: list[str]) -> list[dict[str, object]]:
    rows = _partitions()
    return [{**rows[index], "sam3_source_label": label, "cam_label": label} for index, label in zip(indices, labels)]


def test_visual_prototypes_are_deterministic_and_development_only() -> None:
    features = np.asarray([[2.0, 0.0], [0.0, 3.0], [-100.0, 0.0], [0.0, -100.0]], dtype=np.float32)
    development = _records([0, 1], ["building", "road"])
    prototypes, counts = build_visual_prototypes(features, development, ["building", "road"])

    np.testing.assert_allclose(prototypes, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)
    assert counts == {"building": 1, "road": 1}
    # Heldout vectors deliberately point in the opposite direction and are never passed
    # into the prototype builder, so they cannot influence the fixed prototypes.
    assert float(prototypes[0] @ np.asarray([1.0, 0.0])) == pytest.approx(1.0)


def test_fixed_half_half_fusion_is_deterministic() -> None:
    regions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    visual = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    text_scores, fused = fuse_scores(regions, text, visual)

    np.testing.assert_allclose(text_scores, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(fused, [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)


def test_join_rejects_duplicate_or_missing_split_keys() -> None:
    partitions = _partitions()
    development = _records([0, 1], ["building", "road"])
    heldout = _records([2, 3], ["water", "barren"])
    with pytest.raises(InputValidationError, match="duplicate"):
        join_split_to_partitions(partitions, development + [development[0]], heldout, 4)
    with pytest.raises(InputValidationError, match="cover all"):
        join_split_to_partitions(partitions, development, heldout[:1], 4)


def test_registered_protocol_discloses_nonconfirmatory_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs" / "openai_clip_visual_anchor_exploratory_protocol_v1.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((root / "configs" / "openai_clip_visual_anchor_exploratory_v1.yaml").read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_pre_result"
    assert protocol["scientific_evidence"] is False
    assert protocol["post_hoc_exploratory"] is True
    assert "not blind" in protocol["holdout_interpretation"]
    assert protocol["strategy"]["fixed_text_weight"] == 0.5
    assert protocol["strategy"]["fixed_visual_weight"] == 0.5
    assert len(protocol["prompts"]["group_a_templates"]) == 8
    assert cfg["experiment"]["overwrite"] is False
    assert cfg["paths"]["openai_clip_checkpoint"] is None
    assert all(not Path(str(value)).is_absolute() for value in cfg["paths"].values() if value is not None)
