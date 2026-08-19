"""Unit tests for the Potsdam CTP-v1 external confirmation core logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ov_probe.io import InputValidationError  # noqa: E402
from ov_probe.pixel_ovss import FusionCanvas, method_predictions, method_score_matrices  # noqa: E402

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]


def _synthetic():
    rng = np.random.default_rng(0)
    n = 20
    text = np.abs(rng.standard_normal((n, 5))).astype(np.float32)
    visual = np.abs(rng.standard_normal((n, 5))).astype(np.float32)
    anchored = (0.5 * text + 0.5 * visual) / np.linalg.norm(0.5 * np.ones(5) + 0.5 * np.ones(5))  # placeholder norm
    text_pred = np.argmax(text, axis=1)
    return text, visual, anchored, text_pred


def test_protocol_json_frozen_fields():
    p = json.loads(Path("configs/potsdam_ctp_v1_protocol.json").read_text(encoding="utf-8"))
    assert p["status"] == "frozen_pre_result"
    assert p["model"]["architecture"] == "ViT-B-32-quickgelu"
    assert p["model"]["feature_dimension"] == 512
    assert p["alpha"] == 0.5
    assert p["ctp_decision_rule"]["no_threshold"] is True
    assert p["ctp_decision_rule"]["no_temperature"] is True
    assert p["ctp_decision_rule"]["no_beta"] is True
    assert p["segmentation"]["conflict_threshold"] == 0.03
    assert p["segmentation"]["ignore_index"] == 255
    assert len(p["prompts"]["group_a_templates"]) == 8
    assert p["dataset_role"] == "external held-out dataset evaluation"


def test_gt_isolation_phase_names():
    p = json.loads(Path("configs/potsdam_ctp_v1_protocol.json").read_text(encoding="utf-8"))
    assert p["gt_isolation"]["order"] == ["predictions computed", "prediction/config/support hashes persisted", "then GT read", "then evaluation"]
    assert "semantic GT" in p["gt_isolation"]["predict_phase_forbidden"]


def test_method_predictions_frozen_identity_at_full_support():
    text, visual, anchored, text_pred = _synthetic()
    mask = np.ones(5, dtype=bool)
    score_mats = method_score_matrices(text, visual, anchored, mask, text_pred)
    preds = method_predictions(score_mats, text_pred, mask)
    assert np.array_equal(preds["CTP"], preds["SCC"])
    assert np.array_equal(preds["SCC"], preds["C2"])


def test_support_subset_manifest_deterministic():
    """Partial-support subset generation must be reproducible (pre-registered)."""
    from pixel_partial_support import generate_support_subsets
    m1 = generate_support_subsets(CLASSES, [42, 43, 44], [2, 3])
    m2 = generate_support_subsets(CLASSES, [42, 43, 44], [2, 3])
    assert m1 == m2


def test_fusion_canvas_conflict_margin():
    canvas = FusionCanvas(height=8, width=8)
    mask = np.ones((8, 8), dtype=bool)
    canvas.add_mask(mask, class_id=0, score=0.5, x0=0, y0=0)
    canvas.add_mask(mask, class_id=1, score=0.52, x0=0, y0=0)
    labels = canvas.result()
    assert np.all(labels == 255)  # within margin -> conflict ignore
