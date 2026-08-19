"""Unit tests for the pixel-level OVSS module (synthetic masks/scores)."""

from __future__ import annotations

import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.pixel_ovss import (
    FusionCanvas,
    assemble_semantic_map,
    method_predictions,
    method_score_matrices,
    pixel_confusion,
)
from ov_probe.loveda_partial_support import scc_scores

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]


def _synthetic_inputs(n=20, seed=0):
    rng = np.random.default_rng(seed)
    text_scores = rng.standard_normal((n, 5)).astype(np.float32)
    text_scores = np.abs(text_scores)
    anchored = rng.standard_normal((n, 5)).astype(np.float32)
    anchored = np.abs(anchored)
    text_pred = np.argmax(text_scores, axis=1)
    return text_scores, anchored, text_pred


def test_fusion_canvas_highest_score_wins():
    canvas = FusionCanvas(height=10, width=10)
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:5, :] = True
    canvas.add_mask(mask1, class_id=0, score=0.8, x0=0, y0=0)
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[3:8, :] = True
    canvas.add_mask(mask2, class_id=1, score=0.9, x0=0, y0=0)
    labels = canvas.result()
    # rows 0-2 class 0, rows 3-4 class 1 (higher score), rows 5-7 class 1, row 8-9 uncovered
    assert labels[0, 0] == 0
    assert labels[4, 0] == 1
    assert labels[8, 0] == 255  # uncovered


def test_fusion_conflict_marks_ignore():
    canvas = FusionCanvas(height=10, width=10)
    mask = np.ones((10, 10), dtype=bool)
    canvas.add_mask(mask, class_id=0, score=0.5, x0=0, y0=0)
    canvas.add_mask(mask, class_id=1, score=0.51, x0=0, y0=0)  # within margin 0.03
    labels = canvas.result()
    assert np.all(labels == 255)  # conflict -> ignore


def test_assemble_semantic_map():
    shape = (10, 10)
    regions = [
        {"mask": np.ones((5, 5), dtype=bool), "x0": 0, "y0": 0, "row_index": 0},
        {"mask": np.ones((5, 5), dtype=bool), "x0": 5, "y0": 5, "row_index": 1},
    ]
    pred = np.asarray([0, 1], dtype=np.int64)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    label_map, stats = assemble_semantic_map(shape, regions, pred, scores, CLASSES)
    assert label_map[0, 0] == 0
    assert label_map[7, 7] == 1
    assert label_map[2, 7] == 255  # uncovered corner
    assert label_map[7, 2] == 255  # uncovered corner
    assert stats["pixels_covered"] == 50


def test_method_predictions_consistent():
    text_scores, anchored, text_pred = _synthetic_inputs()
    visual_scores = np.abs(np.random.default_rng(1).standard_normal(text_scores.shape)).astype(np.float32)
    mask = np.ones(5, dtype=bool)
    score_mats = method_score_matrices(text_scores, visual_scores, anchored, mask, text_pred)
    preds = method_predictions(score_mats, text_pred, mask)
    # CTP == SCC == C2 == text argmax when all supported
    assert np.array_equal(preds["CTP"], preds["SCC"])
    assert np.array_equal(preds["SCC"], preds["C2"])
    assert np.array_equal(preds["C2"], np.argmax(anchored, axis=1))


def test_pixel_confusion_ignore_handling():
    pred = np.asarray([[0, 1, 255], [0, 0, 255]], dtype=np.int64)
    gt = np.asarray([[0, 0, 255], [1, 1, 255]], dtype=np.int64)
    metrics = pixel_confusion(pred, gt, CLASSES)
    assert metrics["valid_pixels"] == 4
    assert metrics["ignore_pixels"] == 2
    assert metrics["OA"] == 0.25  # only (0,0) correct of 4 valid
    assert metrics["confusion_matrix"][0][0] == 1


def test_pixel_confusion_rejects_mismatched_shape():
    with pytest.raises(ValueError):
        pixel_confusion(np.zeros((2, 2), dtype=np.int64), np.zeros((3, 3), dtype=np.int64), CLASSES)
