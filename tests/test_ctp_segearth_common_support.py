"""Synthetic invariants for the frozen CTP/SegEarth common-support evaluator."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ov_probe.common_support import CLASSES, IGNORE, TEST_AREAS, metrics, mutual_valid_mask, strict_counts


def test_five_area_and_class_mapping_are_frozen():
    assert TEST_AREAS == (11, 15, 28, 30, 34)
    assert CLASSES == ("impervious_surface", "building", "low_vegetation", "tree", "car")


def test_ctp_ignore_is_strict_fn_and_oa_error():
    omega = np.ones((1, 4), dtype=bool)
    gt = np.asarray([[0, 1, 1, 2]], dtype=np.uint8)
    ctp = np.asarray([[0, IGNORE, 1, IGNORE]], dtype=np.uint8)
    value = metrics(strict_counts(omega, gt, ctp, method="CTP-v1"))
    assert value["denominator"] == 4
    assert value["correct"] == 2
    assert value["OA"] == pytest.approx(0.5)
    assert value["ctp_abstentions"] == 2
    assert value["semantic_predictions"] == 2
    assert value["per_class"]["building"]["FN"] == 1
    assert value["per_class"]["low_vegetation"]["FN"] == 1


def test_both_methods_have_identical_fixed_denominator():
    omega = np.asarray([[True, False], [True, True]])
    gt = np.asarray([[0, 4], [2, 3]], dtype=np.uint8)
    ctp = np.asarray([[0, IGNORE], [IGNORE, 3]], dtype=np.uint8)
    seg = np.asarray([[0, 4], [5, 3]], dtype=np.uint8)  # 5 = SegEarth clutter error
    ctp_value = strict_counts(omega, gt, ctp, method="CTP-v1")
    seg_value = strict_counts(omega, gt, seg, method="SegEarth-OV")
    assert ctp_value.denominator == seg_value.denominator == int(omega.sum())
    assert metrics(seg_value)["per_class"]["low_vegetation"]["FN"] == 1


def test_mutual_valid_mask_excludes_only_nonsemantic_predictions():
    omega = np.ones((2, 2), dtype=bool)
    ctp = np.asarray([[0, IGNORE], [2, 3]], dtype=np.uint8)
    seg = np.asarray([[0, 1], [5, 3]], dtype=np.uint8)
    actual = mutual_valid_mask(omega, ctp, seg)
    assert np.array_equal(actual, np.asarray([[True, False], [False, True]]))


def test_ctp_rejects_unregistered_nonsemantic_label():
    with pytest.raises(ValueError, match="only 0..4 or 255"):
        strict_counts(np.ones((1, 1), dtype=bool), np.zeros((1, 1), dtype=np.uint8), np.full((1, 1), 5, dtype=np.uint8), method="CTP-v1")


def test_evaluator_has_no_inference_imports_or_model_runner():
    tree = ast.parse(Path("scripts/evaluate_ctp_segearth_common_support.py").read_text(encoding="utf-8"))
    modules = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not ({"torch", "open_clip", "sam3_remote_wsss", "mmseg", "mmcv"} & modules)
