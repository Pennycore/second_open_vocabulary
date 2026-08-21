from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("remoteclip_vaihingen_partial", ROOT / "scripts" / "evaluate_remoteclip_vaihingen_partial.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_all_bitmask_registration_has_expected_counts():
    subsets = MODULE._expected_subsets()
    assert len(subsets) == 25
    assert [sum(info["k"] == k for info in subsets.values()) for k in (2, 3, 4)] == [10, 10, 5]
    assert tuple(subsets) == tuple(f"subset_{i}" for i in range(32) if bin(i).count("1") in (2, 3, 4))


def test_source_subset_validation_accepts_json_lexical_key_order(tmp_path: Path):
    subsets = MODULE._expected_subsets()
    source = {
        key: {"k": value["k"], "supported": value["supported"], "unsupported": value["unsupported"], "methods": {method: {metric: 0.0 for metric in MODULE.METRIC_COLUMNS} | {"valid_pixels": 0} for method in MODULE.RUN_METHODS}}
        for key, value in sorted(subsets.items())
    }
    rows = [
        {"subset_index": value["subset_index"], "method": method, **{metric: 0.0 for metric in MODULE.METRIC_COLUMNS}, "valid_pixels": 0}
        for _, value in sorted(subsets.items())
        for method in MODULE.RUN_METHODS
    ]
    path = tmp_path / "partial.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=["subset_index", "method", *MODULE.METRIC_COLUMNS, "valid_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    MODULE._validate_source_partial(source, path)


def test_pixel_accounting_distinguishes_conflicts_from_uncovered():
    pred = np.array([[0, 255], [255, 1]], dtype=np.uint8)
    gt = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    covered = np.array([[True, True], [False, True]])
    result = MODULE._pixel_accounting(pred, gt, covered)
    assert result == {"valid_pixels": 1, "assigned_pixels": 2, "conflict_ignore_pixels": 1, "uncovered_pixels": 1, "gt_ignore_pixels": 1, "total_pixels": 4}


def test_bootstrap_is_deterministic_and_uses_area_clusters():
    subset = {"supported": ["impervious_surface", "building"], "unsupported": ["low_vegetation", "tree", "car"]}
    base = np.eye(5, dtype=np.int64) * 3
    areas = {}
    for area in MODULE.TEST_AREAS:
        c2 = base.copy()
        ctp = base.copy()
        ctp[0, 0] += area
        areas[area] = {"C2": c2, "CTP": ctp}
    first = MODULE._bootstrap(areas, subset, seed=42, repeats=20, metrics=("OA", "macro_f1", "mIoU", "U_IoU", "H_IoU"))
    second = MODULE._bootstrap(areas, subset, seed=42, repeats=20, metrics=("OA", "macro_f1", "mIoU", "U_IoU", "H_IoU"))
    assert first == second
    assert first["cluster_unit"] == "Vaihingen test area"
    assert set(first["direction_by_area"]) == {str(area) for area in MODULE.TEST_AREAS}


def test_score_sets_preserve_saved_text_top1_from_compressed_cache():
    cache = {
        "text_scores": np.array([[0.1, 0.10001, 0.0, 0.0, 0.0]], dtype=np.float32),
        "visual_scores": np.zeros((1, 5), dtype=np.float32),
        "text_prototypes": np.eye(5, dtype=np.float32),
        "visual_prototypes": np.eye(5, dtype=np.float32),
        "text_pred": np.array([0], dtype=np.int64),
    }
    subsets = MODULE._expected_subsets()
    _, predictions = MODULE._score_sets(cache, subsets)
    assert all(values["text_only"].tolist() == [0] for values in predictions.values())
