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
    first = MODULE._bootstrap(areas, subset, seed=42, repeats=20)
    second = MODULE._bootstrap(areas, subset, seed=42, repeats=20)
    assert first == second
    assert first["cluster_unit"] == "Vaihingen test area"
    assert set(first["direction_by_area"]) == {str(area) for area in MODULE.TEST_AREAS}
