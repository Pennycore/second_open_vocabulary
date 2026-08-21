"""Unit contracts for the artifact-complete RemoteCLIP rerun entry point."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("remoteclip_vaihingen_complete", ROOT / "scripts" / "run_remoteclip_vaihingen_complete.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_registered_partial_subsets_are_exactly_frozen():
    subsets = MODULE._subset_definitions()
    assert len(subsets) == 25
    assert {row["k"] for row in subsets} == {2, 3, 4}
    assert all(2 <= len(row["supported"]) <= 4 for row in subsets)


def test_pixel_accounting_separates_conflict_and_uncovered():
    labels = np.asarray([[0, 255], [255, 1]], dtype=np.uint8)
    gt = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    covered = np.asarray([[True, True], [False, True]])
    result = MODULE._pixel_accounting(labels, gt, covered)
    assert result["gt_valid_pixels"] == 4
    assert result["assigned_pixels"] == 2
    assert result["conflict_ignore_pixels"] == 1
    assert result["uncovered_pixels"] == 1


def test_bootstrap_requires_area_clusters_and_all_partial_metrics():
    matrix = np.eye(5, dtype=np.int64)
    matrices = {area: {"C2": matrix, "CTP": matrix * 2} for area in MODULE.TEST_AREAS}
    subset = {"supported": ["impervious_surface", "building"], "unsupported": ["low_vegetation", "tree", "car"]}
    output = MODULE._bootstrap(matrices, subset, seed=42, repeats=3)
    assert output["cluster_unit"] == "Vaihingen test area"
    assert set(output["point_estimate"]) == set(MODULE.METRIC_FIELDS)
    assert output["repeats"] == 3
