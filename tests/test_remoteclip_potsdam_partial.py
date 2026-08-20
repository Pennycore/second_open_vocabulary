"""Unit tests for the cache-only RemoteCLIP partial evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "remoteclip_partial", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_remoteclip_potsdam_partial.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_subset_metrics_and_harmonic_means():
    matrix = np.array([[5, 0, 0, 0, 0], [0, 4, 1, 0, 0], [0, 1, 3, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]])
    row = MODULE._metrics_for_subset(matrix, ["impervious_surface", "building"], ["low_vegetation", "tree", "car"])
    assert 0.0 <= row["S_F1"] <= 1.0
    assert 0.0 <= row["U_IoU"] <= 1.0
    assert min(row["S_F1"], row["U_F1"]) <= row["H_F1"] <= max(row["S_F1"], row["U_F1"])
    assert min(row["S_IoU"], row["U_IoU"]) <= row["H_IoU"] <= max(row["S_IoU"], row["U_IoU"])


def test_expected_frozen_subset_grid():
    assert len(MODULE.EXPECTED_SUBSET_KEYS) == 9
    assert "r25_seed42" in MODULE.EXPECTED_SUBSET_KEYS
    assert "r75_seed44" in MODULE.EXPECTED_SUBSET_KEYS
