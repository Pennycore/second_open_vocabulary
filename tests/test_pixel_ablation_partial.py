"""Unit tests for pixel score-scale ablation and partial-support core logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ov_probe.io import InputValidationError  # noqa: E402
from ov_probe.pixel_ovss import semantic_map_stats  # noqa: E402
from pixel_partial_support import generate_support_subsets  # noqa: E402

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]


def test_generate_support_subsets_deterministic():
    m1 = generate_support_subsets(CLASSES, [42, 43, 44], [2, 3, 4])
    m2 = generate_support_subsets(CLASSES, [42, 43, 44], [2, 3, 4])
    assert m1 == m2  # deterministic
    assert len(m1) == 9  # 3 ks x 3 seeds
    for key, info in m1.items():
        assert info["k"] == len(info["supported"])
        assert len(info["supported"]) + len(info["unsupported"]) == len(CLASSES)
        assert set(info["supported"]) | set(info["unsupported"]) == set(CLASSES)
        assert set(info["supported"]) & set(info["unsupported"]) == set()


def test_generate_support_subsets_seeds_vary():
    m = generate_support_subsets(CLASSES, [42, 43, 44], [2])
    subsets = [frozenset(m[k]["supported"]) for k in m]
    assert len(set(subsets)) >= 2  # seeds produce different subsets (not always, but check diversity)


def test_generate_support_subsets_no_gt_selection():
    """Manifest keys encode k/seed only; supported lists come from seeded RNG
    (sorted by class index, i.e. ascending position in the class list)."""
    m = generate_support_subsets(CLASSES, [42], [3])
    key = "k3_seed42"
    indices = [CLASSES.index(c) for c in m[key]["supported"]]
    assert indices == sorted(indices)
    # recompute with the documented rule (rng = default_rng(seed + k*100))
    rng = np.random.default_rng(42 + 3 * 100)
    idx = sorted(rng.choice(len(CLASSES), size=3, replace=False).tolist())
    assert m[key]["supported"] == [CLASSES[i] for i in idx]


def test_semantic_map_stats():
    label_map = np.full((8, 8), 255, dtype=np.uint8)
    label_map[0:4, 0:4] = 0
    stats = semantic_map_stats(label_map)
    assert stats["pixels_total"] == 64
    assert stats["pixels_labeled"] == 16
    assert stats["pixels_uncovered"] == 48
    assert stats["pixels_assigned"] == 16


def test_semantic_map_stats_conflict():
    label_map = np.full((8, 8), 255, dtype=np.uint8)
    label_map[0:4, 0:4] = 0
    covered = np.zeros((8, 8), dtype=bool)
    covered[0:6, 0:6] = True
    stats = semantic_map_stats(label_map, conflict_source=covered)
    # conflict-ignored = ignore pixels that were covered = 36 - 16 = 20
    assert stats["pixels_conflict_ignored"] == 20
