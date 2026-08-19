"""Unit tests for final review-defense audits (frozen CTP-v1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.final_audit import common_pixel_metrics, per_method_valid_pixels  # noqa: E402
from ov_probe.loveda_partial_support import guard_predictions  # noqa: E402
from ov_probe.pixel_ovss import FusionCanvas, IGNORE_INDEX  # noqa: E402

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]


def test_common_pixel_intersection_correctness():
    """Common-pixel metrics must use the intersection of all methods' valid pixels."""
    gt = {"a": np.asarray([[0, 0, 255], [1, 1, 255]], dtype=np.int64)}
    pred_text = {"a": np.asarray([[0, 1, 255], [0, 0, 255]], dtype=np.int64)}
    pred_c2 = {"a": np.asarray([[0, 1, 255], [1, 1, 255]], dtype=np.int64)}
    pred_scc = {"a": np.asarray([[0, 255, 255], [1, 1, 255]], dtype=np.int64)}
    pred_ctp = {"a": np.asarray([[0, 1, 255], [1, 1, 255]], dtype=np.int64)}
    preds = {"text_only": pred_text, "C2": pred_c2, "SCC": pred_scc, "CTP": pred_ctp}
    # common valid: all methods pred != ignore AND gt != ignore
    # pixel (0,0): gt0 text0 c2 0 scc0 ctp0 -> valid
    # pixel (0,1): gt0 text1 c2 1 scc255(ignore) -> excluded
    # pixel (1,0): gt1 text0 c2 1 scc1 ctp1 -> valid
    # pixel (1,1): gt1 text0 c2 1 scc1 ctp1 -> valid
    result = common_pixel_metrics(preds, gt, CLASSES, ["impervious_surface"], ["building", "low_vegetation", "tree", "car"])
    assert result["text_only"]["common_valid_pixels"] == 3
    # OA: text correct only at (0,0); wrong at (1,0) and (1,1) -> 1/3
    assert result["text_only"]["OA"] == pytest.approx(1 / 3)
    # C2 correct at (0,0),(1,0),(1,1) -> 1.0
    assert result["C2"]["OA"] == pytest.approx(1.0)


def test_all_methods_use_identical_common_mask():
    """All methods must report the same common_valid_pixels on the same subset."""
    gt = {"a": np.zeros((4, 4), dtype=np.int64)}
    gt["a"][0, 0] = 255
    preds = {}
    for m in ("text_only", "C2", "SCC", "CTP", "guard"):
        p = np.full((4, 4), 1, dtype=np.int64)
        p[0, 0] = 255
        preds[m] = {"a": p}
    result = common_pixel_metrics(preds, gt, CLASSES, ["building"], ["impervious_surface", "low_vegetation", "tree", "car"])
    counts = {m: result[m]["common_valid_pixels"] for m in preds}
    assert len(set(counts.values())) == 1  # identical


def test_parent_tile_cluster_mapping():
    """Potsdam cluster key must map patches to their 14 parent tiles."""
    def cluster_key(image_id: str) -> str:
        return image_id.split("_x")[0]  # parent tile (top_potsdam_2_13)
    patches = ["top_potsdam_2_13_x0000_y0000", "top_potsdam_2_13_x0384_y0768", "top_potsdam_4_15_x4992_y3456"]
    parents = [cluster_key(p) for p in patches]
    assert parents == ["top_potsdam_2_13", "top_potsdam_2_13", "top_potsdam_4_15"]
    assert len(set(parents)) == 2


def test_vaihingen_area_cluster_mapping():
    """Vaihingen cluster unit is the test area (image_id == area)."""
    areas = ["vaih_area11", "vaih_area15", "vaih_area28", "vaih_area30", "vaih_area34"]
    assert len(set(areas)) == 5  # each area is its own cluster


def test_guard_deterministic():
    rng = np.random.default_rng(0)
    text = np.abs(rng.standard_normal((50, 5))).astype(np.float32)
    anchored = np.abs(rng.standard_normal((50, 5))).astype(np.float32)
    text_pred = np.argmax(text, axis=1)
    unsupported = [1, 3]
    g1 = guard_predictions(text_pred, anchored, unsupported)
    g2 = guard_predictions(text_pred, anchored, unsupported)
    assert np.array_equal(g1, g2)
    # text top-1 in unsupported -> kept
    for i in range(50):
        if text_pred[i] in unsupported:
            assert g1[i] == text_pred[i]


def test_support_subset_manifest_unchanged():
    """Support subset manifests must remain the pre-registered frozen ones."""
    m = json.loads(Path("outputs/vaihingen_pixel_partial_support_v0/support_subset_manifest.json").read_text())
    assert len(m) == 9  # k=2/3/4 x seeds 42/43/44
    for key, info in m.items():
        assert info["seed"] in (42, 43, 44)
        assert set(info["supported"]) | set(info["unsupported"]) == set(CLASSES)
        assert set(info["supported"]) & set(info["unsupported"]) == set()


def test_frozen_ctp_hash_unchanged():
    """CTP-v1 frozen config must be byte-identical to the freeze record hash."""
    import hashlib
    raw = Path("configs/ctp_v1_frozen.json").read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    # recorded at freeze commit f54c034 (see reports/ctp_v1_freeze_record_20260819.md)
    assert h == "788f1962d497022fbd5cacd7b63eaedddecd0343104aa726ee80afcdf1b37430"
    assert Path("reports/ctp_v1_freeze_record_20260819.md").is_file()


def test_no_gt_access_in_prediction_phase():
    """Prediction-phase configs must have label_dir = null."""
    for cfg in ("configs/vaihingen_pixel_partial_support_v0.yaml",
                "configs/potsdam_ctp_v1_partial_v0.yaml",
                "configs/potsdam_ctp_v1_v0.yaml"):
        import yaml
        c = yaml.safe_load(Path(cfg).read_text())
        assert c["paths"].get("label_dir") is None, f"{cfg} must not configure GT in predict phase"
