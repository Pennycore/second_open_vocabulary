"""Unit tests for SCC (Support-Centered Calibration) and the partial-support benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.loveda_partial_support import (
    _c2_normalizers,
    _metrics,
    scc_scores,
)

CLASSES = ["building", "road", "water", "barren", "forest", "agriculture"]


def _synthetic_scores(n_regions: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(CLASSES)
    text_protos = rng.standard_normal((n, 512))
    text_protos /= np.linalg.norm(text_protos, axis=1, keepdims=True)
    visual_protos = text_protos + 0.3 * rng.standard_normal((n, 512))
    visual_protos /= np.linalg.norm(visual_protos, axis=1, keepdims=True)
    regions = rng.standard_normal((n_regions, 512))
    regions /= np.linalg.norm(regions, axis=1, keepdims=True)
    text_scores = regions @ text_protos.T
    visual_scores = regions @ visual_protos.T
    return text_scores, visual_scores, text_protos, visual_protos


def test_scc_k0_equals_text_only():
    text_scores, visual_scores, t, v = _synthetic_scores()
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / _c2_normalizers(t, v)[None, :]
    mask = np.zeros(len(CLASSES), dtype=bool)  # k=0: no supported class
    scc = scc_scores(text_scores, anchored, mask)
    assert np.array_equal(np.argmax(scc, axis=1), np.argmax(text_scores, axis=1))


def test_scc_k6_equals_c2():
    text_scores, visual_scores, t, v = _synthetic_scores()
    normalizers = _c2_normalizers(t, v)
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
    mask = np.ones(len(CLASSES), dtype=bool)  # k=6: all supported
    scc = scc_scores(text_scores, anchored, mask)
    assert np.array_equal(np.argmax(scc, axis=1), np.argmax(anchored, axis=1))


def test_scc_unsupported_keeps_text_score():
    text_scores, visual_scores, t, v = _synthetic_scores()
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / _c2_normalizers(t, v)[None, :]
    mask = np.asarray([True, False, True, False, True, False], dtype=bool)
    scc = scc_scores(text_scores, anchored, mask)
    # unsupported columns must be exactly the text scores
    for i, flag in enumerate(mask):
        if not flag:
            assert np.allclose(scc[:, i], text_scores[:, i])
    # supported columns are anchored minus a per-row constant b
    supported_idx = np.where(mask)[0]
    b = (anchored[:, supported_idx] - text_scores[:, supported_idx]).mean(axis=1)
    for i in supported_idx:
        assert np.allclose(scc[:, i], anchored[:, i] - b)


def test_c2_normalizers_rejects_degenerate():
    t = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    with pytest.raises(InputValidationError):
        _c2_normalizers(t, v)


def test_metrics_consistency():
    rng = np.random.default_rng(1)
    pred = rng.integers(0, len(CLASSES), size=500)
    gt = rng.integers(0, len(CLASSES), size=500)
    m = _metrics(pred, gt, CLASSES)
    assert m["count"] == 500
    assert set(m["per_class_f1"]) == set(CLASSES)
    assert set(m["per_class_iou"]) == set(CLASSES)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["macro_iou"] <= 1.0
    # macro f1 is the mean of per-class f1
    assert m["macro_f1"] == pytest.approx(float(np.mean(list(m["per_class_f1"].values()))))
