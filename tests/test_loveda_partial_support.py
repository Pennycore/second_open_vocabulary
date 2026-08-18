"""Unit tests for SCC (Support-Centered Calibration), CTP, and partial-support benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.loveda_partial_support import (
    _c2_normalizers,
    _metrics,
    ctp_predictions,
    guard_predictions,
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


def test_ctp_k0_equals_text_only():
    text_scores, visual_scores, t, v = _synthetic_scores()
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / _c2_normalizers(t, v)[None, :]
    mask = np.zeros(len(CLASSES), dtype=bool)  # k=0
    scc = scc_scores(text_scores, anchored, mask)
    text_top1 = np.argmax(text_scores, axis=1)
    ctp = ctp_predictions(text_top1, text_scores, scc, mask)
    assert np.array_equal(ctp, text_top1)


def test_ctp_k6_equals_scc_and_c2():
    text_scores, visual_scores, t, v = _synthetic_scores()
    normalizers = _c2_normalizers(t, v)
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
    mask = np.ones(len(CLASSES), dtype=bool)  # k=6
    scc = scc_scores(text_scores, anchored, mask)
    text_top1 = np.argmax(text_scores, axis=1)
    ctp = ctp_predictions(text_top1, text_scores, scc, mask)
    assert np.array_equal(ctp, np.argmax(scc, axis=1))
    assert np.array_equal(ctp, np.argmax(anchored, axis=1))


def test_ctp_gate_preserves_confident_text_unsupported():
    """A region whose text top-1 is unsupported and strictly beats all supported
    SCC competition scores keeps the text prediction."""
    n = 6
    text_scores = np.zeros((n, 6), dtype=np.float32)
    anchored = np.zeros((n, 6), dtype=np.float32)
    # class 0 (building) is unsupported; classes 1..5 supported
    for i in range(n):
        text_scores[i, 0] = 0.9 + 0.01 * i   # text prefers unsupported class 0
        text_scores[i, 1] = 0.5
        text_scores[i, 2] = 0.4
        text_scores[i, 3] = 0.3
        text_scores[i, 4] = 0.2
        text_scores[i, 5] = 0.1
        anchored[i, 1:] = 0.6                 # supported C2 scores below text top-1
        anchored[i, 0] = 0.2
    mask = np.asarray([False, True, True, True, True, True], dtype=bool)
    scc = scc_scores(text_scores, anchored, mask)
    text_top1 = np.argmax(text_scores, axis=1)
    ctp = ctp_predictions(text_top1, text_scores, scc, mask)
    assert np.array_equal(ctp, np.zeros(n, dtype=np.int64))  # all preserved as class 0


def test_ctp_follows_scc_when_text_not_confident():
    """When the unsupported text top-1 plus its margin does NOT beat every
    supported SCC competition score, CTP follows the SCC competition."""
    n = 4
    text_scores = np.zeros((n, 6), dtype=np.float32)
    anchored = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        text_scores[i, 0] = 0.6    # text top-1 = unsupported class 0
        text_scores[i, 1] = 0.59   # tiny margin (0.01)
        text_scores[i, 2] = 0.1
        anchored[i, 1] = 1.2       # supported class 1 anchored boost
        anchored[i, 0] = 0.1
    mask = np.asarray([False, True, True, True, True, True], dtype=bool)
    scc = scc_scores(text_scores, anchored, mask)
    # SCC supported class 1: A_1 - b, b = mean(A_1 - T_1) = 1.2 - 0.59 = 0.61
    # S_1 = 1.2 - 0.61 = 0.59; T_0 + margin = 0.6 + 0.01 = 0.61 > 0.59? -> preserves!
    # Force the not-confident branch: make the margin tiny relative to the gap.
    text_scores[:, 1] = 0.599
    text_scores[:, 0] = 0.6
    scc = scc_scores(text_scores, anchored, mask)
    text_top1 = np.argmax(text_scores, axis=1)
    ctp = ctp_predictions(text_top1, text_scores, scc, mask)
    scc_pred = np.argmax(scc, axis=1)
    assert np.array_equal(ctp, scc_pred)
    assert not np.array_equal(ctp, text_top1)  # does not blindly preserve


def test_ctp_preserves_when_margin_covers_gap():
    """A clear text winner (large margin) for an unsupported class is preserved
    even when the SCC competition slightly out-scores the raw text score."""
    n = 3
    text_scores = np.zeros((n, 6), dtype=np.float32)
    anchored = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        text_scores[i, 0] = 0.60   # text top-1 = unsupported class 0
        text_scores[i, 1] = 0.40   # margin 0.20
        anchored[i, 1] = 0.75      # supported class 1 anchored boost
        anchored[i, 0] = 0.1
    mask = np.asarray([False, True, True, True, True, True], dtype=bool)
    scc = scc_scores(text_scores, anchored, mask)
    # S_1 = A_1 - b = 0.75 - (0.75-0.40) = 0.40; T_0 + margin = 0.60+0.20 = 0.80 > 0.40
    text_top1 = np.argmax(text_scores, axis=1)
    ctp = ctp_predictions(text_top1, text_scores, scc, mask)
    assert np.array_equal(ctp, text_top1)  # preserved as class 0


def test_ctp_rejects_shape_mismatch():
    text_scores, visual_scores, t, v = _synthetic_scores()
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / _c2_normalizers(t, v)[None, :]
    mask = np.asarray([True, False, True, False, True, False], dtype=bool)
    scc = scc_scores(text_scores, anchored, mask)
    text_top1 = np.argmax(text_scores, axis=1)
    with pytest.raises(InputValidationError):
        ctp_predictions(text_top1, text_scores[:, :-1], scc, mask)
    with pytest.raises(InputValidationError):
        ctp_predictions(text_top1, text_scores, scc, mask[:-1])
