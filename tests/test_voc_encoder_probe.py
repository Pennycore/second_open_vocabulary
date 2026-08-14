import numpy as np
import pytest

from ov_probe.io import InputValidationError
from ov_probe.voc_encoder_probe import average_precision, macro_average_precision, paired_bootstrap


def test_average_precision_matches_hand_calculation() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=bool)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    assert average_precision(labels, scores) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_macro_average_precision_and_bootstrap_are_deterministic() -> None:
    labels = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    remote = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    clip = np.asarray([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]])
    macro, per_class = macro_average_precision(labels, remote)
    assert macro == pytest.approx(1.0)
    assert per_class.tolist() == pytest.approx([1.0, 1.0])
    first = paired_bootstrap(labels, remote, clip, replicates=20, seed=42)
    second = paired_bootstrap(labels, remote, clip, replicates=20, seed=42)
    assert first == second


def test_average_precision_rejects_no_positive_class() -> None:
    with pytest.raises(InputValidationError, match="without positives"):
        average_precision(np.zeros(3, dtype=bool), np.arange(3, dtype=float))
