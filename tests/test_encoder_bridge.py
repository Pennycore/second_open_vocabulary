from __future__ import annotations

import numpy as np

from ov_probe.encoder_bridge import _normalize


def test_normalize_produces_unit_rows() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    actual = _normalize(values)
    assert np.allclose(np.linalg.norm(actual, axis=1), 1.0)
    assert np.allclose(actual[0], [0.6, 0.8])

