import numpy as np

from ov_probe.metrics import alignment_metrics, cosine_similarity, l2_normalize


def test_l2_normalize_and_cosine_identity():
    values = np.eye(3, dtype=np.float32)
    normalized = l2_normalize(values * 5)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    np.testing.assert_allclose(cosine_similarity(values, values), np.eye(3), atol=1e-6)


def test_alignment_metrics_known_ranks():
    scores = np.array([[0.9, 0.1, 0.0], [0.8, 0.7, 0.1], [0.2, 0.1, 0.6]])
    names = ["a", "b", "c"]
    summary, detail = alignment_metrics(scores, names, names, [1, 3, 5])
    assert summary["top_1_accuracy"] == 2 / 3
    assert summary["top_3_accuracy"] == 1.0
    assert summary["top_5_accuracy"] == 1.0
    assert detail.loc[1, "correct_rank"] == 2


def test_zero_vector_is_rejected():
    try:
        l2_normalize(np.zeros((1, 4), dtype=np.float32))
    except ValueError as exc:
        assert "zero" in str(exc)
    else:
        raise AssertionError("zero vector should fail")
