"""Strict frozen-support scoring primitives for external semantic baselines.

The functions here never load models, candidates, or images.  They score only
already-serialized prediction arrays against an already-decoded GT array and a
frozen boolean support mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CLASSES = (
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
)
TEST_AREAS = (11, 15, 28, 30, 34)
IGNORE = 255
N_CLASSES = len(CLASSES)


@dataclass(frozen=True)
class StrictCounts:
    """Sufficient statistics under a fixed support denominator."""

    confusion: np.ndarray
    fn_extra: np.ndarray
    denominator: int
    correct: int
    semantic_predictions: int
    nonsemantic_predictions: int
    ctp_abstentions: int

    def add(self, other: "StrictCounts") -> "StrictCounts":
        return StrictCounts(
            confusion=self.confusion + other.confusion,
            fn_extra=self.fn_extra + other.fn_extra,
            denominator=self.denominator + other.denominator,
            correct=self.correct + other.correct,
            semantic_predictions=self.semantic_predictions + other.semantic_predictions,
            nonsemantic_predictions=self.nonsemantic_predictions + other.nonsemantic_predictions,
            ctp_abstentions=self.ctp_abstentions + other.ctp_abstentions,
        )


def semantic_valid(prediction: np.ndarray) -> np.ndarray:
    """Return pixels predicted as one of the five evaluated semantic classes."""
    return (prediction >= 0) & (prediction < N_CLASSES)


def mutual_valid_mask(omega: np.ndarray, ctp: np.ndarray, segearth: np.ndarray) -> np.ndarray:
    """Frozen diagnostic mask: Omega ∩ CTP semantic ∩ SegEarth semantic."""
    _validate_shapes(omega, ctp, segearth)
    return omega.astype(bool, copy=False) & semantic_valid(ctp) & semantic_valid(segearth)


def _validate_shapes(omega: np.ndarray, gt: np.ndarray, prediction: np.ndarray) -> None:
    if omega.ndim != 2 or gt.ndim != 2 or prediction.ndim != 2:
        raise ValueError("Omega, GT, and prediction must all be 2-D arrays.")
    if omega.shape != gt.shape or omega.shape != prediction.shape:
        raise ValueError("Omega, GT, and prediction shapes must be identical.")


def strict_counts(
    omega: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    *,
    method: str,
) -> StrictCounts:
    """Score a map without shrinking the frozen Omega denominator.

    ``255`` in CTP is an abstention: it is wrong for OA and adds one FN to the
    GT class, with no semantic TP/FP.  SegEarth clutter/ignore/non-target
    labels use the same five-class accounting but are recorded separately as
    nonsemantic predictions; their GT class still receives an FN.
    """
    _validate_shapes(omega, gt, prediction)
    if method not in {"CTP-v1", "SegEarth-OV"}:
        raise ValueError(f"Unknown method: {method}")
    mask = omega.astype(bool, copy=False)
    truth = gt[mask].astype(np.int64, copy=False)
    pred = prediction[mask].astype(np.int64, copy=False)
    if truth.size and (np.any(truth < 0) or np.any(truth >= N_CLASSES)):
        raise ValueError("Frozen Omega contains non-five-class GT labels.")
    valid = (pred >= 0) & (pred < N_CLASSES)
    if method == "CTP-v1" and np.any((pred != IGNORE) & ~valid):
        raise ValueError("CTP frozen maps may contain only 0..4 or 255.")
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    if np.any(valid):
        np.add.at(confusion, (truth[valid], pred[valid]), 1)
    invalid_truth = truth[~valid]
    fn_extra = np.bincount(invalid_truth, minlength=N_CLASSES).astype(np.int64, copy=False)
    correct = int(np.trace(confusion))
    abstentions = int(np.count_nonzero(pred == IGNORE)) if method == "CTP-v1" else 0
    return StrictCounts(
        confusion=confusion,
        fn_extra=fn_extra,
        denominator=int(truth.size),
        correct=correct,
        semantic_predictions=int(np.count_nonzero(valid)),
        nonsemantic_predictions=int(np.count_nonzero(~valid)),
        ctp_abstentions=abstentions,
    )


def metrics(counts: StrictCounts) -> dict[str, object]:
    """Compute OA and unweighted five-class F1/IoU using integer counts."""
    tp = np.diag(counts.confusion).astype(np.int64, copy=False)
    fp = counts.confusion.sum(axis=0).astype(np.int64, copy=False) - tp
    fn = counts.confusion.sum(axis=1).astype(np.int64, copy=False) - tp + counts.fn_extra
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, tp + fp, out=np.zeros(N_CLASSES, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(N_CLASSES, dtype=float), where=(tp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(N_CLASSES, dtype=float), where=(2 * tp + fp + fn) > 0)
        iou = np.divide(tp, tp + fp + fn, out=np.zeros(N_CLASSES, dtype=float), where=(tp + fp + fn) > 0)
    per_class = {
        name: {
            "Precision": float(precision[index]), "Recall": float(recall[index]),
            "F1": float(f1[index]), "IoU": float(iou[index]),
            "TP": int(tp[index]), "FP": int(fp[index]), "FN": int(fn[index]),
        }
        for index, name in enumerate(CLASSES)
    }
    return {
        "OA": float(counts.correct / counts.denominator) if counts.denominator else 0.0,
        "Macro_F1": float(f1.mean()),
        "mIoU": float(iou.mean()),
        "denominator": counts.denominator,
        "correct": counts.correct,
        "semantic_predictions": counts.semantic_predictions,
        "nonsemantic_predictions": counts.nonsemantic_predictions,
        "ctp_abstentions": counts.ctp_abstentions,
        "ctp_abstention_ratio": float(counts.ctp_abstentions / counts.denominator) if counts.denominator else 0.0,
        "per_class": per_class,
    }


def bootstrap_area_deltas(
    ctp_by_area: dict[int, StrictCounts],
    segearth_by_area: dict[int, StrictCounts],
    *,
    seed: int = 42,
    repeats: int = 5000,
) -> dict[str, object]:
    """Area-cluster bootstrap; descriptive only because n=5 areas."""
    if tuple(sorted(ctp_by_area)) != TEST_AREAS or tuple(sorted(segearth_by_area)) != TEST_AREAS:
        raise ValueError("Area clusters must be exactly the frozen five Vaihingen areas.")
    if seed != 42 or repeats != 5000:
        raise ValueError("Frozen bootstrap requires seed=42 and repeats=5000.")
    names = ("OA", "Macro_F1", "mIoU")

    def aggregate(source: dict[int, StrictCounts], drawn: np.ndarray) -> StrictCounts:
        first = source[int(drawn[0])]
        result = StrictCounts(np.zeros_like(first.confusion), np.zeros_like(first.fn_extra), 0, 0, 0, 0, 0)
        for area in drawn:
            result = result.add(source[int(area)])
        return result

    areas = np.asarray(TEST_AREAS, dtype=np.int64)
    point_ctp, point_seg = metrics(aggregate(ctp_by_area, areas)), metrics(aggregate(segearth_by_area, areas))
    point = {name: float(point_ctp[name] - point_seg[name]) for name in names}
    values = {name: np.empty(repeats, dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)
    for index in range(repeats):
        drawn = rng.choice(areas, size=len(areas), replace=True)
        ctp_metric, seg_metric = metrics(aggregate(ctp_by_area, drawn)), metrics(aggregate(segearth_by_area, drawn))
        for name in names:
            values[name][index] = float(ctp_metric[name] - seg_metric[name])
    return {
        "comparison": "CTP-v1_minus_SegEarth-OV",
        "cluster_unit": "Vaihingen test area",
        "areas": list(TEST_AREAS), "seed": seed, "repeats": repeats,
        "point_estimate": point,
        "bootstrap": {
            name: {"mean": float(value.mean()), "ci95_low": float(np.quantile(value, 0.025)), "ci95_high": float(np.quantile(value, 0.975))}
            for name, value in values.items()
        },
        "direction_by_area": {
            str(area): {name: float(metrics(ctp_by_area[area])[name] - metrics(segearth_by_area[area])[name]) for name in names}
            for area in TEST_AREAS
        },
        "interpretation": "Descriptive area-cluster bootstrap only; five clusters do not support a large-sample significance claim.",
    }
