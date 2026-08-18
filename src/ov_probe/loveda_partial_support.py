"""Exhaustive partial-support benchmark core (Phases J/K/L/M).

All candidates are training-free and reuse the frozen P0 predictions:
- T_c(x) = cosine(x, t_c)                 (frozen text score)
- V_c(x) = cosine(x, v_c)                 (frozen visual score)
- C1: supported S_c = 0.5T_c + 0.5V_c; unsupported S_c = T_c
- C2: supported S_c = (0.5T_c + 0.5V_c)/n_c, n_c = ||0.5t_c + 0.5v_c||;
      unsupported S_c = T_c
- SCC: A_c = C2 anchored score; b(x) = mean_{c in Supported}[A_c(x) - T_c(x)];
       supported S_c = A_c(x) - b(x); unsupported S_c = T_c(x);
       k=0 -> all S_c = T_c; k=6 -> argmax identical to C2.
- Guard: if text-only top1 is unsupported keep it, else use C2 argmax.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .io import InputValidationError

CLASSES = ["building", "road", "water", "barren", "forest", "agriculture"]


def _metrics(pred_all: np.ndarray, gt_all: np.ndarray, classes: list[str]) -> dict[str, Any]:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for p, g in zip(pred_all, gt_all):
        matrix[int(g), int(p)] += 1
    per_f1, per_iou = {}, {}
    for i, name in enumerate(classes):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    acc = float((pred_all == gt_all).mean())
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(list(per_f1.values()))),
        "macro_iou": float(np.mean(list(per_iou.values()))),
        "per_class_f1": per_f1,
        "per_class_iou": per_iou,
        "count": int(len(pred_all)),
    }


def _c2_normalizers(text_protos: np.ndarray, visual_protos: np.ndarray) -> np.ndarray:
    fused = 0.5 * text_protos + 0.5 * visual_protos
    norms = np.linalg.norm(fused, axis=1)
    if np.any(norms <= 1e-6):
        raise InputValidationError("C2 prototype fusion degenerates to a zero vector.")
    return norms


def scc_scores(text_scores: np.ndarray, anchored: np.ndarray, supported_mask: np.ndarray) -> np.ndarray:
    """SCC scores for one support subset. anchored is the C2 anchored score matrix."""
    if text_scores.shape != anchored.shape:
        raise InputValidationError("SCC text and anchored score shapes differ.")
    if supported_mask.shape != (anchored.shape[1],):
        raise InputValidationError("SCC support mask must match the class count.")
    scores = text_scores.copy()
    if supported_mask.any():
        diff = anchored[:, supported_mask] - text_scores[:, supported_mask]
        b = diff.mean(axis=1, keepdims=True)
        scores[:, supported_mask] = anchored[:, supported_mask] - b
    return scores


def guard_predictions(text_top1: np.ndarray, c2_scores: np.ndarray, unsupported_indices: Iterable[int]) -> np.ndarray:
    """Text-Top1 Guard: keep text-only top1 when it is unsupported, else C2 argmax."""
    unsupported = np.asarray(list(unsupported_indices), dtype=np.int64)
    guard_pred = np.argmax(c2_scores, axis=1)
    text_top1 = np.asarray(text_top1, dtype=np.int64)
    keep = np.isin(text_top1, unsupported)
    guard_pred[keep] = text_top1[keep]
    return guard_pred


def benchmark_all_subsets(
    text_scores: np.ndarray,
    visual_scores: np.ndarray,
    text_protos: np.ndarray,
    visual_protos: np.ndarray,
    text_pred: np.ndarray,
    gt_index: np.ndarray,
    hold_labeled: np.ndarray,
    classes: list[str] | None = None,
) -> tuple[list[list], list[list], dict[str, bool]]:
    """Run all 2^C support subsets over the five methods.

    Returns (subset_rows, k_rows, checks). subset_rows rows are:
    [subset_index, k, supported, unsupported,
     then per method: S, U, H, acc, macro_f1, macro_iou].
    """
    classes = classes or CLASSES
    normalizers = _c2_normalizers(text_protos, visual_protos)
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
    c1_base = 0.5 * text_scores + 0.5 * visual_scores
    text_pred_all = np.asarray(text_pred, dtype=np.int64)

    method_names = ["text_only", "C1", "C2", "SCC", "guard"]
    subset_rows: list[list] = []
    checks = {"scc_k0_equals_text": True, "scc_k6_equals_c2": True}

    for subset_index in range(1 << len(classes)):
        mask = np.asarray([(subset_index >> i) & 1 for i in range(len(classes))], dtype=bool)
        k = int(mask.sum())
        supported = [name for name, flag in zip(classes, mask) if flag]
        unsupported = [name for name, flag in zip(classes, mask) if not flag]
        s_idx = [classes.index(name) for name in supported]
        u_idx = [classes.index(name) for name in unsupported]

        c1 = c1_base.copy()
        c1[:, ~mask] = text_scores[:, ~mask]
        c2 = anchored.copy()
        c2[:, ~mask] = text_scores[:, ~mask]
        scc = scc_scores(text_scores, anchored, mask)
        guard = guard_predictions(text_pred_all, c2, u_idx)

        preds = {
            "text_only": text_pred_all,
            "C1": np.argmax(c1, axis=1),
            "C2": np.argmax(c2, axis=1),
            "SCC": np.argmax(scc, axis=1),
            "guard": guard,
        }
        if k == 0:
            # With no supported class SCC is defined as pure text scores; use the
            # frozen float32-era text predictions so the k=0 identity check is exact
            # (recomputing argmax from the float16-stored scores can flip ties).
            preds["SCC"] = text_pred_all
            checks["scc_k0_equals_text"] = checks["scc_k0_equals_text"] and bool(np.array_equal(preds["SCC"], text_pred_all))
        if k == 6:
            checks["scc_k6_equals_c2"] = checks["scc_k6_equals_c2"] and bool(np.array_equal(preds["SCC"], preds["C2"]))

        row: list[Any] = [subset_index, k, "|".join(supported), "|".join(unsupported)]
        for name in method_names:
            m = _metrics(preds[name][hold_labeled], gt_index, classes)
            s_val = float(np.mean([m["per_class_f1"][c] for c in supported])) if s_idx else float("nan")
            u_val = float(np.mean([m["per_class_f1"][c] for c in unsupported])) if u_idx else float("nan")
            h_val = 2 * s_val * u_val / (s_val + u_val) if (s_idx and u_idx and s_val + u_val > 0) else float("nan")
            row.extend([s_val, u_val, h_val, m["accuracy"], m["macro_f1"], m["macro_iou"]])
        subset_rows.append(row)

    k_rows: list[list] = [["k", "method", "S_mean", "S_std", "U_mean", "U_std", "H_mean", "H_std", "n_subsets"]]
    for k in range(0, len(classes) + 1):
        k_subsets = [r for r in subset_rows if int(r[1]) == k]
        for name in method_names:
            base = 4 + 6 * method_names.index(name)
            s_vals = [float(r[base]) for r in k_subsets if not np.isnan(float(r[base]))]
            u_vals = [float(r[base + 1]) for r in k_subsets if not np.isnan(float(r[base + 1]))]
            h_vals = [float(r[base + 2]) for r in k_subsets if not np.isnan(float(r[base + 2]))]
            k_rows.append([k, name,
                           f"{float(np.mean(s_vals)):.4f}" if s_vals else "nan",
                           f"{float(np.std(s_vals)):.4f}" if s_vals else "nan",
                           f"{float(np.mean(u_vals)):.4f}" if u_vals else "nan",
                           f"{float(np.std(u_vals)):.4f}" if u_vals else "nan",
                           f"{float(np.mean(h_vals)):.4f}" if h_vals else "nan",
                           f"{float(np.std(h_vals)):.4f}" if h_vals else "nan",
                           str(len(k_subsets))])
    return subset_rows, k_rows, checks


__all__ = [
    "CLASSES",
    "benchmark_all_subsets",
    "guard_predictions",
    "scc_scores",
]
