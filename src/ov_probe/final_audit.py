"""Final review-defense audits (frozen CTP-v1, no new methods):

1. Guard pixel semantic maps for partial-support (frozen region-level Guard rule).
2. Common-pixel fairness audit (Omega_common intersection re-scoring).
3. Correct cluster-level bootstrap (area/tile/image clusters, not patches).

All inputs are frozen predictions/scores/masks/subsets/GT; no re-inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import InputValidationError
from .pixel_ovss import IGNORE_INDEX, assemble_semantic_map, load_candidate_masks
from .loveda_partial_support import guard_predictions

CLASSES_VAH = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
CLASSES_POT = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
METHODS = ["text_only", "C2", "SCC", "CTP", "guard"]


def build_guard_semantic_maps(
    run_root: str | Path,
    candidates_dir: str | Path,
    predictions_npz: str | Path,
    records_jsonl: str | Path,
    subsets: dict[str, dict],
    classes: list[str],
    only_images: set[str] | None = None,
) -> dict[str, int]:
    """Generate Guard pixel semantic maps for all partial-support subsets.

    Frozen Guard rule (region-level): if text-only top-1 is an unsupported class,
    keep the text prediction; otherwise use C2 anchored competition. No threshold,
    margin, temperature, or new gating. Fusion uses the Guard-decision class with
    the C2 anchored score (calibrated competition) / text score (preserved class).
    """
    run_root = Path(run_root)
    d = np.load(predictions_npz, allow_pickle=False)
    text_scores_all = d["text_scores"].astype(np.float32)
    visual_scores_all = d["visual_scores"].astype(np.float32)
    anchored_all = d["anchored_scores"].astype(np.float32)
    text_pred_all = d["text_pred"].astype(np.int64)
    records = [json.loads(line) for line in Path(records_jsonl).open(encoding="utf-8")]

    # For Potsdam, records have {image_id, candidate_count}; for Vaihingen they have
    # {row_index, image_id, candidate_index}. Build per-image candidate order.
    from collections import OrderedDict
    by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    image_ids = sorted(by_image)
    if only_images is not None:
        image_ids = [i for i in image_ids if i in only_images]

    # global score index: Vaihingen uses row_index->pos; Potsdam uses cumulative candidate counts
    if "row_index" in records[0]:
        pos_by_row = {r["row_index"]: i for i, r in enumerate(records)}
    else:
        pos_by_row = None

    counts = {}
    for key, info in subsets.items():
        mask = np.asarray([c in info["supported"] for c in classes], dtype=bool)
        c2 = anchored_all.copy()
        c2[:, ~mask] = text_scores_all[:, ~mask]
        u_idx = [i for i, flag in enumerate(mask) if not flag]
        guard_pred_all = guard_predictions(text_pred_all, c2, u_idx)
        for image_id in image_ids:
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            if pos_by_row is not None:
                # Vaihingen: region index -> row_index -> global pos
                region_by_index = {int(r["candidate_index"]): r for r in by_image[image_id]}
                ordered = []
                for index in range(len(regions)):
                    record = region_by_index[index]
                    pos = pos_by_row[record["row_index"]]
                    ordered.append({"mask": regions[index]["mask"], "x0": regions[index]["x0"],
                                    "y0": regions[index]["y0"], "pos": pos})
            else:
                # Potsdam: candidate order within image = global cumulative order
                base = sum(records[j]["candidate_count"] for j in range(image_ids.index(image_id)))
                ordered = []
                for index in range(len(regions)):
                    ordered.append({"mask": regions[index]["mask"], "x0": regions[index]["x0"],
                                    "y0": regions[index]["y0"], "pos": base + index})
            pred_sel = np.asarray([guard_pred_all[o["pos"]] for o in ordered], dtype=np.int64)
            score_sel = np.asarray([float(c2[o["pos"], guard_pred_all[o["pos"]]]) for o in ordered], dtype=np.float32)
            label_map, _ = assemble_semantic_map(shape, ordered, pred_sel, score_sel, classes)
            path = run_root / f"{key}_guard_{image_id}_semantic.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, label_map=label_map)
            counts[path.name] = counts.get(path.name, 0) + 1
    return counts


def load_label_maps(run_root: str | Path, key: str, method: str, image_ids: list[str]) -> dict[str, np.ndarray]:
    run_root = Path(run_root)
    maps = {}
    for image_id in image_ids:
        path = run_root / f"{key}_{method}_{image_id}_semantic.npz"
        if not path.is_file():
            raise InputValidationError(f"Missing semantic map: {path}")
        with np.load(path, allow_pickle=False) as archive:
            maps[image_id] = archive["label_map"].astype(np.int64)
    return maps


def per_image_confusion(pred_maps: dict[str, np.ndarray], gt_maps: dict[str, np.ndarray], classes: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per-image confusion matrices on the common valid-pixel intersection.

    Returns {method: {image_id: C x C int64 matrix}} where each matrix counts
    (gt, pred) pairs over pixels that are GT-valid and valid for EVERY method
    (identical mask across methods by construction).
    """
    common_valid = {}
    for image_id in gt_maps:
        valid = gt_maps[image_id] != IGNORE_INDEX
        for method in pred_maps:
            valid = valid & (pred_maps[method][image_id] != IGNORE_INDEX)
        common_valid[image_id] = valid
    c = len(classes)
    out = {method: {} for method in pred_maps}
    for image_id in gt_maps:
        valid = common_valid[image_id]
        gt_v = gt_maps[image_id][valid]
        for method, maps in pred_maps.items():
            pred_v = maps[image_id][valid]
            flat = gt_v.astype(np.int64) * c + pred_v
            matrix = np.bincount(flat, minlength=c * c).reshape(c, c).astype(np.int64)
            out[method][image_id] = matrix
    return out


def metrics_from_matrix(matrix: np.ndarray, classes: list[str], supported: list[str], unsupported: list[str]) -> dict[str, float]:
    """S/U/H-F1, S/U/H-IoU, OA, macro F1, mIoU from one confusion matrix."""
    per_iou, per_f1 = {}, {}
    for i, name in enumerate(classes):
        tp = float(matrix[i, i]); fp = float(matrix[:, i].sum() - tp); fn = float(matrix[i, :].sum() - tp)
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    s_iou = float(np.mean([per_iou[c] for c in supported]))
    u_iou = float(np.mean([per_iou[c] for c in unsupported]))
    s_f1 = float(np.mean([per_f1[c] for c in supported]))
    u_f1 = float(np.mean([per_f1[c] for c in unsupported]))
    h_iou = 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou > 0 else 0.0
    h_f1 = 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 > 0 else 0.0
    total = float(matrix.sum())
    return {
        "OA": float(np.trace(matrix)) / total if total else 0.0,
        "macro_f1": float(np.mean(list(per_f1.values()))),
        "mIoU": float(np.mean(list(per_iou.values()))),
        "S_F1": s_f1, "U_F1": u_f1, "H_F1": h_f1,
        "S_IoU": s_iou, "U_IoU": u_iou, "H_IoU": h_iou,
    }


def common_pixel_metrics(
    pred_maps: dict[str, np.ndarray],
    gt_maps: dict[str, np.ndarray],
    classes: list[str],
    supported: list[str],
    unsupported: list[str],
) -> dict[str, Any]:
    """Compute metrics on the common valid-pixel intersection of all methods."""
    per_image = per_image_confusion(pred_maps, gt_maps, classes)
    results = {}
    for method, images in per_image.items():
        matrix = sum(images.values(), np.zeros((len(classes), len(classes)), dtype=np.int64))
        metrics = metrics_from_matrix(matrix, classes, supported, unsupported)
        metrics["common_valid_pixels"] = int(matrix.sum())
        results[method] = metrics
    return results


def per_method_valid_pixels(pred_maps: dict[str, np.ndarray], gt_maps: dict[str, np.ndarray]) -> dict[str, int]:
    out = {}
    for method, maps in pred_maps.items():
        total = 0
        for image_id in gt_maps:
            valid = (gt_maps[image_id] != IGNORE_INDEX) & (maps[image_id] != IGNORE_INDEX)
            total += int(valid.sum())
        out[method] = total
    return out


__all__ = [
    "METHODS",
    "build_guard_semantic_maps",
    "common_pixel_metrics",
    "load_label_maps",
    "metrics_from_matrix",
    "per_image_confusion",
    "per_method_valid_pixels",
]
