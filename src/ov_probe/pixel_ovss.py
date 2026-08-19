"""Pixel-level OVSS pipeline (protocol v0): SAM3 candidate masks + frozen OpenAI CLIP
scores -> FusionCanvas semantic maps -> hashed predictions -> GT evaluation.

Implements the frozen protocol configs/pixel_ovss_protocol_v0.json:
- proposal source: first-paper SAM3 candidate caches (no new proposal network)
- semantic assignment: OpenAI CLIP + Text-only / C2 / SCC / CTP frozen formulas
- fusion: per-pixel highest region score wins; conflict margin 0.03 marks ignore;
  uncovered pixels = 255; no background class (uncovered=ignore)
- GT isolation: predictions + config + hashes persisted before any GT read
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io import InputValidationError, sha256_file, write_json
from .loveda_partial_support import ctp_predictions, scc_scores

IGNORE_INDEX = 255
CONFLICT_MARGIN = 0.03


@dataclass
class FusionCanvas:
    """First-paper FusionCanvas semantics (per-pixel highest score wins)."""

    height: int
    width: int
    ignore_index: int = IGNORE_INDEX
    uncovered_label: int = IGNORE_INDEX
    conflict_margin: float = CONFLICT_MARGIN
    labels: np.ndarray = field(init=False)
    scores: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.labels = np.full((self.height, self.width), self.uncovered_label, dtype=np.uint8)
        self.scores = np.zeros((self.height, self.width), dtype=np.float32)

    def add_mask(self, mask: np.ndarray, class_id: int, score: float, x0: int, y0: int) -> None:
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return
        h, w = mask_bool.shape
        region_labels = self.labels[y0:y0 + h, x0:x0 + w]
        region_scores = self.scores[y0:y0 + h, x0:x0 + w]
        better = mask_bool & (score > region_scores + self.conflict_margin)
        close_conflict = (
            mask_bool
            & (region_scores > 0)
            & (region_labels != class_id)
            & (np.abs(score - region_scores) <= self.conflict_margin)
        )
        region_labels[better] = class_id
        region_scores[better] = score
        region_labels[close_conflict] = self.ignore_index

    def result(self) -> np.ndarray:
        return self.labels


def method_score_matrices(
    text_scores: np.ndarray,
    anchored: np.ndarray,
    mask: np.ndarray,
    text_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Frozen per-method final score matrices (same shape as text_scores)."""
    scc = scc_scores(text_scores, anchored, mask)
    ctp = ctp_predictions(text_pred, text_scores, scc, mask)
    c2 = anchored.copy()
    c2[:, ~mask] = text_scores[:, ~mask]
    # For CTP, kept text predictions carry their text score (SCC unsupported = T);
    # the argmax class of the CTP prediction is what matters for the label map.
    return {
        "text_only": text_scores,
        "C2": c2,
        "SCC": scc,
        "CTP": scc,  # scores unchanged; CTP only alters argmax decisions
    }


def method_predictions(scores: dict[str, np.ndarray], text_pred: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    scc = scores["SCC"]
    ctp = ctp_predictions(text_pred, scores["text_only"], scc, mask)
    return {
        "text_only": np.argmax(scores["text_only"], axis=1).astype(np.int64),
        "C2": np.argmax(scores["C2"], axis=1).astype(np.int64),
        "SCC": np.argmax(scc, axis=1).astype(np.int64),
        "CTP": ctp.astype(np.int64),
    }


def assemble_semantic_map(
    image_shape: tuple[int, int],
    regions: list[dict[str, Any]],
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Assemble one per-method semantic map from region masks and per-region argmax.

    regions: list of dicts with mask (bool HxW at full-image coords), x0, y0, row_index.
    predictions: per-region class index (full region set, row_index order).
    scores: per-region final score of the predicted class (row_index order).
    """
    height, width = image_shape
    canvas = FusionCanvas(height=height, width=width)
    covered = 0
    for index, region in enumerate(regions):
        class_id = int(predictions[index])
        score = float(scores[index])
        mask = np.asarray(region["mask"], dtype=bool)
        x0, y0 = int(region["x0"]), int(region["y0"])
        if x0 < 0 or y0 < 0 or x0 + mask.shape[1] > width or y0 + mask.shape[0] > height:
            raise InputValidationError("Candidate mask exceeds image bounds.")
        canvas.add_mask(mask, class_id=class_id, score=score, x0=x0, y0=y0)
        covered += int(mask.sum())
    label_map = canvas.result()
    stats = {
        "pixels_total": int(height * width),
        "pixels_covered": covered,
        "pixels_labeled": int((label_map != IGNORE_INDEX).sum()),
        "pixels_uncovered": int((label_map == IGNORE_INDEX).sum()),
    }
    return label_map, stats


def pixel_confusion(pred_map: np.ndarray, gt_map: np.ndarray, classes: list[str], ignore: int = IGNORE_INDEX) -> dict[str, Any]:
    """Pixel-level metrics from aligned pred/GT label maps (same ignore convention)."""
    valid = (gt_map != ignore) & (pred_map != ignore)
    pred = pred_map[valid]
    gt = gt_map[valid]
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for p, g in zip(pred, gt):
        if 0 <= g < len(classes) and 0 <= p < len(classes):
            matrix[int(g), int(p)] += 1
    per_iou: dict[str, float] = {}
    per_f1: dict[str, float] = {}
    for i, name in enumerate(classes):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    oa = float((pred == gt).mean()) if len(pred) else 0.0
    return {
        "OA": oa,
        "macro_f1": float(np.mean(list(per_f1.values()))),
        "mIoU": float(np.mean(list(per_iou.values()))),
        "per_class_iou": per_iou,
        "per_class_f1": per_f1,
        "confusion_matrix": matrix.tolist(),
        "valid_pixels": int(len(pred)),
        "ignore_pixels": int((~valid).sum()),
    }


def load_candidate_masks(candidates_dir: str | Path, image_id: str) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Load first-paper candidate cache; returns (image_shape, regions with full-image masks)."""
    import sys
    sys.path.insert(0, "/home/undergr/Sheungzhen_project_1/sam3_remote_wsss/src")
    from sam3_remote_wsss.candidate_cache import load_candidate_cache

    metadata, candidates = load_candidate_cache(Path(candidates_dir), image_id)
    shape = tuple(int(v) for v in metadata["image_shape"])
    regions = []
    for index, candidate in enumerate(candidates):
        regions.append({
            "index": index,
            "mask": np.asarray(candidate.mask, dtype=bool),
            "x0": int(candidate.x0),
            "y0": int(candidate.y0),
            "class_name": str(candidate.class_name),
            "score": float(candidate.score),
        })
    return shape, regions


__all__ = [
    "FusionCanvas",
    "assemble_semantic_map",
    "load_candidate_masks",
    "method_predictions",
    "method_score_matrices",
    "pixel_confusion",
]
