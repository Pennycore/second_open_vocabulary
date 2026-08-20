"""Independent RemoteCLIP Potsdam baseline runner.

This module intentionally reuses only frozen generic primitives (candidate masks,
FusionCanvas and CTP-v1 score rules). It never imports OpenAI CLIP outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .io import InputValidationError, sha256_file
from .loveda_partial_support import ctp_predictions, scc_scores
from .pixel_ovss import assemble_semantic_map, load_candidate_masks, pixel_confusion

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
COLORS = {
    "impervious_surface": (255, 255, 255),
    "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255),
    "tree": (0, 255, 0),
    "car": (255, 255, 0),
}
METHODS = ["text_only", "C2", "SCC", "CTP"]


def directory_sha256(path: Path, suffix: str) -> tuple[str, int]:
    """Hash an immutable input directory by relative path and file contents."""
    if not path.is_dir():
        raise InputValidationError(f"Required input directory is missing: {path}")
    files = sorted(item for item in path.glob(suffix) if item.is_file())
    if not files:
        raise InputValidationError(f"No {suffix} files found in required input directory: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def crop_views(image_rgb: np.ndarray, mask: np.ndarray, x0: int, y0: int,
               context_ratio: float = 0.25, min_crop_size: int = 48,
               background_retain: float = 0.25) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise InputValidationError("Cannot encode an empty candidate mask.")
    h, w = mask.shape
    ih, iw = image_rgb.shape[:2]
    left, top = x0 + int(xs.min()), y0 + int(ys.min())
    right, bottom = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
    size = max(min_crop_size, int(np.ceil(max(right-left, bottom-top) * (1 + 2 * context_ratio))))
    size = min(size, iw, ih)
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    cl = max(0, min(int(np.floor(cx - size / 2)), iw - size))
    ct = max(0, min(int(np.floor(cy - size / 2)), ih - size))
    cr, cb = cl + size, ct + size
    context = np.ascontiguousarray(image_rgb[ct:cb, cl:cr])
    local = np.zeros((size, size), dtype=bool)
    il, it, ir, ib = max(cl, x0), max(ct, y0), min(cr, x0 + w), min(cb, y0 + h)
    if il < ir and it < ib:
        local[it-ct:ib-ct, il-cl:ir-cl] = mask[it-y0:ib-y0, il-x0:ir-x0]
    masked = context.astype(np.float32)
    masked[~local] *= background_retain
    masked = np.rint(masked).clip(0, 255).astype(np.uint8)
    return context, np.ascontiguousarray(masked), (cl, ct, cr, cb)


def score_methods(text_scores: np.ndarray, visual_scores: np.ndarray,
                  text_prototypes: np.ndarray, visual_prototypes: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fused = 0.5 * text_prototypes + 0.5 * visual_prototypes
    norms = np.linalg.norm(fused, axis=1)
    if np.any(norms <= 1e-8):
        raise InputValidationError("RemoteCLIP C2 prototype has zero norm.")
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / norms[None, :]
    support = np.ones(len(CLASSES), dtype=bool)
    scc = scc_scores(text_scores, anchored, support)
    text_pred = np.argmax(text_scores, axis=1).astype(np.int64)
    ctp = ctp_predictions(text_pred, text_scores, scc, support)
    pred = {
        "text_only": text_pred,
        "C2": np.argmax(anchored, axis=1).astype(np.int64),
        "SCC": np.argmax(scc, axis=1).astype(np.int64),
        "CTP": ctp,
    }
    return pred, {"text_only": text_scores, "C2": anchored, "SCC": scc, "CTP": scc}


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(n <= 0):
        raise InputValidationError("RemoteCLIP features are non-finite or zero.")
    return x / n


def _load_model(checkpoint: Path, protocol: dict[str, Any], device: str):
    import open_clip
    import torch
    if open_clip.__version__ != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from RemoteCLIP protocol.")
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise InputValidationError("RemoteCLIP checkpoint hash differs from protocol.")
    model, _, preprocess = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise InputValidationError(f"RemoteCLIP checkpoint mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
    model.eval().to(device)
    return model, preprocess, open_clip.get_tokenizer(protocol["model"]["architecture"]), torch


def text_prototypes(model: Any, tokenizer: Any, protocol: dict[str, Any], device: str, torch: Any) -> tuple[np.ndarray, str]:
    texts = [t.format(**{"class": c}) for c in CLASSES for t in protocol["prompts"]["group_a_templates"]]
    tokens = tokenizer(texts).to(device)
    with torch.inference_mode():
        values = model.encode_text(tokens).float().cpu().numpy()
    values = _normalize(values).reshape(len(CLASSES), -1, 512)
    vectors = _normalize(values.mean(axis=1))
    return vectors, hashlib.sha256(tokenizer(texts).cpu().numpy().astype(np.int64).tobytes()).hexdigest()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("paths"), dict):
        raise InputValidationError("RemoteCLIP baseline config must contain a paths mapping.")
    project_root = config_path.parents[1]
    for key, value in list(cfg["paths"].items()):
        if value is None:
            continue
        path = Path(os.path.expandvars(str(value)))
        cfg["paths"][key] = str((path if path.is_absolute() else project_root / path).resolve())
    protocol_path = Path(cfg["paths"]["protocol_file"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("RemoteCLIP baseline must set overwrite=false.")
    required = {"candidates_dir", "image_dir", "label_dir", "checkpoint", "output_root"}
    missing = sorted(name for name in required if not cfg["paths"].get(name))
    if missing:
        raise InputValidationError(f"RemoteCLIP baseline missing required paths: {missing}")
    if list(protocol.get("classes", [])) != CLASSES or protocol.get("methods") != METHODS:
        raise InputValidationError("Protocol classes or method matrix differs from the frozen baseline.")
    if protocol.get("alpha") != 0.5:
        raise InputValidationError("RemoteCLIP baseline alpha must remain frozen at 0.5.")
    return cfg, protocol


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = np.sum([np.asarray(r["confusion_matrix"], dtype=np.int64) for r in results], axis=0)
    per_iou, per_f1 = {}, {}
    for i, name in enumerate(CLASSES):
        tp = float(matrix[i, i]); fp = float(matrix[:, i].sum()-tp); fn = float(matrix[i, :].sum()-tp)
        per_iou[name] = tp/(tp+fp+fn) if tp+fp+fn else 0.0
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        per_f1[name] = 2*p*r/(p+r) if p+r else 0.0
    valid = int(matrix.sum())
    return {"OA": float(np.trace(matrix)/valid) if valid else 0.0, "macro_f1": float(np.mean(list(per_f1.values()))), "mIoU": float(np.mean(list(per_iou.values()))), "per_class_f1": per_f1, "per_class_iou": per_iou, "confusion_matrix": matrix.tolist(), "valid_pixels": valid}


__all__ = ["CLASSES", "METHODS", "COLORS", "crop_views", "score_methods", "load_config", "directory_sha256", "_load_model", "text_prototypes", "_normalize", "_aggregate"]
