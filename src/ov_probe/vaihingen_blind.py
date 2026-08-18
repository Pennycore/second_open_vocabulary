"""Blind Vaihingen evaluation of SCC-v1 (predict + evaluate phases).

Predict phase (no GT access):
1. Load frozen SAM3 candidate caches for all 16 GT areas (weak proposals).
2. Build region pixel views (crop around candidate mask, like LoveDA pixel pack).
3. Encode regions with frozen OpenAI CLIP ViT-B/32 quick-GELU.
4. Build text prototypes (8 Group-A templates, Vaihingen vocabulary) and
   visual prototypes (train-area SAM3 weak labels only).
5. Score test-area regions: text-only, visual-only, C2, SCC, Text-Top1 Guard,
   and all 2^5 partial-support subsets.
6. Persist predictions + support manifest; hashes are recorded before any GT read.

Evaluate phase (GT unlocked only after predict manifest verified):
- Read official Vaihingen GT (RGB color map), derive region GT by majority
  vote inside the candidate mask; clutter/ignore pixels never vote.
- Compute OA / Macro F1 / mIoU, per-class P/R/F1/IoU, confusion matrix,
  S/U/H-F1, S/U/H-IoU, image-cluster bootstrap (seed 42, 5000 repeats).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .io import InputValidationError, sha256_file, write_json
from .loveda_partial_support import (
    _c2_normalizers,
    _metrics,
    guard_predictions,
    scc_scores,
)
from .openai_clip_visual_anchor import _normalize

CLASSES = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
GT_COLOR_MAP = {
    "impervious_surface": (255, 255, 255),
    "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255),
    "tree": (0, 255, 0),
    "car": (255, 255, 0),
}
CLUTTER_COLOR = (255, 0, 0)
TRAIN_AREAS = [1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37]
TEST_AREAS = [11, 15, 28, 30, 34]


def _area_from_id(image_id: str) -> int:
    prefix = "vaih_area"
    if not image_id.startswith(prefix):
        raise InputValidationError(f"Unexpected Vaihingen image id: {image_id}")
    suffix = image_id[len(prefix):]
    if not suffix.isdigit():
        raise InputValidationError(f"Unexpected Vaihingen image id: {image_id}")
    return int(suffix)


def _load_candidates(cache_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load all candidate caches; returns image_id -> candidate records."""
    import sys
    sys.path.insert(0, "/home/undergr/Sheungzhen_project_1/sam3_remote_wsss/src")
    from sam3_remote_wsss.candidate_cache import load_candidate_cache

    root = Path(cache_dir)
    result: dict[str, list[dict[str, Any]]] = {}
    for data_path in sorted(root.glob("*.npz")):
        image_id = data_path.name[:-4]
        metadata, candidates = load_candidate_cache(root, image_id)
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append({
                "index": index,
                "class_id": int(candidate.class_id),
                "class_name": str(candidate.class_name),
                "prompt": str(candidate.prompt),
                "score": float(candidate.score),
                "mask": np.asarray(candidate.mask, dtype=bool),
                "x0": int(candidate.x0),
                "y0": int(candidate.y0),
            })
        result[image_id] = rows
    return result


def _crop_view(image_rgb: np.ndarray, mask: np.ndarray, x0: int, y0: int,
               context_ratio: float = 0.25, min_crop_size: int = 48,
               background_retain: float = 0.25) -> dict[str, Any]:
    height, width = mask.shape
    image_height, image_width = image_rgb.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + width > image_width or y0 + height > image_height:
        raise InputValidationError("Candidate bounds exceed the source image.")
    ys, xs = np.nonzero(mask)
    left, top = x0 + int(xs.min()), y0 + int(ys.min())
    right, bottom = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
    target_size = max(right - left, bottom - top)
    crop_size = max(int(min_crop_size), int(np.ceil(target_size * (1 + 2 * context_ratio))))

    def centered(center: float, size: int, limit: int) -> tuple[int, int]:
        bounded = min(int(size), int(limit))
        start = int(np.floor(center - bounded / 2))
        start = max(0, min(start, limit - bounded))
        return start, start + bounded

    crop_left, crop_right = centered((left + right) / 2, crop_size, image_width)
    crop_top, crop_bottom = centered((top + bottom) / 2, crop_size, image_height)
    crop = np.ascontiguousarray(image_rgb[crop_top:crop_bottom, crop_left:crop_right])
    crop_mask = np.zeros(crop.shape[:2], dtype=bool)
    il, it = max(crop_left, x0), max(crop_top, y0)
    ir = min(crop_right, x0 + width)
    ib = min(crop_bottom, y0 + height)
    if il < ir and it < ib:
        crop_mask[it - crop_top: ib - crop_top, il - crop_left: ir - crop_left] = mask[
            it - y0: ib - y0, il - x0: ir - x0
        ]
    masked = crop.astype(np.float32)
    masked[~crop_mask] *= float(background_retain)
    masked = np.rint(masked).clip(0, 255).astype(np.uint8)
    return {
        "context": crop,
        "crop_mask": np.ascontiguousarray(crop_mask),
        "masked": np.ascontiguousarray(masked),
        "crop_box": (crop_left, crop_top, crop_right, crop_bottom),
        "mask_fraction": float(crop_mask.mean()),
    }


def _encode_regions(image_rgb: np.ndarray, candidates: list[dict[str, Any]],
                    model, preprocess, device, batch: int = 32) -> np.ndarray:
    import torch
    from PIL import Image
    features = np.empty((len(candidates), 512), dtype=np.float32)
    for start in range(0, len(candidates), batch):
        views = [_crop_view(image_rgb, c["mask"], c["x0"], c["y0"]) for c in candidates[start:start + batch]]
        tensors = torch.stack([preprocess(Image.fromarray(view["masked"])).to(device) for view in views])
        with torch.inference_mode():
            encoded = model.encode_image(tensors).float().cpu().numpy()
        features[start:start + batch] = _normalize(encoded)
    return features


def _text_prototypes(protocol: dict[str, Any], checkpoint: Path, device: str) -> tuple[np.ndarray, str]:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise InputValidationError("Vaihingen text construction requires torch and open_clip.") from exc
    if getattr(open_clip, "__version__", None) != protocol["model"]["open_clip_version"]:
        raise InputValidationError("OpenCLIP version differs from the frozen protocol.")
    model, _, _ = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.eval().to(device)
    templates = protocol["prompts"]["group_a_templates"]
    classes = protocol["classes"]
    texts = [template.format(**{"class": name}) for name in classes for template in templates]
    tokenizer = open_clip.get_tokenizer(protocol["model"]["architecture"])
    tokens = tokenizer(texts)
    token_hash = hashlib.sha256(tokens.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    with torch.inference_mode():
        encoded = model.encode_text(tokens.to(device)).float().cpu().numpy()
    encoded = _normalize(encoded).reshape(len(classes), len(templates), -1)
    return _normalize(encoded.mean(axis=1)), token_hash


def _region_gt(candidates: list[dict[str, Any]], label_rgb: np.ndarray) -> list[str | None]:
    results: list[str | None] = []
    for candidate in candidates:
        mask = candidate["mask"]
        x0, y0 = candidate["x0"], candidate["y0"]
        height, width = mask.shape
        crop = label_rgb[y0:y0 + height, x0:x0 + width]
        if crop.shape[:2] != mask.shape:
            raise InputValidationError("GT crop geometry differs from candidate mask.")
        pixels = crop[mask]
        votes: dict[str, int] = {}
        for name, color in GT_COLOR_MAP.items():
            matches = (pixels == np.asarray(color, dtype=np.uint8)).all(axis=1)
            votes[name] = int(matches.sum())
        total = sum(votes.values())
        results.append(max(votes, key=lambda name: votes[name]) if total > 0 else None)
    return results


__all__ = [
    "CLASSES", "GT_COLOR_MAP", "CLUTTER_COLOR", "TRAIN_AREAS", "TEST_AREAS",
    "_crop_view", "_encode_regions", "_region_gt", "_text_prototypes",
]
