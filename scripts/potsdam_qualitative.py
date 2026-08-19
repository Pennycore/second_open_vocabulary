"""Phase G: qualitative visualization for Potsdam CTP-v1 confirmation.

Selection rule (recorded, not cherry-picked): pick patches from the C2 full-support
run where C2 mislabels a large unsupported-class area (e.g., car or low_vegetation
assigned to building/impervious) AND CTP recovers the text class. Concretely:
for each patch compute the pixel count where C2 == wrong class and CTP == GT class
(after GT read, but the rule is fixed beforehand), rank by that count, take top 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError  # noqa: E402
from ov_probe.vaihingen_blind import CLASSES  # noqa: E402

IGNORE = 255
GT_COLOR_MAP = {
    "impervious_surface": (255, 255, 255),
    "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255),
    "tree": (0, 255, 0),
    "car": (255, 255, 0),
}
CLASS_COLORS = {
    "impervious_surface": (128, 128, 128),
    "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255),
    "tree": (0, 255, 0),
    "car": (255, 255, 0),
}
UNSUPPORTED_POOL = ["car", "low_vegetation", "tree"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Potsdam qualitative cases.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    import tifffile
    from PIL import Image

    records = [json.loads(line) for line in (run_root / "records.jsonl").open(encoding="utf-8")]
    image_ids = sorted({r["image_id"] for r in records})

    # GT maps
    gt_maps = {}
    for image_id in image_ids:
        arr = tifffile.imread(Path(args.label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    def load(method, image_id):
        with np.load(run_root / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
            return archive["label_map"].astype(np.int64)

    # fixed selection rule: rank by |{pixel: C2 wrong AND CTP == GT}| for GT classes in UNSUPPORTED_POOL
    scored = []
    for image_id in image_ids:
        c2 = load("C2", image_id)
        ctp = load("CTP", image_id)
        gt = gt_maps[image_id]
        recovery = 0
        for name in UNSUPPORTED_POOL:
            ci = CLASSES.index(name)
            mask = (gt == ci) & (c2 != ci) & (ctp == ci)
            recovery += int(mask.sum())
        scored.append((recovery, image_id))
    scored.sort(reverse=True)
    top = scored[: args.top_k]
    print("selected (recovery_pixels, image_id):")
    for recovery, image_id in top:
        print(f"  {recovery:8d} {image_id}")
    with (out_dir / "selection_record.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "selection_rule": "rank by pixel count where C2 wrong AND CTP == GT for GT classes in {car, low_vegetation, tree}; take top-k; fixed before viewing",
            "unsupported_pool": UNSUPPORTED_POOL,
            "top_k": args.top_k,
            "selected": [{"image_id": image_id, "recovery_pixels": int(recovery)} for recovery, image_id in top],
        }, handle, indent=2, sort_keys=True)

    # render panels
    methods = ["text_only", "C2", "SCC", "CTP"]
    for rank, (recovery, image_id) in enumerate(top, start=1):
        arr = tifffile.imread(Path(args.image_dir) / f"{image_id}_RGB.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        if rgb.dtype != np.uint8:
            lo = float(np.percentile(rgb, 1)); hi = float(np.percentile(rgb, 99))
            rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        gt = gt_maps[image_id]
        panels = {"rgb": rgb, "gt": _colorize(gt)}
        for method in methods:
            panels[method] = _colorize(load(method, image_id))
        Image.fromarray(_hstack(panels), "RGB").save(out_dir / f"case{rank}_{image_id}.png")
    print("wrote panels to", out_dir)


def _colorize(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for ci, (name, color) in enumerate(CLASS_COLORS.items()):
        out[label_map == ci] = np.asarray(color, dtype=np.uint8)
    out[label_map == IGNORE] = (0, 0, 0)
    return out


def _hstack(panels: dict[str, np.ndarray]) -> np.ndarray:
    order = ["rgb", "gt", "text_only", "C2", "SCC", "CTP"]
    imgs = [panels[k] for k in order]
    height = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != height:
            ratio = height / h
            im = np.asarray(Image.fromarray(im).resize((int(w * ratio), height)), dtype=np.uint8)
        resized.append(im)
    return np.concatenate(resized, axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
