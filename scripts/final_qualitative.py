"""Final Phase D qualitative: Representative + Failure case panels (frozen CTP-v1).

Frozen inputs only (no re-inference): partial-support semantic maps from
potsdam_ctp_v1_partial_v0 (or vaihingen_pixel_partial_support_v0), guard maps
from the final audit guard_maps phase, official GT for visualization only.

Groups (selection rules fixed BEFORE rendering, no cherry-picking):
  representative : rng(42) draw without replacement from test patches whose GT
                   contains >= 1% unsupported-class pixels (informative filter),
                   take min(n, available).
  failure        : rank by |{pixel: CTP != GT AND guard == GT}| (CTP fails while
                   hard text-preservation recovers the pixel), take top n.

Panels (7): RGB | GT | text_only | C2 | SCC | CTP | guard.

Usage:
    python scripts/final_qualitative.py --run-dir outputs/potsdam_ctp_v1_partial_v0 \
        --label-dir /home/undergr/remote_dataset/Potsdam_test_v1/labels \
        --image-dir /home/undergr/remote_dataset/Potsdam_test_v1/images \
        --out-dir outputs/final_audit/qualitative \
        --subset-key r50_seed42 --group representative --n 4
    python scripts/final_qualitative.py ... --group failure --n 3
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
METHODS = ["text_only", "C2", "SCC", "CTP", "guard"]
PANEL_ORDER = ["rgb", "gt", "text_only", "C2", "SCC", "CTP", "guard"]


def load(run_root: Path, key: str, method: str, image_id: str) -> np.ndarray:
    path = run_root / f"{key}_{method}_{image_id}_semantic.npz"
    if not path.is_file():
        raise InputValidationError(f"Missing semantic map: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return archive["label_map"].astype(np.int64)


def _colorize(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for ci, (name, color) in enumerate(CLASS_COLORS.items()):
        out[label_map == ci] = np.asarray(color, dtype=np.uint8)
    out[label_map == IGNORE] = (0, 0, 0)
    return out


def _hstack(panels: dict[str, np.ndarray]) -> np.ndarray:
    imgs = [panels[k] for k in PANEL_ORDER]
    height = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != height:
            ratio = height / h
            im = np.asarray(Image.fromarray(im).resize((int(w * ratio), height)), dtype=np.uint8)
        resized.append(im)
    return np.concatenate(resized, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Final qualitative panels (Representative / Failure).")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subset-key", required=True, help="partial-support subset, e.g. r50_seed42")
    parser.add_argument("--records-jsonl", default=None,
                        help="records file (partial run roots reference the full-support records.jsonl)")
    parser.add_argument("--group", required=True, choices=["representative", "failure"])
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    import tifffile
    from PIL import Image

    run_root = Path(args.run_dir)
    records_path = Path(args.records_jsonl) if args.records_jsonl else run_root / "records.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in records_path.open(encoding="utf-8")]
    image_ids = sorted({r["image_id"] for r in records})

    gt_maps = {}
    for image_id in image_ids:
        arr = tifffile.imread(Path(args.label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    subset_manifest_path = run_root / "support_subset_manifest.json"
    if not subset_manifest_path.is_file():
        raise InputValidationError("support_subset_manifest.json missing in run dir.")
    manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    if args.subset_key not in manifest:
        raise InputValidationError(f"subset {args.subset_key} not in support_subset_manifest.json")
    unsupported = manifest[args.subset_key]["unsupported"]
    unsupported_idx = [CLASSES.index(name) for name in unsupported]

    selected: list[dict] = []
    if args.group == "representative":
        # fixed informative filter: >=1% GT pixels belong to unsupported classes
        min_unsup_frac = 0.01
        pool = []
        for image_id in image_ids:
            gt = gt_maps[image_id]
            unsup_count = int(sum((gt == ci).sum() for ci in unsupported_idx))
            total = int((gt != IGNORE).sum())
            if total and unsup_count / total >= min_unsup_frac:
                pool.append(image_id)
        rng = np.random.default_rng(42)
        chosen = rng.choice(pool, size=min(args.n, len(pool)), replace=False)
        for image_id in chosen:
            selected.append({"image_id": str(image_id), "group": "representative"})
        record = {
            "group": "representative",
            "selection_rule": (
                f"rng(seed=42) draw without replacement from test patches whose GT contains "
                f">= {min_unsup_frac:.0%} unsupported-class pixels (informative filter); take min(n={args.n}, pool size)"
            ),
            "subset_key": args.subset_key,
            "unsupported": unsupported,
            "pool_size": len(pool),
            "selected": selected,
        }
    else:  # failure
        scored = []
        for image_id in image_ids:
            ctp = load(run_root, args.subset_key, "CTP", image_id)
            guard = load(run_root, args.subset_key, "guard", image_id)
            gt = gt_maps[image_id]
            n_fail = int(((ctp != gt) & (guard == gt) & (gt != IGNORE)).sum())
            scored.append((n_fail, image_id))
        scored.sort(reverse=True)
        top = scored[: args.n]
        for n_fail, image_id in top:
            selected.append({"image_id": image_id, "group": "failure", "ctp_fail_guard_ok_pixels": int(n_fail)})
        record = {
            "group": "failure",
            "selection_rule": (
                "rank test patches by |{pixel: CTP != GT AND guard == GT AND GT != ignore}| "
                "(CTP fails while hard text-preservation recovers the pixel); take top n"
            ),
            "subset_key": args.subset_key,
            "unsupported": unsupported,
            "selected": selected,
        }

    with (out_dir / f"selection_record_{args.group}.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))

    for rank, entry in enumerate(selected, start=1):
        image_id = entry["image_id"]
        arr = tifffile.imread(Path(args.image_dir) / f"{image_id}_RGB.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        if rgb.dtype != np.uint8:
            lo = float(np.percentile(rgb, 1)); hi = float(np.percentile(rgb, 99))
            rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        panels = {"rgb": rgb, "gt": _colorize(gt_maps[image_id])}
        for method in METHODS:
            panels[method] = _colorize(load(run_root, args.subset_key, method, image_id))
        out_path = out_dir / f"{args.group}_{rank:02d}_{image_id}.png"
        Image.fromarray(_hstack(panels), "RGB").save(out_path)
        print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
