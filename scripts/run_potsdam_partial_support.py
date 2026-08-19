"""Potsdam pixel partial-support for CTP-v1 (protocol v0).

Support subsets pre-registered: ratios 25/50/75% of the 5 classes, seeds 42/43/44
(deterministic `default_rng(seed + int(ratio*100)*1000)`). Saved before GT read.

Usage:
    python scripts/run_potsdam_partial_support.py --config <yaml> --phase manifest|predict|evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError, sha256_file, write_json  # noqa: E402
from ov_probe.pixel_ovss import (  # noqa: E402
    IGNORE_INDEX,
    assemble_semantic_map,
    load_candidate_masks,
    method_predictions,
    method_score_matrices,
    pixel_confusion,
)
from ov_probe.vaihingen_blind import CLASSES  # noqa: E402

METHODS = ["text_only", "C2", "SCC", "CTP"]
RATIOS = [0.25, 0.5, 0.75]
SEEDS = [42, 43, 44]


def generate_subsets(classes: list[str], ratios: list[float], seeds: list[int]) -> dict[str, dict]:
    manifest = {}
    for ratio in ratios:
        for seed in seeds:
            k = max(1, min(len(classes) - 1, int(round(len(classes) * ratio))))
            rng = np.random.default_rng(seed + int(ratio * 100) * 1000)
            supported = sorted(rng.choice(len(classes), size=k, replace=False).tolist())
            supported_names = [classes[i] for i in supported]
            key = f"r{int(ratio*100)}_seed{seed}"
            manifest[key] = {
                "ratio": ratio,
                "k": k,
                "seed": seed,
                "supported": supported_names,
                "unsupported": [c for c in classes if c not in supported_names],
            }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Potsdam pixel partial-support.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=["manifest", "predict", "evaluate"])
    args = parser.parse_args()

    import yaml
    project_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Config must set experiment.overwrite=false.")
    protocol = json.loads(Path(cfg["paths"]["protocol_file"]).read_text(encoding="utf-8"))
    protocol["sha256"] = hashlib.sha256(Path(cfg["paths"]["protocol_file"]).read_bytes()).hexdigest()

    run_root = Path(cfg["paths"]["output_root"]).resolve()
    candidates_dir = Path(cfg["paths"]["candidates_dir"]).resolve()
    label_dir = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir).resolve() if label_dir else None

    subsets = generate_subsets(CLASSES, RATIOS, SEEDS)

    if args.phase == "manifest":
        run_root.mkdir(parents=True, exist_ok=False)
        with (run_root / "support_subset_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(subsets, handle, indent=2, sort_keys=True)
        print(json.dumps(subsets, indent=2, sort_keys=True))
        return 0

    manifest_path = run_root / "support_subset_manifest.json"
    if not manifest_path.is_file():
        raise InputValidationError("support_subset_manifest.json must be created first (--phase manifest).")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved != subsets:
        raise InputValidationError("Support subset manifest does not match deterministic generation.")

    # load frozen predictions from the full-support run
    pred_npz = Path(cfg["paths"]["predictions_npz"])
    if not pred_npz.is_file():
        raise InputValidationError("predictions.npz missing; run full-support predict first.")
    d = np.load(pred_npz, allow_pickle=False)
    text_scores_all = d["text_scores"].astype(np.float32)
    visual_scores_all = d["visual_scores"].astype(np.float32)
    anchored_all = d["anchored_scores"].astype(np.float32)
    text_pred_all = d["text_pred"].astype(np.int64)

    records = [json.loads(line) for line in Path(cfg["paths"]["records_jsonl"]).open(encoding="utf-8")]
    pos_by_row = {r["row_index"]: i for i, r in enumerate(records)}
    from collections import OrderedDict
    by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        # load regions per image once
        image_regions = {}
        for image_id in by_image:
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            region_by_index = {int(r["candidate_index"]): r for r in by_image[image_id]}
            ordered = []
            for index in range(len(regions)):
                record = region_by_index[index]
                pos = pos_by_row[record["row_index"]]
                ordered.append({
                    "mask": regions[index]["mask"],
                    "x0": regions[index]["x0"],
                    "y0": regions[index]["y0"],
                    "row_index": record["row_index"],
                    "pos": pos,
                })
            image_regions[image_id] = (shape, ordered)

        for key, info in subsets.items():
            mask = np.asarray([c in info["supported"] for c in CLASSES], dtype=bool)
            score_mats = method_score_matrices(text_scores_all, visual_scores_all, anchored_all, mask, text_pred_all)
            preds = method_predictions(score_mats, text_pred_all, mask)
            for image_id, (shape, ordered) in image_regions.items():
                for method in METHODS:
                    pred_array = preds[method]
                    scores_matrix = score_mats[method]
                    pred_sel = np.asarray([pred_array[o["pos"]] for o in ordered], dtype=np.int64)
                    score_sel = np.asarray([float(scores_matrix[o["pos"], pred_array[o["pos"]]]) for o in ordered], dtype=np.float32)
                    label_map, _ = assemble_semantic_map(shape, ordered, pred_sel, score_sel, CLASSES)
                    path = run_root / f"{key}_{method}_{image_id}_semantic.npz"
                    with path.open("xb") as handle:
                        np.savez_compressed(handle, label_map=label_map)
        artifacts = {}
        for path in sorted(run_root.glob("*_semantic.npz")):
            artifacts[path.name] = sha256_file(path)
        manifest = {
            "format_version": 1,
            "phase": "predict",
            "status": "completed",
            "scientific_evidence": True,
            "protocol": {"sha256": protocol["sha256"]},
            "dataset": "Potsdam",
            "methods": METHODS,
            "support_subsets": subsets,
            "support_subset_manifest_sha256": sha256_file(manifest_path),
            "fusion": {"canvas": "FusionCanvas", "conflict_margin": 0.03, "uncovered_label": 255, "ignore_index": 255},
            "predictions_npz_sha256": sha256_file(pred_npz),
            "artifacts": artifacts,
        }
        write_json(run_root / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    # ---------------- evaluate ----------------
    if label_dir is None or not label_dir.is_dir():
        raise InputValidationError("Evaluate phase requires the GT label directory.")
    predict_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if predict_manifest.get("phase") != "predict" or predict_manifest.get("status") != "completed":
        raise InputValidationError("Predict manifest missing or incomplete.")
    for name, expected in predict_manifest["artifacts"].items():
        path = run_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise InputValidationError(f"Semantic artifact changed since predict: {name}")

    import tifffile
    GT_COLOR_MAP = {
        "impervious_surface": (255, 255, 255),
        "building": (0, 0, 255),
        "low_vegetation": (0, 255, 255),
        "tree": (0, 255, 0),
        "car": (255, 255, 0),
    }
    test_images = sorted(by_image)
    gt_maps = {}
    for image_id in test_images:
        arr = tifffile.imread(Path(label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    def aggregate(per_image):
        oa_num = sum(m["OA"] * m["valid_pixels"] for m in per_image)
        oa_den = sum(m["valid_pixels"] for m in per_image)
        total_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        for m in per_image:
            total_matrix += np.asarray(m["confusion_matrix"], dtype=np.int64)
        per_iou, per_f1 = {}, {}
        for i, name in enumerate(CLASSES):
            tp = float(total_matrix[i, i]); fp = float(total_matrix[:, i].sum() - tp); fn = float(total_matrix[i, :].sum() - tp)
            per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {
            "OA": oa_num / oa_den if oa_den else 0.0,
            "macro_f1": float(np.mean(list(per_f1.values()))),
            "mIoU": float(np.mean(list(per_iou.values()))),
            "per_class_iou": per_iou,
            "per_class_f1": per_f1,
            "confusion_matrix": total_matrix.tolist(),
            "valid_pixels": int(oa_den),
        }

    results = {}
    for key, info in subsets.items():
        supported, unsupported = info["supported"], info["unsupported"]
        per_image = {m: [] for m in METHODS}
        for image_id in test_images:
            for method in METHODS:
                with np.load(run_root / f"{key}_{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                    pred_map = archive["label_map"].astype(np.int64)
                per_image[method].append(pixel_confusion(pred_map, gt_maps[image_id].astype(np.int64), CLASSES))
        subset_out = {}
        for method in METHODS:
            agg = aggregate(per_image[method])
            s_iou = float(np.mean([agg["per_class_iou"][c] for c in supported]))
            u_iou = float(np.mean([agg["per_class_iou"][c] for c in unsupported]))
            s_f1 = float(np.mean([agg["per_class_f1"][c] for c in supported]))
            u_f1 = float(np.mean([agg["per_class_f1"][c] for c in unsupported]))
            h_iou = 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou > 0 else 0.0
            h_f1 = 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 > 0 else 0.0
            subset_out[method] = {
                "OA": agg["OA"], "macro_f1": agg["macro_f1"], "mIoU": agg["mIoU"],
                "S_IoU": s_iou, "U_IoU": u_iou, "H_IoU": h_iou,
                "S_F1": s_f1, "U_F1": u_f1, "H_F1": h_f1,
                "valid_pixels": agg["valid_pixels"],
            }
        results[key] = subset_out

    with (run_root / "pixel_partial_support_results.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))
    print("\nwrote pixel_partial_support_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
