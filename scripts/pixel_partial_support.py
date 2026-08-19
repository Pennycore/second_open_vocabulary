"""Experiment B: Vaihingen pixel-level partial-support evaluation (k=2/3/4).

Support subsets are pre-registered by seeds 42/43/44 (random class subsets,
saved before any GT read). Methods compared: Text-only / C2 / SCC / CTP with
frozen formulas, FusionCanvas, GT isolation.

Usage:
    python scripts/pixel_partial_support.py --config <yaml> --phase manifest
    python scripts/pixel_partial_support.py --config <yaml> --phase predict
    python scripts/pixel_partial_support.py --config <yaml> --phase evaluate
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
from ov_probe.vaihingen_blind import CLASSES, GT_COLOR_MAP, TEST_AREAS  # noqa: E402

METHODS = ["text_only", "C2", "SCC", "CTP"]
SEEDS = [42, 43, 44]
KS = [2, 3, 4]


def generate_support_subsets(classes: list[str], seeds: list[int], ks: list[int]) -> dict[str, dict]:
    """Deterministic pre-registered support subsets (seeded random choice)."""
    manifest: dict[str, dict] = {}
    for k in ks:
        for seed in seeds:
            rng = np.random.default_rng(seed + k * 100)
            supported = sorted(rng.choice(len(classes), size=k, replace=False).tolist())
            supported_names = [classes[i] for i in supported]
            key = f"k{k}_seed{seed}"
            manifest[key] = {
                "k": k,
                "seed": seed,
                "supported": supported_names,
                "unsupported": [c for c in classes if c not in supported_names],
            }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Vaihingen pixel partial-support.")
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
    region_scores_npz = Path(cfg["paths"]["region_scores_npz"]).resolve()
    label_dir = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir).resolve() if label_dir else None

    subset_manifest = generate_support_subsets(CLASSES, SEEDS, KS)

    if args.phase == "manifest":
        run_root.mkdir(parents=True, exist_ok=False)
        with (run_root / "support_subset_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(subset_manifest, handle, indent=2, sort_keys=True)
        print(json.dumps(subset_manifest, indent=2, sort_keys=True))
        return 0

    # manifest must exist and match the deterministic generation (pre-GT)
    manifest_path = run_root / "support_subset_manifest.json"
    if not manifest_path.is_file():
        raise InputValidationError("support_subset_manifest.json must be created first (--phase manifest).")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved != subset_manifest:
        raise InputValidationError("Support subset manifest does not match the deterministic generation.")

    d = np.load(region_scores_npz, allow_pickle=False)
    text_scores_all = d["text_scores"].astype(np.float32)
    visual_scores_all = d["visual_scores"].astype(np.float32)
    anchored_all = d["anchored_scores"].astype(np.float32)
    text_pred_all = d["text_pred"].astype(np.int64)
    records = [json.loads(line) for line in Path(cfg["paths"]["records_jsonl"]).open(encoding="utf-8")]
    test_records = [r for r in records if r["split"] == "test"]
    row_to_pos = {int(r["row_index"]): i for i, r in enumerate(records)}
    from collections import OrderedDict
    test_by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in test_records:
        test_by_image.setdefault(str(r["image_id"]), []).append(r)

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        run_root.mkdir(parents=True, exist_ok=True)  # manifest phase already created it
        # Precompute per-subset method score matrices and predictions
        subset_preds: dict[str, dict[str, np.ndarray]] = {}
        for key, info in subset_manifest.items():
            mask = np.asarray([c in info["supported"] for c in CLASSES], dtype=bool)
            score_mats = method_score_matrices(text_scores_all, visual_scores_all, anchored_all, mask, text_pred_all)
            subset_preds[key] = method_predictions(score_mats, text_pred_all, mask)

        # load regions once per image
        image_regions: dict[str, tuple[tuple[int, int], list[dict]]] = {}
        for image_id in test_by_image:
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            region_by_index = {int(r["candidate_index"]): r for r in test_by_image[image_id]}
            ordered = []
            for index in range(len(regions)):
                record = region_by_index[index]
                pos = row_to_pos[int(record["row_index"])]
                ordered.append({
                    "mask": regions[index]["mask"],
                    "x0": regions[index]["x0"],
                    "y0": regions[index]["y0"],
                    "row_index": int(record["row_index"]),
                    "pos": pos,
                })
            image_regions[image_id] = (shape, ordered)

        for key in subset_manifest:
            preds = subset_preds[key]
            for method in METHODS:
                pred_array = preds[method]
                # scores matrix for the method (need for fusion)
                mask = np.asarray([c in subset_manifest[key]["supported"] for c in CLASSES], dtype=bool)
                score_mats = method_score_matrices(text_scores_all, visual_scores_all, anchored_all, mask, text_pred_all)
                scores_matrix = score_mats[method]
                for image_id, (shape, ordered) in image_regions.items():
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
            "dataset": "Vaihingen",
            "test_areas": TEST_AREAS,
            "methods": METHODS,
            "support_subsets": {key: {"k": info["k"], "seed": info["seed"], "supported": info["supported"], "unsupported": info["unsupported"]} for key, info in subset_manifest.items()},
            "support_subset_manifest_sha256": sha256_file(manifest_path),
            "fusion": {"canvas": "FusionCanvas", "conflict_margin": 0.03, "uncovered_label": 255, "ignore_index": 255},
            "region_scores_npz_sha256": sha256_file(region_scores_npz),
            "records_jsonl_sha256": sha256_file(Path(cfg["paths"]["records_jsonl"])),
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
    test_images = sorted(test_by_image)
    gt_maps = {}
    for image_id in test_images:
        arr = tifffile.imread(Path(label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    def pixel_metrics_aggregate(per_image: list[dict]) -> dict:
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

    results: dict[str, dict[str, dict]] = {}
    for key, info in subset_manifest.items():
        supported, unsupported = info["supported"], info["unsupported"]
        per_image = {m: [] for m in METHODS}
        for image_id in test_images:
            for method in METHODS:
                with np.load(run_root / f"{key}_{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                    pred_map = archive["label_map"].astype(np.int64)
                per_image[method].append(pixel_confusion(pred_map, gt_maps[image_id].astype(np.int64), CLASSES))
        subset_out = {}
        for method in METHODS:
            agg = pixel_metrics_aggregate(per_image[method])
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
