"""Runner: pixel-level OVSS on Vaihingen (protocol v0).

    python scripts/run_pixel_ovss_vaihingen.py --config <yaml> --phase predict|evaluate
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
from ov_probe.vaihingen_blind import CLASSES, GT_COLOR_MAP, TEST_AREAS, TRAIN_AREAS, _area_from_id  # noqa: E402

METHODS = ["text_only", "C2", "SCC", "CTP"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pixel-level OVSS on Vaihingen.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=["predict", "evaluate"])
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
    region_scores_npz = Path(cfg["paths"]["region_scores_npz"]).resolve()  # frozen region-level predictions.npz
    label_dir = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir).resolve() if label_dir else None

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        run_root.mkdir(parents=True, exist_ok=False)
        d = np.load(region_scores_npz, allow_pickle=False)
        text_scores_all = d["text_scores"].astype(np.float32)
        anchored_all = d["anchored_scores"].astype(np.float32)
        text_pred_all = d["text_pred"].astype(np.int64)
        test_positions = d["test_positions"].astype(np.int64)
        records = [json.loads(line) for line in Path(cfg["paths"]["records_jsonl"]).open(encoding="utf-8")]
        test_records = [r for r in records if r["split"] == "test"]
        # row_index -> position within all-region arrays
        row_to_pos = {int(r["row_index"]): i for i, r in enumerate(records)}

        # per-image test region lists (row_index order within image)
        from collections import OrderedDict
        test_by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
        for r in test_records:
            test_by_image.setdefault(str(r["image_id"]), []).append(r)

        # compute per-method predictions for ALL regions once (row_index order)
        # k=5 full support (all 5 classes supported) per protocol main experiment
        mask_full = np.ones(len(CLASSES), dtype=bool)
        score_mats = method_score_matrices(text_scores_all, None, anchored_all, mask_full, text_pred_all)
        preds = method_predictions(score_mats, text_pred_all, mask_full)

        maps: dict[str, dict[str, Any]] = {}
        for image_id, image_records in test_by_image.items():
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            # map region index -> row_index -> position
            region_by_index = {int(r["candidate_index"]): r for r in image_records}
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
            image_maps = {}
            for method in METHODS:
                pred_array = preds[method]
                scores_matrix = score_mats[method]
                pred_sel = np.asarray([pred_array[o["pos"]] for o in ordered], dtype=np.int64)
                score_sel = np.asarray([float(scores_matrix[o["pos"], pred_array[o["pos"]]]) for o in ordered], dtype=np.float32)
                label_map, stats = assemble_semantic_map(shape, ordered, pred_sel, score_sel, CLASSES)
                image_maps[method] = {"label_map": label_map, "stats": stats}
            maps[image_id] = image_maps

        # persist
        for image_id, image_maps in maps.items():
            for method in METHODS:
                entry = image_maps[method]
                path = run_root / f"{method}_{image_id}_semantic.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, label_map=entry["label_map"])
        with (run_root / "pixel_stats.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({img: {m: maps[img][m]["stats"] for m in METHODS} for img in maps}, handle, indent=2, sort_keys=True)

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
            "support": {"k": len(CLASSES), "classes": CLASSES, "mask_full": True},
            "fusion": {"canvas": "FusionCanvas", "conflict_margin": 0.03, "uncovered_label": 255, "ignore_index": 255},
            "region_scores_npz_sha256": sha256_file(region_scores_npz),
            "records_jsonl_sha256": sha256_file(Path(cfg["paths"]["records_jsonl"])),
            "artifacts": artifacts,
            "pixel_stats_json_sha256": sha256_file(run_root / "pixel_stats.json"),
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
    # verify all artifacts unchanged
    for name, expected in predict_manifest["artifacts"].items():
        path = run_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise InputValidationError(f"Semantic artifact changed since predict: {name}")
    if sha256_file(run_root / "pixel_stats.json") != predict_manifest["pixel_stats_json_sha256"]:
        raise InputValidationError("Pixel stats changed since predict.")

    import tifffile
    from PIL import Image
    # GT color maps
    test_images = sorted({str(r["image_id"]) for r in [json.loads(line) for line in Path(cfg["paths"]["records_jsonl"]).open(encoding="utf-8")] if r["split"] == "test"})
    gt_maps: dict[str, np.ndarray] = {}
    for image_id in test_images:
        arr = tifffile.imread(Path(label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        # clutter stays ignore
        gt_maps[image_id] = gt

    results: dict[str, list[dict]] = {m: [] for m in METHODS}
    for image_id in test_images:
        for method in METHODS:
            with np.load(run_root / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                pred_map = archive["label_map"].astype(np.int64)
            metrics = pixel_confusion(pred_map, gt_maps[image_id].astype(np.int64), CLASSES)
            results[method].append(metrics)

    # aggregate per method over images (weighted by valid pixels)
    overall = {}
    for method in METHODS:
        oa_num = sum(m["OA"] * m["valid_pixels"] for m in results[method])
        oa_den = sum(m["valid_pixels"] for m in results[method])
        per_iou = {c: 0.0 for c in CLASSES}
        per_f1 = {c: 0.0 for c in CLASSES}
        for c in CLASSES:
            tp = sum(m["confusion_matrix"][CLASSES.index(c)][CLASSES.index(c)] for m in results[method])
            fp = sum(sum(row) - row[CLASSES.index(c)] for row, m in zip([m["confusion_matrix"][CLASSES.index(c)] for m in results[method]], results[method])) if False else 0
            # simpler: aggregate confusion matrices
        # aggregate confusion matrix
        total_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        for m in results[method]:
            total_matrix += np.asarray(m["confusion_matrix"], dtype=np.int64)
        per_iou = {}
        per_f1 = {}
        for i, name in enumerate(CLASSES):
            tp = float(total_matrix[i, i]); fp = float(total_matrix[:, i].sum() - tp); fn = float(total_matrix[i, :].sum() - tp)
            per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        overall[method] = {
            "OA": oa_num / oa_den if oa_den else 0.0,
            "macro_f1": float(np.mean(list(per_f1.values()))),
            "mIoU": float(np.mean(list(per_iou.values()))),
            "per_class_iou": per_iou,
            "per_class_f1": per_f1,
            "confusion_matrix": total_matrix.tolist(),
            "valid_pixels": int(oa_den),
        }

    with (run_root / "pixel_overall.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)
    with (run_root / "pixel_per_image.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(overall, indent=2, sort_keys=True))
    print("\nwrote pixel_overall.json / pixel_per_image.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
