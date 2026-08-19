"""Runner: pixel-level OVSS on LoveDA (protocol v0).

Uses the frozen 6000-region pixel pack shards (crop_box + crop_mask in full-image
coordinates) as the proposal source; semantic assignment via frozen OpenAI CLIP
scores (predictions.npz); FusionCanvas fusion; GT isolated until hashes persist.
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
    method_predictions,
    method_score_matrices,
    pixel_confusion,
)
from ov_probe.loveda_blind_gt import _CLASSES, _load_record_masks  # noqa: E402

METHODS = ["text_only", "C2", "SCC", "CTP"]
GT_COLOR_MAP = {
    "building": (255, 0, 0),
    "road": (255, 255, 0),
    "water": (0, 0, 255),
    "barren": (159, 129, 183),
    "forest": (0, 255, 0),
    "agriculture": (255, 255, 255),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pixel-level OVSS on LoveDA.")
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
    pixel_pack = Path(cfg["paths"]["pixel_pack"]).resolve()
    region_scores_npz = Path(cfg["paths"]["region_scores_npz"]).resolve()
    label_dir = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir).resolve() if label_dir else None

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        run_root.mkdir(parents=True, exist_ok=False)
        d = np.load(region_scores_npz, allow_pickle=False)
        text_scores_all = d["text_scores"].astype(np.float32)
        visual_scores_all = d["visual_scores"].astype(np.float32)
        text_protos = d["text_prototypes"].astype(np.float32)
        visual_protos = d["visual_prototypes"].astype(np.float32)
        text_pred_all = d["text_only"].astype(np.int64)
        hold_indices = d["heldout_row_indices"].astype(np.int64)
        normalizers = np.linalg.norm(0.5 * text_protos + 0.5 * visual_protos, axis=1)
        anchored_all = (0.5 * text_scores_all + 0.5 * visual_scores_all) / normalizers[None, :]

        records = [json.loads(line) for line in Path(cfg["paths"]["heldout_keys_jsonl"]).open(encoding="utf-8")]
        hold_positions = {int(row["row_index"]): position for position, row in enumerate(records)}
        mask_info = _load_record_masks(pixel_pack)  # row_index -> crop_box/mask (crop coords)

        # per-image test regions (heldout 411 images)
        from collections import OrderedDict
        heldout_by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
        for record in records:
            heldout_by_image.setdefault(str(record["image_id"]), []).append(record)

        mask_full = np.ones(len(_CLASSES), dtype=bool)
        score_mats = method_score_matrices(text_scores_all, visual_scores_all, anchored_all, mask_full, text_pred_all)
        preds = method_predictions(score_mats, text_pred_all, mask_full)

        maps: dict[str, dict[str, dict]] = {}
        for image_id, image_records in heldout_by_image.items():
            ordered = []
            for record in image_records:
                row_index = int(record["row_index"])
                info = mask_info[row_index]
                # crop mask is in crop coordinates; place at crop_box origin (full image)
                crop_box = info["crop_box"]  # (x1, y1, x2, y2)
                mask = info["mask"]
                ordered.append({
                    "mask": mask,
                    "x0": crop_box[0],
                    "y0": crop_box[1],
                    "row_index": row_index,
                    "pos": hold_positions[row_index],
                })
            image_maps = {}
            for method in METHODS:
                pred_array = preds[method]
                scores_matrix = score_mats[method]
                pred_sel = np.asarray([pred_array[o["pos"]] for o in ordered], dtype=np.int64)
                score_sel = np.asarray([float(scores_matrix[o["pos"], pred_array[o["pos"]]]) for o in ordered], dtype=np.float32)
                # image shape is 1024x1024 per protocol (LoveDA native)
                label_map, stats = assemble_semantic_map((1024, 1024), ordered, pred_sel, score_sel, _CLASSES)
                image_maps[method] = {"label_map": label_map, "stats": stats}
            maps[image_id] = image_maps

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
            "dataset": "LoveDA",
            "methods": METHODS,
            "support": {"k": len(_CLASSES), "classes": _CLASSES, "mask_full": True},
            "fusion": {"canvas": "FusionCanvas", "conflict_margin": 0.03, "uncovered_label": 255, "ignore_index": 255},
            "region_scores_npz_sha256": sha256_file(region_scores_npz),
            "heldout_keys_jsonl_sha256": sha256_file(Path(cfg["paths"]["heldout_keys_jsonl"])),
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
    for name, expected in predict_manifest["artifacts"].items():
        path = run_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise InputValidationError(f"Semantic artifact changed since predict: {name}")
    if sha256_file(run_root / "pixel_stats.json") != predict_manifest["pixel_stats_json_sha256"]:
        raise InputValidationError("Pixel stats changed since predict.")

    from PIL import Image
    records = [json.loads(line) for line in Path(cfg["paths"]["heldout_keys_jsonl"]).open(encoding="utf-8")]
    test_images = sorted({str(r["image_id"]) for r in records})
    gt_maps: dict[str, np.ndarray] = {}
    for image_id in test_images:
        with Image.open(Path(label_dir) / f"{image_id}_label.png") as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    results: dict[str, list[dict]] = {m: [] for m in METHODS}
    for image_id in test_images:
        for method in METHODS:
            with np.load(run_root / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                pred_map = archive["label_map"].astype(np.int64)
            metrics = pixel_confusion(pred_map, gt_maps[image_id].astype(np.int64), _CLASSES)
            results[method].append(metrics)

    overall = {}
    for method in METHODS:
        oa_num = sum(m["OA"] * m["valid_pixels"] for m in results[method])
        oa_den = sum(m["valid_pixels"] for m in results[method])
        total_matrix = np.zeros((len(_CLASSES), len(_CLASSES)), dtype=np.int64)
        for m in results[method]:
            total_matrix += np.asarray(m["confusion_matrix"], dtype=np.int64)
        per_iou, per_f1 = {}, {}
        for i, name in enumerate(_CLASSES):
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
