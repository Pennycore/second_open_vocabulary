"""Potsdam external held-out pixel OVSS for frozen CTP-v1 (protocol v0).

Phases: predict (GT isolated) -> hash manifest -> evaluate (GT unlocked).
Methods: Text-only / C2 normalized / SCC / CTP (all frozen formulas).

Usage:
    python scripts/run_potsdam_ctp_v1.py --config <yaml> --phase predict
    python scripts/run_potsdam_ctp_v1.py --config <yaml> --phase evaluate
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Potsdam pixel OVSS for CTP-v1.")
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
    label_dir = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir).resolve() if label_dir else None
    image_dir = Path(cfg["paths"]["image_dir"]).resolve()
    records_jsonl = Path(cfg["paths"]["records_jsonl"]).resolve()

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        run_root.mkdir(parents=True, exist_ok=False)
        import tifffile

        # Build per-image records for all test patches (image id -> candidate index list)
        records = []
        for npz_path in sorted(candidates_dir.glob("*.npz")):
            image_id = npz_path.name[:-4]
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            if not regions:
                # no proposals: nothing to predict for this patch; skip (protocol: uncovered=ignore)
                continue
            records.append({"image_id": image_id, "image_shape": list(shape), "candidate_count": len(regions)})
        with (run_root / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        print("patches with proposals:", len(records))

        # Encode all regions with frozen OpenAI CLIP (same checkpoint/protocol as Vaihingen)
        from ov_probe.vaihingen_blind import _encode_regions
        checkpoint = Path(cfg["paths"]["openai_clip_checkpoint"])
        import torch
        import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
        model.eval().to(device)

        # build text prototypes
        text_protos, token_hash = None, None
        from ov_probe.vaihingen_blind import _text_prototypes
        text_protos, token_hash = _text_prototypes(protocol, checkpoint, device)

        # encode per image; collect all regions
        all_features = []
        region_rows = []
        row_index = 0
        for record in records:
            image_id = record["image_id"]
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            image_path = image_dir / f"{image_id}_RGB.tif"
            arr = tifffile.imread(image_path)
            rgb = arr[:, :, :3] if arr.ndim == 3 else arr
            if rgb.dtype != np.uint8:
                lo = float(np.percentile(rgb, 1)); hi = float(np.percentile(rgb, 99))
                rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            features = _encode_regions(rgb, regions, model, preprocess, device)
            all_features.append(features)
            for index, region in enumerate(regions):
                region_rows.append({
                    "row_index": row_index,
                    "image_id": image_id,
                    "candidate_index": index,
                    "sam3_source_label": str(region["class_name"]),
                    "x0": region["x0"], "y0": region["y0"],
                })
                row_index += 1
        features_all = np.concatenate(all_features, axis=0).astype(np.float32)
        # visual prototypes: ALL-POSITIVE assumption means all classes supported -> build from ALL regions' SAM3 labels
        # (test-tile prototypes from SAM3 weak labels; consistent with protocol note)
        visual_protos = np.empty((len(CLASSES), 512), dtype=np.float32)
        counts = {}
        for ci, name in enumerate(CLASSES):
            positions = [r["row_index"] for r in region_rows if r["sam3_source_label"] == name]
            if not positions:
                raise InputValidationError(f"No SAM3 candidates for class {name}.")
            rows = features_all[positions]
            counts[name] = len(positions)
            visual_protos[ci] = rows.mean(axis=0)
            norm = np.linalg.norm(visual_protos[ci])
            visual_protos[ci] = visual_protos[ci] / norm

        regions_norm = features_all / np.linalg.norm(features_all, axis=1, keepdims=True)
        text_scores = regions_norm @ (text_protos / np.linalg.norm(text_protos, axis=1, keepdims=True)).T
        visual_scores = regions_norm @ (visual_protos / np.linalg.norm(visual_protos, axis=1, keepdims=True)).T
        normalizers = np.linalg.norm(0.5 * text_protos + 0.5 * visual_protos, axis=1)
        anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
        text_pred = np.argmax(text_scores, axis=1).astype(np.int64)

        mask_full = np.ones(len(CLASSES), dtype=bool)
        score_mats = method_score_matrices(text_scores, visual_scores, anchored, mask_full, text_pred)
        preds = method_predictions(score_mats, text_pred, mask_full)

        # per-image semantic maps
        from collections import OrderedDict
        by_image: "OrderedDict[str, list[dict]]" = OrderedDict()
        for r in region_rows:
            by_image.setdefault(r["image_id"], []).append(r)
        pos_by_row = {r["row_index"]: i for i, r in enumerate(region_rows)}

        maps = {}
        for image_id, image_records in by_image.items():
            shape, regions = load_candidate_masks(candidates_dir, image_id)
            region_by_index = {int(r["candidate_index"]): r for r in image_records}
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
            image_maps = {}
            for method in METHODS:
                pred_array = preds[method]
                scores_matrix = score_mats[method]
                pred_sel = np.asarray([pred_array[o["pos"]] for o in ordered], dtype=np.int64)
                score_sel = np.asarray([float(scores_matrix[o["pos"], pred_array[o["pos"]]]) for o in ordered], dtype=np.float32)
                label_map, _ = assemble_semantic_map(tuple(shape), ordered, pred_sel, score_sel, CLASSES)
                image_maps[method] = {"label_map": label_map}
            maps[image_id] = image_maps

        for image_id, image_maps in maps.items():
            for method in METHODS:
                path = run_root / f"{method}_{image_id}_semantic.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, label_map=image_maps[method]["label_map"])

        out = run_root / "predictions.npz"
        with out.open("xb") as handle:
            np.savez_compressed(handle,
                features=features_all.astype(np.float16),
                text_scores=text_scores.astype(np.float16),
                visual_scores=visual_scores.astype(np.float16),
                anchored_scores=anchored.astype(np.float16),
                text_prototypes=text_protos.astype(np.float16),
                visual_prototypes=visual_protos.astype(np.float16),
                text_pred=text_pred,
            )
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
            "test_patches": len(records),
            "methods": METHODS,
            "support": {"k": len(CLASSES), "classes": CLASSES, "mask_full": True},
            "fusion": {"canvas": "FusionCanvas", "conflict_margin": 0.03, "uncovered_label": 255, "ignore_index": 255},
            "text_token_sha256": token_hash,
            "visual_prototype_counts": counts,
            "records_jsonl_sha256": sha256_file(records_jsonl) if records_jsonl.exists() else None,
            "predictions_npz_sha256": sha256_file(out),
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
    pred_path = run_root / "predictions.npz"
    if predict_manifest["predictions_npz_sha256"] != sha256_file(pred_path):
        raise InputValidationError("predictions.npz changed since predict.")

    import tifffile
    from collections import OrderedDict
    records = [json.loads(line) for line in (run_root / "records.jsonl").open(encoding="utf-8")]
    # Evaluate only images with actual candidate proposals (semantic maps exist).
    test_images = sorted({str(r["image_id"]) for r in records
                          if (run_root / f"text_only_{r['image_id']}_semantic.npz").is_file()})
    print("evaluating images with proposals:", len(test_images))
    gt_maps = {}
    GT_COLOR_MAP = {
        "impervious_surface": (255, 255, 255),
        "building": (0, 0, 255),
        "low_vegetation": (0, 255, 255),
        "tree": (0, 255, 0),
        "car": (255, 255, 0),
    }
    for image_id in test_images:
        arr = tifffile.imread(Path(label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        for ci, (name, color) in enumerate(GT_COLOR_MAP.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt

    results = {m: [] for m in METHODS}
    for image_id in test_images:
        for method in METHODS:
            with np.load(run_root / f"{method}_{image_id}_semantic.npz", allow_pickle=False) as archive:
                pred_map = archive["label_map"].astype(np.int64)
            results[method].append(pixel_confusion(pred_map, gt_maps[image_id].astype(np.int64), CLASSES))

    overall = {}
    for method in METHODS:
        oa_num = sum(m["OA"] * m["valid_pixels"] for m in results[method])
        oa_den = sum(m["valid_pixels"] for m in results[method])
        total_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        for m in results[method]:
            total_matrix += np.asarray(m["confusion_matrix"], dtype=np.int64)
        per_iou, per_f1 = {}, {}
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
            "per_class_precision": {name: (float(total_matrix[i, i]) / float(total_matrix[:, i].sum()) if total_matrix[:, i].sum() > 0 else 0.0) for i, name in enumerate(CLASSES)},
            "per_class_recall": {name: (float(total_matrix[i, i]) / float(total_matrix[i, :].sum()) if total_matrix[i, :].sum() > 0 else 0.0) for i, name in enumerate(CLASSES)},
            "confusion_matrix": total_matrix.tolist(),
            "valid_pixels": int(oa_den),
        }

    with (run_root / "pixel_overall.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)
    print(json.dumps({m: {k: v for k, v in overall[m].items() if k != "confusion_matrix"} for m in METHODS}, indent=2, sort_keys=True))
    print("\nwrote pixel_overall.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
