"""Final audit runner: Guard maps + common-pixel metrics + cluster bootstrap.

Usage:
    python scripts/run_final_audit.py --dataset vaihingen --phase guard_maps
    python scripts/run_final_audit.py --dataset potsdam  --phase guard_maps
    python scripts/run_final_audit.py --dataset vaihingen --phase five_method
    python scripts/run_final_audit.py --dataset potsdam  --phase five_method
    python scripts/run_final_audit.py --dataset vaihingen --phase common_pixel
    python scripts/run_final_audit.py --dataset potsdam  --phase common_pixel
    python scripts/run_final_audit.py --dataset vaihingen --phase cluster_bootstrap
    python scripts/run_final_audit.py --dataset potsdam  --phase cluster_bootstrap
    python scripts/run_final_audit.py --dataset loveda   --phase cluster_bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError, sha256_file, write_json  # noqa: E402
from ov_probe.final_audit import (  # noqa: E402
    METHODS,
    build_guard_semantic_maps,
    common_pixel_metrics,
    load_label_maps,
    metrics_from_matrix,
    per_image_confusion,
    per_method_valid_pixels,
)

IGNORE = 255


def _gt_maps(label_dir: Path, image_ids: list[str], color_map: dict[str, tuple]) -> dict[str, np.ndarray]:
    import tifffile
    gt_maps = {}
    for image_id in image_ids:
        arr = tifffile.imread(Path(label_dir) / f"{image_id}_label.tif")
        rgb = arr[:, :, :3] if arr.ndim == 3 else arr
        gt = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
        for ci, (name, color) in enumerate(color_map.items()):
            gt[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = ci
        gt_maps[image_id] = gt
    return gt_maps


def main() -> int:
    parser = argparse.ArgumentParser(description="Final audit runner.")
    parser.add_argument("--dataset", required=True, choices=["vaihingen", "potsdam", "loveda"])
    parser.add_argument("--phase", required=True, choices=["guard_maps", "five_method", "common_pixel", "cluster_bootstrap"])
    args = parser.parse_args()

    if args.dataset == "vaihingen":
        run_root = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/vaihingen_pixel_partial_support_v0")
        candidates_dir = Path("/home/undergr/Sheungzhen_project_1/sam3_remote_wsss/runs/vaihingen_sam3_v0/candidates")
        predictions_npz = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/vaihingen_blind_scc_v1/predictions.npz")
        records_jsonl = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/vaihingen_blind_scc_v1/records.jsonl")
        label_dir = Path("/home/undergr/remote_dataset/Vaihingen_main_v1/labels")
        subsets = json.loads((run_root / "support_subset_manifest.json").read_text())
        classes = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
        color_map = {"impervious_surface": (255, 255, 255), "building": (0, 0, 255),
                     "low_vegetation": (0, 255, 255), "tree": (0, 255, 0), "car": (255, 255, 0)}
        cluster_key = lambda image_id: image_id  # area == image_id for Vaihingen
    elif args.dataset == "potsdam":
        run_root = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/potsdam_ctp_v1_partial_v0")
        candidates_dir = Path("/home/undergr/Sheungzhen_project_1/sam3_remote_wsss/runs/potsdam_sam3_test_v1/candidates")
        predictions_npz = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/potsdam_ctp_v1_v0/predictions.npz")
        records_jsonl = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/potsdam_ctp_v1_v0/records.jsonl")
        label_dir = Path("/home/undergr/remote_dataset/Potsdam_test_v1/labels")
        subsets = json.loads((run_root / "support_subset_manifest.json").read_text())
        classes = ["impervious_surface", "building", "low_vegetation", "tree", "car"]
        color_map = {"impervious_surface": (255, 255, 255), "building": (0, 0, 255),
                     "low_vegetation": (0, 255, 255), "tree": (0, 255, 0), "car": (255, 255, 0)}
        cluster_key = lambda image_id: image_id.split("_x")[0]  # parent tile (top_potsdam_2_13)
    else:  # loveda
        run_root = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/loveda_blind_gt_v0")
        predictions_npz = run_root / "predictions.npz"
        records_jsonl = run_root / "heldout_keys.jsonl"
        label_dir = Path("/home/undergr/remote_dataset/LoveDA_main_v1/train/labels")
        classes = ["building", "road", "water", "barren", "forest", "agriculture"]
        color_map = {"building": (255, 0, 0), "road": (255, 255, 0), "water": (0, 0, 255),
                     "barren": (159, 129, 183), "forest": (0, 255, 0), "agriculture": (255, 255, 255)}
        subsets = None
        cluster_key = lambda image_id: image_id

    if args.phase == "guard_maps":
        only_images = None
        if args.dataset == "vaihingen":
            # records.jsonl covers all areas; partial-support maps exist for test areas only
            recs = [json.loads(line) for line in records_jsonl.open(encoding="utf-8")]
            only_images = {str(r["image_id"]) for r in recs if r.get("split") == "test"}
        counts = build_guard_semantic_maps(run_root, candidates_dir, predictions_npz, records_jsonl, subsets, classes, only_images)
        print(f"guard maps written: {len(counts)}")
        return 0

    # common images (from records)
    records = [json.loads(line) for line in records_jsonl.open(encoding="utf-8")]
    if args.dataset == "vaihingen":
        # partial-support semantic maps exist for test areas only (TEST_AREAS)
        image_ids = sorted({str(r["image_id"]) for r in records if r.get("split") == "test"})
    else:
        image_ids = sorted({str(r["image_id"]) for r in records})
    gt_maps = _gt_maps(label_dir, image_ids, color_map)

    if args.phase == "five_method":
        out_rows = []
        for key, info in subsets.items():
            supported, unsupported = info["supported"], info["unsupported"]
            pred_maps = {m: load_label_maps(run_root, key, m, image_ids) for m in METHODS}
            common = common_pixel_metrics(pred_maps, gt_maps, classes, supported, unsupported)
            row = {"subset": key, "k": info.get("k"), "ratio": info.get("ratio"), "seed": info.get("seed"),
                   "supported": "|".join(supported), "unsupported": "|".join(unsupported)}
            for method in METHODS:
                # original metrics: each method scored on its OWN valid pixels
                matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
                valid_total = 0
                for image_id in image_ids:
                    valid = (gt_maps[image_id] != IGNORE) & (pred_maps[method][image_id] != IGNORE)
                    flat = gt_maps[image_id][valid].astype(np.int64) * len(classes) + pred_maps[method][image_id][valid]
                    matrix += np.bincount(flat, minlength=len(classes) * len(classes)).reshape(len(classes), len(classes))
                    valid_total += int(valid.sum())
                orig = metrics_from_matrix(matrix, classes, supported, unsupported)
                for metric in ("OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU"):
                    row[f"{method}_orig_{metric}"] = orig[metric]
                row[f"{method}_orig_valid"] = valid_total
                for metric in ("OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU"):
                    row[f"{method}_common_{metric}"] = common[method][metric]
                row[f"{method}_common_valid"] = common[method]["common_valid_pixels"]
            out_rows.append(row)
        out_path = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/final_audit")
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / f"five_method_metrics_{args.dataset}.json").open("x", encoding="utf-8") as handle:
            json.dump(out_rows, handle, indent=2, sort_keys=True)
        print(f"five-method metrics done for {args.dataset}: {len(out_rows)} subsets")
        return 0

    if args.phase == "common_pixel":
        out_rows = []
        for key, info in subsets.items():
            supported, unsupported = info["supported"], info["unsupported"]
            pred_maps = {m: load_label_maps(run_root, key, m, image_ids) for m in METHODS}
            common = common_pixel_metrics(pred_maps, gt_maps, classes, supported, unsupported)
            orig_valid = per_method_valid_pixels(pred_maps, gt_maps)
            common_total = common["text_only"]["common_valid_pixels"]
            row = {"subset": key, "k": info.get("k"), "ratio": info.get("ratio"), "seed": info.get("seed"),
                   "supported": "|".join(supported), "unsupported": "|".join(unsupported),
                   "common_valid_pixels": common_total}
            for method in METHODS:
                row[f"{method}_OA"] = common[method]["OA"]
                row[f"{method}_macro_f1"] = common[method]["macro_f1"]
                row[f"{method}_mIoU"] = common[method]["mIoU"]
                row[f"{method}_S_F1"] = common[method]["S_F1"]
                row[f"{method}_U_F1"] = common[method]["U_F1"]
                row[f"{method}_H_F1"] = common[method]["H_F1"]
                row[f"{method}_S_IoU"] = common[method]["S_IoU"]
                row[f"{method}_U_IoU"] = common[method]["U_IoU"]
                row[f"{method}_H_IoU"] = common[method]["H_IoU"]
                row[f"{method}_orig_valid"] = orig_valid[method]
                row[f"{method}_coverage_ratio"] = common_total / orig_valid[method] if orig_valid[method] else 0.0
            out_rows.append(row)
        out_path = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/final_audit")
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / f"common_pixel_metrics_{args.dataset}.json").open("x", encoding="utf-8") as handle:
            json.dump(out_rows, handle, indent=2, sort_keys=True)
        print(f"common-pixel done for {args.dataset}: {len(out_rows)} subsets; wrote {out_path / ('common_pixel_metrics_' + args.dataset + '.json')}")
        return 0

    if args.phase == "cluster_bootstrap":
        if args.dataset == "loveda":
            raise InputValidationError("LoveDA cluster bootstrap handled by region-level script; use region scores.")
        seed = 42
        repeats = 5000
        rng = np.random.default_rng(seed)
        # cluster -> image ids
        cluster_to_images: dict[str, list[str]] = {}
        for image_id in image_ids:
            cluster_to_images.setdefault(cluster_key(image_id), []).append(image_id)
        clusters = sorted(cluster_to_images)
        print(f"clusters: {len(clusters)} -> {clusters}")

        DELTA_METRICS = ["OA", "macro_f1", "mIoU", "H_F1", "H_IoU"]
        out_rows = []
        for key, info in subsets.items():
            supported, unsupported = info["supported"], info["unsupported"]
            pred_maps = {m: load_label_maps(run_root, key, m, image_ids) for m in METHODS}
            # per-image confusion matrices on the common valid-pixel intersection (identical mask across methods)
            per_image = per_image_confusion(pred_maps, gt_maps, classes)
            # aggregate per cluster per method
            cluster_matrix = {m: {c: sum((per_image[m][i] for i in cluster_to_images[c]),
                                         np.zeros((len(classes), len(classes)), dtype=np.int64))
                                  for c in clusters} for m in METHODS}
            cluster_metrics = {m: {c: metrics_from_matrix(cluster_matrix[m][c], classes, supported, unsupported)
                                   for c in clusters} for m in METHODS}
            # observed (point) deltas on all clusters together
            full_matrix = {m: sum(cluster_matrix[m].values(),
                                  np.zeros((len(classes), len(classes)), dtype=np.int64)) for m in METHODS}
            full_metrics = {m: metrics_from_matrix(full_matrix[m], classes, supported, unsupported) for m in METHODS}
            # per-cluster direction consistency (CTP vs ref)
            per_cluster = {ref: {c: cluster_metrics["CTP"][c]["H_IoU"] - cluster_metrics[ref][c]["H_IoU"]
                                 for c in clusters} for ref in ("C2", "SCC", "guard", "text_only")}
            # cluster bootstrap: resample clusters, sum their matrices, re-score
            draws = []
            for _ in range(repeats):
                chosen = rng.choice(clusters, size=len(clusters), replace=True)
                mat = {m: sum((cluster_matrix[m][c] for c in chosen),
                              np.zeros((len(classes), len(classes)), dtype=np.int64)) for m in METHODS}
                met = {m: metrics_from_matrix(mat[m], classes, supported, unsupported) for m in METHODS}
                draws.append({ref: {d: met["CTP"][d] - met[ref][d] for d in DELTA_METRICS}
                              for ref in ("C2", "SCC", "guard", "text_only")})
            row = {"subset": key, "k": info.get("k"), "ratio": info.get("ratio"), "seed": info.get("seed"),
                   "clusters": clusters, "cluster_unit": ("area" if args.dataset == "vaihingen" else "parent_tile")}
            for ref in ("C2", "SCC", "guard", "text_only"):
                for d in DELTA_METRICS:
                    arr = np.asarray([draw[ref][d] for draw in draws])
                    row[f"d{d}_vs_{ref}"] = {
                        "point": float(full_metrics["CTP"][d] - full_metrics[ref][d]),
                        "mean": float(arr.mean()),
                        "ci95_low": float(np.percentile(arr, 2.5)),
                        "ci95_high": float(np.percentile(arr, 97.5)),
                        "pct_sign": float(((arr < 0) if full_metrics["CTP"][d] < full_metrics[ref][d]
                                           else (arr > 0)).mean()),
                    }
            row["per_cluster_dH_IoU_vs"] = per_cluster
            out_rows.append(row)
        out_path = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/final_audit")
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / f"cluster_bootstrap_{args.dataset}.json").open("x", encoding="utf-8") as handle:
            json.dump(out_rows, handle, indent=2, sort_keys=True)
        print(f"cluster bootstrap done for {args.dataset}: {len(out_rows)} subsets")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
