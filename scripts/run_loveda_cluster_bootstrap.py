"""Phase B (LoveDA): correct cluster-level bootstrap on region predictions.

Cluster unit = original image_id (NOT patches/regions). All inputs frozen:
- loveda_blind_gt_v0/predictions.npz (text_scores, fused_scores, text_only) and
  heldout_keys.jsonl, verified against the predict-phase manifest hashes BEFORE
  any GT access (same GT-isolation rule as the blind protocol).
- Official LoveDA Train pixel GT (majority vote inside the pixel-pack mask),
  pixel pack, frozen protocol.

Methods per subset (from frozen scores only): text_only, C2 (argmax fused),
SCC, Guard, CTP. Subsets: all bitmask subsets with k=2..5 supported classes
(same enumeration as the freeze record). Bootstrap: seed 42, 5000 repeats,
resampling image_id clusters with replacement; per-cluster direction
consistency recorded. Delta metrics: OA / macro_f1 / mIoU / H_F1 / H_IoU with
point / mean / 95% CI / sign-consistency.

Usage:
    python scripts/run_loveda_cluster_bootstrap.py \
        --config configs/loveda_blind_gt_v0.2080ti.evaluate.yaml \
        --run-root outputs/loveda_blind_gt_v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError, sha256_file, write_json  # noqa: E402
from ov_probe.loveda_partial_support import ctp_predictions, guard_predictions, scc_scores  # noqa: E402
from ov_probe.final_audit import metrics_from_matrix  # noqa: E402

IGNORE = 255
METHODS = ["text_only", "C2", "SCC", "CTP", "guard"]
REFS = ["text_only", "C2", "SCC", "guard"]
DELTA_METRICS = ["OA", "macro_f1", "mIoU", "H_F1", "H_IoU"]


def main() -> int:
    parser = argparse.ArgumentParser(description="LoveDA region-level cluster bootstrap (Phase B).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()

    import yaml
    run_root = Path(args.run_root).resolve()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(cfg["paths"]["protocol_file"]).read_text(encoding="utf-8"))
    classes = [str(c) for c in protocol["classes"]]
    color_map = {name: tuple(int(v) for v in color) for name, color in protocol["ground_truth"]["color_map"].items()}
    if set(color_map) != set(classes):
        raise InputValidationError("GT color map differs from the frozen classes.")

    # ---- verify frozen predictions BEFORE GT access ----
    predict_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if predict_manifest.get("phase") != "predict" or predict_manifest.get("status") != "completed":
        raise InputValidationError("Predict-phase manifest is not a completed predict run.")
    arrays_path = run_root / "predictions.npz"
    keys_path = run_root / "heldout_keys.jsonl"
    if predict_manifest["outputs"]["predictions"]["sha256"] != sha256_file(arrays_path):
        raise InputValidationError("Prediction artifact changed since the predict phase.")
    if predict_manifest["heldout"]["keys_sha256"] != sha256_file(keys_path):
        raise InputValidationError("Heldout keys changed since the predict phase.")

    with np.load(arrays_path, allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        fused_scores = archive["fused_scores"].astype(np.float32)  # C2 anchored score
        text_pred = archive["text_only"].astype(np.int64)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    if text_scores.shape[1] != len(classes) or fused_scores.shape != text_scores.shape:
        raise InputValidationError("Score matrix shapes differ from the frozen classes.")
    records = [json.loads(line) for line in keys_path.open(encoding="utf-8")]
    if len(records) != len(hold_indices) or any(int(r["row_index"]) != int(i) for r, i in zip(records, hold_indices)):
        raise InputValidationError("Heldout key records do not match prediction row indices.")

    # ---- GT access (after verification) ----
    from ov_probe.loveda_blind_gt import _load_record_masks, _region_gt_from_label
    mask_info = _load_record_masks(cfg["paths"]["pixel_pack"])
    label_dir = Path(cfg["paths"]["loveda_label_dir"])
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            from PIL import Image
            with Image.open(label_dir / f"{image_id}_label.png") as im:
                label_cache[image_id] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        info = mask_info.get(int(row["row_index"]))
        if info is None:
            raise InputValidationError(f"No pixel-pack mask for heldout row {row['row_index']}.")
        label, _, _ = _region_gt_from_label(info, label_cache[image_id], color_map)
        gt_labels.append(label)
    label_index = {name: i for i, name in enumerate(classes)}
    labeled = np.asarray([g is not None for g in gt_labels], dtype=bool)
    gt_index = np.asarray([label_index[g] for g in gt_labels if g is not None], dtype=np.int64)
    image_ids = np.asarray([str(r["image_id"]) for r in records], dtype=object)
    print(f"heldout regions: {len(records)}, labeled: {int(labeled.sum())}, "
          f"images: {len(set(image_ids))}")

    # ---- per-region method predictions per subset (frozen scores only) ----
    out_rows = []
    rng = np.random.default_rng(args.seed)
    for subset_index in range(1 << len(classes)):
        mask = np.asarray([(subset_index >> i) & 1 for i in range(len(classes))], dtype=bool)
        k = int(mask.sum())
        if k < 2 or k > 5:  # same k-range as the freeze record (partial support regime)
            continue
        supported = [classes[i] for i, flag in enumerate(mask) if flag]
        unsupported = [classes[i] for i, flag in enumerate(mask) if not flag]
        c2 = fused_scores.copy()
        c2[:, ~mask] = text_scores[:, ~mask]
        scc = scc_scores(text_scores, fused_scores, mask)
        u_idx = [i for i, flag in enumerate(mask) if not flag]
        pred = {
            "text_only": text_pred,
            "C2": np.argmax(c2, axis=1),
            "SCC": np.argmax(scc, axis=1),
            "guard": guard_predictions(text_pred, c2, u_idx),
            "CTP": ctp_predictions(text_pred, text_scores, scc, mask),
        }
        # per-region labeled confusion (5x5 per image per method)
        labeled_pos = np.where(labeled)[0]
        pos_to_local = {int(p): i for i, p in enumerate(labeled_pos)}
        n_classes = len(classes)
        image_to_regions: dict[str, list[int]] = {}
        for pos in labeled_pos:
            image_to_regions.setdefault(str(image_ids[pos]), []).append(int(pos))
        clusters = sorted(image_to_regions)
        per_image_matrix = {m: {c: np.zeros((n_classes, n_classes), dtype=np.int64)
                                for c in clusters} for m in METHODS}
        for m in METHODS:
            p = pred[m][hold_indices[labeled_pos]]
            g = gt_index
            for c in clusters:
                idx = np.asarray([pos_to_local[pos] for pos in image_to_regions[c]], dtype=np.int64)
                np.add.at(per_image_matrix[m][c], (g[idx], p[idx]), 1)
        cluster_metrics = {m: {c: metrics_from_matrix(per_image_matrix[m][c], classes, supported, unsupported)
                               for c in clusters} for m in METHODS}
        full_matrix = {m: sum(per_image_matrix[m].values(), np.zeros((n_classes, n_classes), dtype=np.int64))
                       for m in METHODS}
        full_metrics = {m: metrics_from_matrix(full_matrix[m], classes, supported, unsupported) for m in METHODS}
        per_cluster = {ref: {c: cluster_metrics["CTP"][c]["H_IoU"] - cluster_metrics[ref][c]["H_IoU"]
                             for c in clusters} for ref in REFS}
        draws = []
        for _ in range(args.repeats):
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            mat = {m: sum((per_image_matrix[m][c] for c in chosen),
                          np.zeros((n_classes, n_classes), dtype=np.int64)) for m in METHODS}
            met = {m: metrics_from_matrix(mat[m], classes, supported, unsupported) for m in METHODS}
            draws.append({ref: {d: met["CTP"][d] - met[ref][d] for d in DELTA_METRICS} for ref in REFS})
        row = {"subset": subset_index, "k": k, "supported": "|".join(supported),
               "unsupported": "|".join(unsupported), "n_clusters": len(clusters),
               "cluster_unit": "image_id", "n_labeled_regions": int(labeled.sum())}
        for ref in REFS:
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
        print(f"subset {subset_index} k={k} done ({len(clusters)} clusters)")

    out_path = Path("/home/undergr/Sheungzhen_project_2/second_open_vocabulary/workspaces/openai_clip_715cb5c/outputs/final_audit")
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "cluster_bootstrap_loveda.json").open("x", encoding="utf-8") as handle:
        json.dump(out_rows, handle, indent=2, sort_keys=True)
    print(f"wrote {out_path / 'cluster_bootstrap_loveda.json'}: {len(out_rows)} subsets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
