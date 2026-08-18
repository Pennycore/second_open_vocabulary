"""Phase P: LoveDA pre-freeze final metric audit.

Recomputes the exhaustive 64-subset benchmark with the unified metric set:
OA / Macro F1 / mIoU, S-F1 / U-F1 / H-F1, S-IoU / U-IoU / H-IoU per subset,
then aggregates by support coverage k with mean +/- std.

H aggregation is mean over subsets of per-subset H (mean(H_i)), NOT
H(mean(S_i), mean(U_i)); the report states this explicitly.

Also verifies: SCC k=0 strictly equals Text-only, SCC k=6 strictly equals C2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError  # noqa: E402
from ov_probe.loveda_blind_gt import (  # noqa: E402
    _CLASSES,
    _load_record_masks,
    _region_gt_from_label,
    load_loveda_blind_gt_config,
)
from ov_probe.loveda_partial_support import benchmark_all_subsets  # noqa: E402

METHODS = ["text_only", "C1", "C2", "SCC", "guard"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase P final metric audit.")
    parser.add_argument("--config", required=True, help="Evaluate-phase deployment YAML.")
    parser.add_argument("--run-dir", required=True, help="Run directory with predictions.npz / heldout_keys.jsonl.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
    run_dir = Path(args.run_dir).resolve()

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        text_protos = archive["text_prototypes"].astype(np.float32)
        visual_protos = archive["visual_prototypes"].astype(np.float32)
        text_pred_frozen = archive["text_only"].astype(np.int64)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    records = [json.loads(line) for line in (run_dir / "heldout_keys.jsonl").open(encoding="utf-8")]
    label_dir = cfg["paths"]["loveda_label_dir"]
    color_map = {name: tuple(int(v) for v in color) for name, color in protocol["ground_truth"]["color_map"].items()}
    mask_info = _load_record_masks(cfg["paths"]["pixel_pack"])
    from PIL import Image
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            with Image.open(Path(label_dir) / f"{image_id}_label.png") as image:
                label_cache[image_id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        info = mask_info.get(int(row["row_index"]))
        if info is None:
            raise InputValidationError(f"No pixel-pack mask for row {row['row_index']}.")
        label, _, _ = _region_gt_from_label(info, label_cache[image_id], color_map)
        gt_labels.append(label)
    labeled = np.asarray([gt is not None for gt in gt_labels], dtype=bool)
    gt_index = np.asarray([_CLASSES.index(gt) for gt in gt_labels if gt is not None], dtype=np.int64)
    hold_labeled = hold_indices[labeled]

    subset_rows, k_rows, checks = benchmark_all_subsets(
        text_scores, visual_scores, text_protos, visual_protos,
        text_pred_frozen, gt_index, hold_labeled, _CLASSES,
    )

    # Rebuild per-subset S/U/H in F1 and IoU with the same definitions used in the
    # benchmark (S/U means over supported/unsupported classes of per-class metric).
    # benchmark_all_subsets already writes S/U/H in F1 and OA/macro_f1/macro_iou.
    # Extend rows with IoU-based S/U/H by recomputing per-subset predictions here.
    from ov_probe.loveda_partial_support import _c2_normalizers, _metrics, guard_predictions, scc_scores

    normalizers = _c2_normalizers(text_protos, visual_protos)
    anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]
    c1_base = 0.5 * text_scores + 0.5 * visual_scores

    def subset_preds(mask: np.ndarray):
        scc = scc_scores(text_scores, anchored, mask)
        c2 = anchored.copy()
        c2[:, ~mask] = text_scores[:, ~mask]
        c1 = c1_base.copy()
        c1[:, ~mask] = text_scores[:, ~mask]
        u_idx = [i for i, flag in enumerate(mask) if not flag]
        guard = guard_predictions(text_pred_frozen, c2, u_idx)
        scc_pred = np.argmax(scc, axis=1)
        if not mask.any():
            # SCC with |S|=0 must strictly degenerate to frozen Text-only predictions.
            scc_pred = text_pred_frozen
        return {
            "text_only": text_pred_frozen,
            "C1": np.argmax(c1, axis=1),
            "C2": np.argmax(c2, axis=1),
            "SCC": scc_pred,
            "guard": guard,
        }

    # Per-subset full rows: subset, k, supported, unsupported,
    # then per method OA, macro_f1, mIoU, S-F1, U-F1, H-F1, S-IoU, U-IoU, H-IoU
    header = ["subset_index", "k", "supported", "unsupported"]
    for name in METHODS:
        header += [f"{name}_OA", f"{name}_macro_f1", f"{name}_mIoU",
                   f"{name}_S_F1", f"{name}_U_F1", f"{name}_H_F1",
                   f"{name}_S_IoU", f"{name}_U_IoU", f"{name}_H_IoU"]
    rows: list[list] = []
    for subset_index in range(1 << len(_CLASSES)):
        mask = np.asarray([(subset_index >> i) & 1 for i in range(len(_CLASSES))], dtype=bool)
        k = int(mask.sum())
        supported = [name for name, flag in zip(_CLASSES, mask) if flag]
        unsupported = [name for name, flag in zip(_CLASSES, mask) if not flag]
        preds = subset_preds(mask)
        row = [subset_index, k, "|".join(supported), "|".join(unsupported)]
        for name in METHODS:
            m = _metrics(preds[name][hold_labeled], gt_index, _CLASSES)
            s_f1 = float(np.mean([m["per_class_f1"][c] for c in supported])) if supported else float("nan")
            u_f1 = float(np.mean([m["per_class_f1"][c] for c in unsupported])) if unsupported else float("nan")
            s_iou = float(np.mean([m["per_class_iou"][c] for c in supported])) if supported else float("nan")
            u_iou = float(np.mean([m["per_class_iou"][c] for c in unsupported])) if unsupported else float("nan")
            h_f1 = 2 * s_f1 * u_f1 / (s_f1 + u_f1) if supported and unsupported and s_f1 + u_f1 > 0 else 0.0
            h_iou = 2 * s_iou * u_iou / (s_iou + u_iou) if supported and unsupported and s_iou + u_iou > 0 else 0.0
            row += [f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}", f"{m['macro_iou']:.6f}",
                    f"{s_f1:.6f}", f"{u_f1:.6f}", f"{h_f1:.6f}",
                    f"{s_iou:.6f}", f"{u_iou:.6f}", f"{h_iou:.6f}"]
        rows.append(row)

    with (run_dir / "phaseP_final_metrics.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")

    # Aggregate by k: mean +/- std of OA, macro_f1, mIoU, S-F1, U-F1, H-F1, S-IoU, U-IoU, H-IoU
    agg_header = ["k", "method", "n", "OA_mean", "OA_std", "macroF1_mean", "macroF1_std", "mIoU_mean", "mIoU_std",
                  "S_F1_mean", "S_F1_std", "U_F1_mean", "U_F1_std", "H_F1_mean", "H_F1_std",
                  "S_IoU_mean", "S_IoU_std", "U_IoU_mean", "U_IoU_std", "H_IoU_mean", "H_IoU_std"]
    agg_rows: list[list] = []
    for k in range(0, 7):
        subset_k = [r for r in rows if int(r[1]) == k]
        for mi, name in enumerate(METHODS):
            base = 4 + 9 * mi
            def col(idx):
                vals = [float(r[base + idx]) for r in subset_k]
                return vals
            def ms(idx):
                vals = col(idx)
                return (f"{float(np.mean(vals)):.4f}" if vals else "nan",
                        f"{float(np.std(vals)):.4f}" if vals else "nan")
            oa = ms(0); mf = ms(1); iou = ms(2)
            sf = ms(3); uf = ms(4); hf = ms(5)
            si = ms(6); ui = ms(7); hi = ms(8)
            agg_rows.append([k, name, str(len(subset_k)), oa[0], oa[1], mf[0], mf[1], iou[0], iou[1],
                             sf[0], sf[1], uf[0], uf[1], hf[0], hf[1], si[0], si[1], ui[0], ui[1], hi[0], hi[1]])
    with (run_dir / "phaseP_metrics_by_k.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(agg_header) + "\n")
        for row in agg_rows:
            handle.write(",".join(str(v) for v in row) + "\n")

    print("H aggregation rule: mean over subsets of per-subset H (mean(H_i)); not H(mean(S), mean(U)).")
    print(f"SCC k=0 == Text-only: {checks['scc_k0_equals_text']}")
    print(f"SCC k=6 == C2: {checks['scc_k6_equals_c2']}")
    print("\n=== k=1..5 H-F1 and H-IoU (mean over subsets) ===")
    for row in agg_rows:
        if row[0] in (1, 2, 3, 4, 5):
            print("  k=%s %-10s H_F1=%s H_IoU=%s (n=%s)" % (row[0], row[1], row[14], row[20], row[2]))
    print("\nwrote", run_dir / "phaseP_final_metrics.csv", "and", run_dir / "phaseP_metrics_by_k.csv")
    return 0 if (checks["scc_k0_equals_text"] and checks["scc_k6_equals_c2"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
