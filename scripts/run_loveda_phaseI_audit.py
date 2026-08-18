"""Phase I: metric consistency audit of the Phase B unsupported diagnosis.

Recomputes per-class integer counts with CORRECT row indexing (row_index into the
6000-row frozen prediction arrays, not heldout position), and verifies:
  - recall_text = TP_text / n_gt
  - F1_text <= 2*recall/(1+recall)
  - A_count == TP_text when fusion unsupported TP == 0
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase I metric consistency audit.")
    parser.add_argument("--config", required=True, help="Evaluate-phase deployment YAML.")
    parser.add_argument("--run-dir", required=True, help="Run directory with predictions.npz / heldout_keys.jsonl.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
    run_dir = Path(args.run_dir).resolve()

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        text_pred_frozen = archive["text_only"].astype(np.int64)
        fused_pred_frozen = archive["fused_text_visual"].astype(np.int64)
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

    label_index = {name: i for i, name in enumerate(_CLASSES)}
    labeled = np.asarray([gt is not None for gt in gt_labels], dtype=bool)
    # CORRECT alignment: prediction arrays are indexed by row_index; hold_indices
    # carries the row_index per heldout record in record order.
    gt_by_row = {int(row["row_index"]): gt for row, gt in zip(records, gt_labels) if gt is not None}

    # per-class counts over all labeled heldout regions
    rows_out = [["class", "n_gt", "TP_text", "FP_text", "FN_text", "TP_fusion", "FP_fusion", "FN_fusion",
                 "precision_text", "recall_text", "f1_text", "precision_fusion", "recall_fusion", "f1_fusion",
                 "f1_upper_bound_2r/(1+r)", "recall_ok", "f1_bound_ok"]]
    all_ok = True
    for unsupported in _CLASSES:
        u = label_index[unsupported]
        n_gt = 0
        tp_t = fp_t = fn_t = 0
        tp_f = fp_f = fn_f = 0
        for row_idx, gt_name in gt_by_row.items():
            if gt_name != unsupported:
                continue
            n_gt += 1
            pred_t = int(text_pred_frozen[row_idx])
            pred_f = int(fused_pred_frozen[row_idx])
            if pred_t == u:
                tp_t += 1
            else:
                fn_t += 1
                if pred_t != u:
                    pass
            # FP computed globally below (needs pred==u with gt!=u); collect here via second pass
        # second pass for FP
        for row_idx, gt_name in gt_by_row.items():
            pred_t = int(text_pred_frozen[row_idx])
            pred_f = int(fused_pred_frozen[row_idx])
            if pred_t == u and gt_name != unsupported:
                fp_t += 1
            if pred_f == u and gt_name != unsupported:
                fp_f += 1
            if pred_f == u and gt_name == unsupported:
                tp_f += 1
        # FN_fusion
        fn_f = n_gt - tp_f
        prec_t = tp_t / (tp_t + fp_t) if tp_t + fp_t else 0.0
        rec_t = tp_t / n_gt if n_gt else 0.0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if prec_t + rec_t else 0.0
        prec_f = tp_f / (tp_f + fp_f) if tp_f + fp_f else 0.0
        rec_f = tp_f / n_gt if n_gt else 0.0
        f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if prec_f + rec_f else 0.0
        upper = 2 * rec_t / (1 + rec_t) if rec_t < 1 else 1.0
        recall_ok = abs(rec_t - tp_t / n_gt) < 1e-12 if n_gt else True
        bound_ok = f1_t <= upper + 1e-9
        all_ok = all_ok and recall_ok and bound_ok
        rows_out.append([unsupported, n_gt, tp_t, fp_t, fn_t, tp_f, fp_f, fn_f,
                         f"{prec_t:.4f}", f"{rec_t:.4f}", f"{f1_t:.4f}",
                         f"{prec_f:.4f}", f"{rec_f:.4f}", f"{f1_f:.4f}",
                         f"{upper:.4f}", str(recall_ok), str(bound_ok)])
        print(f"{unsupported:11s} n_gt={n_gt:4d} TP_t={tp_t:4d} FP_t={fp_t:4d} FN_t={fn_t:4d} "
              f"prec_t={prec_t:.4f} rec_t={rec_t:.4f} f1_t={f1_t:.4f} (bound {upper:.4f}) | "
              f"TP_f={tp_f:4d} FP_f={fp_f:4d} FN_f={fn_f:4d} prec_f={prec_f:.4f} rec_f={rec_f:.4f} f1_f={f1_f:.4f}")

    # A == TP_text check for each LOCO fold (fusion unsupported TP must be 0 for A==TP_text)
    a_rows = [["fold_unsupported", "n_gt", "A_count(text_ok_fusion_wrong)", "TP_text", "TP_fusion", "A_eq_TP_text"]]
    a_ok = True
    for unsupported in _CLASSES:
        u = label_index[unsupported]
        loo = 0.5 * text_scores + 0.5 * visual_scores
        loo[:, u] = text_scores[:, u]
        loo_pred = np.argmax(loo, axis=1)
        n_gt = tp_t = a_count = tp_f = 0
        for row_idx, gt_name in gt_by_row.items():
            if gt_name != unsupported:
                continue
            n_gt += 1
            pred_t = int(text_pred_frozen[row_idx])
            pred_l = int(loo_pred[row_idx])
            if pred_t == u:
                tp_t += 1
                if pred_l != u:
                    a_count += 1
            if pred_l == u:
                tp_f += 1
        eq = (a_count == tp_t) or (tp_f == 0 and a_count == tp_t)
        a_ok = a_ok and (a_count == tp_t)
        a_rows.append([unsupported, n_gt, a_count, tp_t, tp_f, str(a_count == tp_t)])
        print(f"  LOCO {unsupported:11s} n_gt={n_gt:4d} A={a_count:4d} TP_text={tp_t:4d} TP_fusion={tp_f:4d} A==TP_text: {a_count == tp_t}")

    with (run_dir / "phaseI_metric_audit.csv").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows_out:
            handle.write(",".join(str(v) for v in row) + "\n")
        handle.write("\n")
        for row in a_rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print(f"\nALL RECALL/F1 CHECKS PASSED: {all_ok}; ALL A==TP_text: {a_ok}")
    print("wrote", run_dir / "phaseI_metric_audit.csv")
    return 0 if (all_ok and a_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
