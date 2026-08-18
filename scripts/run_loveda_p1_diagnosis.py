"""Phase B: unsupported-collapse root-cause diagnosis for the six LOCO folds.

Reads only the frozen P0/P1 predictions and derives per-region diagnostics for
GT=unsupported regions: text score of the GT class, final fused score of the GT
class, winning supported class and its score, margin, text-only prediction and
its correctness, plus the A/B/C confusion statistics per fold.
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

COLS = [
    "fold_unsupported", "row_index", "image_id", "candidate_index",
    "gt_class", "sam3_source_label",
    "T_gt", "S_gt", "pred_supported", "S_win", "margin",
    "pred_text", "text_correct",
]


def _load_gt(cfg, records):
    label_dir = cfg["paths"]["loveda_label_dir"]
    color_map = {name: tuple(int(v) for v in color) for name, color in
                 json.loads(Path(cfg["paths"]["protocol_file"]).read_text(encoding="utf-8"))["ground_truth"]["color_map"].items()}
    mask_info = _load_record_masks(cfg["paths"]["pixel_pack"])
    from PIL import Image
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    for row in records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            label_path = Path(label_dir) / f"{image_id}_label.png"
            with Image.open(label_path) as image:
                label_cache[image_id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        info = mask_info.get(int(row["row_index"]))
        if info is None:
            raise InputValidationError(f"No pixel-pack mask for row {row['row_index']}.")
        label, _, _ = _region_gt_from_label(info, label_cache[image_id], color_map)
        gt_labels.append(label)
    return gt_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 unsupported-collapse diagnosis.")
    parser.add_argument("--config", required=True, help="Evaluate-phase deployment YAML (provides pixel_pack + label dir).")
    parser.add_argument("--run-dir", required=True, help="Run directory containing predictions.npz and heldout_keys.jsonl.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
    run_dir = Path(args.run_dir).resolve()

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    records = [json.loads(line) for line in (run_dir / "heldout_keys.jsonl").open(encoding="utf-8")]
    gt_labels = _load_gt(cfg, records)

    label_index = {name: i for i, name in enumerate(_CLASSES)}
    labeled = [gt is not None for gt in gt_labels]
    gt_index = np.asarray([label_index[gt] for gt in gt_labels if gt is not None], dtype=np.int64)
    hold_labeled = hold_indices[np.asarray(labeled, dtype=bool)]
    pos = {int(row): i for i, row in enumerate(hold_indices.tolist())}

    per_region: list[list] = []
    summary_rows: list[list] = []

    for unsupported in _CLASSES:
        u_idx = label_index[unsupported]
        loo = 0.5 * text_scores + 0.5 * visual_scores
        loo[:, u_idx] = text_scores[:, u_idx]
        loo_pred = np.argmax(loo, axis=1)
        text_pred = np.argmax(text_scores, axis=1)
        supported = [name for name in _CLASSES if name != unsupported]

        rows = []
        for position, (row_idx, gt_name) in enumerate(zip(hold_indices.tolist(), gt_labels)):
            if gt_name != unsupported:
                continue
            r = pos[row_idx]
            T_gt = float(text_scores[r, u_idx])
            S_gt = float(loo[r, u_idx])
            pred_supported_name = _CLASSES[int(loo_pred[r])]
            S_win = float(loo[r, int(loo_pred[r])])
            pred_text_name = _CLASSES[int(text_pred[r])]
            rows.append({
                "row_index": row_idx,
                "image_id": str(records[position]["image_id"]),
                "candidate_index": int(records[position]["candidate_index"]),
                "sam3_source_label": str(records[position].get("sam3_source_label", "")),
                "T_gt": T_gt, "S_gt": S_gt,
                "pred_supported": pred_supported_name, "S_win": S_win,
                "margin": S_win - S_gt,
                "pred_text": pred_text_name,
                "text_correct": pred_text_name == unsupported,
                "final_correct": pred_supported_name == unsupported,
            })
        n = len(rows)
        text_ok_final_wrong = sum(1 for r in rows if r["text_correct"] and not r["final_correct"])
        text_wrong = sum(1 for r in rows if not r["text_correct"])
        text_top1_stolen = sum(1 for r in rows if r["text_correct"] and not r["final_correct"])
        margins = [r["margin"] for r in rows]

        def f1_stats(all_pred):
            if n == 0:
                return "0", "0", "0"
            gt_all = gt_index
            pred_all = all_pred[hold_labeled]
            tp_all = int(((pred_all == u_idx) & (gt_all == u_idx)).sum())
            fp_all = int(((pred_all == u_idx) & (gt_all != u_idx)).sum())
            fn_all = int(((pred_all != u_idx) & (gt_all == u_idx)).sum())
            precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
            recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return f"{f1:.4f}", f"{recall:.4f}", f"{precision:.4f}"

        text_f1, text_rec, text_prec = f1_stats(text_pred)
        fus_f1, fus_rec, fus_prec = f1_stats(loo_pred)
        # destination distribution of the GT=unsupported regions under fusion
        from collections import Counter
        dest = Counter(r["pred_supported"] for r in rows)
        dest_str = ";".join(f"{name}:{dest.get(name, 0)}" for name in _CLASSES)
        # per-region csv rows
        for r in rows:
            per_region.append([
                unsupported, r["row_index"], r["image_id"], r["candidate_index"],
                unsupported, r["sam3_source_label"],
                f"{r['T_gt']:.6f}", f"{r['S_gt']:.6f}", r["pred_supported"],
                f"{r['S_win']:.6f}", f"{r['margin']:.6f}",
                r["pred_text"], "1" if r["text_correct"] else "0",
            ])
        summary_rows.append([
            unsupported,
            n,
            f"{text_ok_final_wrong / n:.4f}" if n else "0",
            f"{text_wrong / n:.4f}" if n else "0",
            f"{text_top1_stolen / n:.4f}" if n else "0",
            f"{float(np.mean(margins)):.4f}" if margins else "0",
            f"{float(np.std(margins)):.4f}" if margins else "0",
            f"{float(np.median(margins)):.4f}" if margins else "0",
            text_f1, text_rec, text_prec,
            fus_f1, fus_rec, fus_prec,
            dest_str,
        ])
        print(f"fold={unsupported:11s} n={n:4d} A={text_ok_final_wrong:4d} ({text_ok_final_wrong/max(n,1):.3f}) "
              f"B={text_wrong:4d} ({text_wrong/max(n,1):.3f}) "
              f"C={text_top1_stolen:4d} ({text_top1_stolen/max(n,1):.3f}) "
              f"margin_mean={np.mean(margins):+.4f} median={np.median(margins):+.4f} "
              f"textF1={text_f1} fusF1={fus_f1} dest={dest_str}")

    with (run_dir / "p1_diagnosis_per_region.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(COLS) + "\n")
        for row in per_region:
            handle.write(",".join(str(v) for v in row) + "\n")
    with (run_dir / "p1_diagnosis_summary.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("fold_unsupported,n,A_text_ok_final_wrong_frac,B_text_wrong_frac,C_text_top1_stolen_frac,margin_mean,margin_std,margin_median,text_unsup_f1,text_unsup_recall,text_unsup_precision,fusion_unsup_f1,fusion_unsup_recall,fusion_unsup_precision,destination_distribution\n")
        for row in summary_rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print("wrote", run_dir / "p1_diagnosis_per_region.csv", "and", run_dir / "p1_diagnosis_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
