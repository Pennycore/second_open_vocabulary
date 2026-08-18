"""Phases C/D/E/F: C0/C1/C2 calibration candidates, LOCO S/U/H, fully-supported
comparison, and image-cluster bootstrap of the original P0 results.

All candidates are training-free and reuse the frozen P0 predictions:
- C0: current original implementation (unsupported class keeps the full text
      score, i.e. identical to C1 by the Phase A audit).
- C1: support-aware text fallback: supported S_c = 0.5*T_c + 0.5*V_c,
      unsupported S_c = T_c.
- C2: normalized prototype calibration: supported p_c = L2(0.5*t_c + 0.5*v_c)
      and S_c = cosine(x, p_c); unsupported p_c = t_c and S_c = cosine(x, t_c).
      Since p_c is a unit vector, S_c = (0.5*T_c + 0.5*V_c) / ||0.5*t_c+0.5*v_c||,
      computable exactly from the frozen T/V scores and stored prototypes.
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


def _metrics(pred_all: np.ndarray, gt_all: np.ndarray, classes: list[str]) -> dict:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for p, g in zip(pred_all, gt_all):
        matrix[int(g), int(p)] += 1
    per_f1, per_iou = {}, {}
    for i, name in enumerate(classes):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_f1[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_iou[name] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    acc = float((pred_all == gt_all).mean())
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(list(per_f1.values()))),
        "macro_iou": float(np.mean(list(per_iou.values()))),
        "per_class_f1": per_f1,
        "per_class_iou": per_iou,
        "confusion_matrix": matrix.tolist(),
        "count": int(len(pred_all)),
    }


def _c2_normalizers(text_protos: np.ndarray, visual_protos: np.ndarray) -> np.ndarray:
    """Per-class ||0.5*t_c + 0.5*v_c|| from stored unit prototypes."""
    fused = 0.5 * text_protos + 0.5 * visual_protos
    norms = np.linalg.norm(fused, axis=1)
    if np.any(norms <= 1e-6):
        raise InputValidationError("C2 prototype fusion degenerates to a zero vector for some class.")
    return norms


def _c2_scores(text_scores: np.ndarray, visual_scores: np.ndarray, normalizers: np.ndarray) -> np.ndarray:
    return (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]


def _c1_loo_scores(text_scores: np.ndarray, visual_scores: np.ndarray, unsupported_idx: int) -> np.ndarray:
    loo = 0.5 * text_scores + 0.5 * visual_scores
    loo[:, unsupported_idx] = text_scores[:, unsupported_idx]
    return loo


def _c2_loo_scores(text_scores: np.ndarray, visual_scores: np.ndarray, normalizers: np.ndarray, unsupported_idx: int) -> np.ndarray:
    loo = _c2_scores(text_scores, visual_scores, normalizers)
    loo[:, unsupported_idx] = text_scores[:, unsupported_idx]
    return loo


def _load_gt(cfg, protocol, records):
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
    return gt_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="C0/C1/C2 calibration comparison + bootstrap.")
    parser.add_argument("--config", required=True, help="Evaluate-phase deployment YAML.")
    parser.add_argument("--run-dir", required=True, help="Run directory with predictions.npz / heldout_keys.jsonl.")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg, protocol = load_loveda_blind_gt_config(args.config, project_root)
    run_dir = Path(args.run_dir).resolve()

    with np.load(run_dir / "predictions.npz", allow_pickle=False) as archive:
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        fused_scores = archive["fused_scores"].astype(np.float32)
        text_protos = archive["text_prototypes"].astype(np.float32)
        visual_protos = archive["visual_prototypes"].astype(np.float32)
        text_pred_frozen = archive["text_only"].astype(np.int64)
        visual_pred_frozen = archive["visual_only"].astype(np.int64)
        fused_pred_frozen = archive["fused_text_visual"].astype(np.int64)
        hold_indices = archive["heldout_row_indices"].astype(np.int64)
    records = [json.loads(line) for line in (run_dir / "heldout_keys.jsonl").open(encoding="utf-8")]
    gt_labels = _load_gt(cfg, protocol, records)

    label_index = {name: i for i, name in enumerate(_CLASSES)}
    labeled = np.asarray([gt is not None for gt in gt_labels], dtype=bool)
    gt_index = np.asarray([label_index[gt] for gt in gt_labels if gt is not None], dtype=np.int64)
    hold_labeled = hold_indices[labeled]
    image_ids_labeled = [str(r["image_id"]) for r, keep in zip(records, labeled) if keep]
    normalizers = _c2_normalizers(text_protos, visual_protos)

    # ---------- Phase D: LOCO for C1 and C2 (C0 == C1 by Phase A audit) ----------
    loo_rows: list[list] = []
    summary_rows: list[list] = []
    for unsupported in _CLASSES:
        u = label_index[unsupported]
        supported = [c for c in _CLASSES if c != unsupported]
        c1_pred = np.argmax(_c1_loo_scores(text_scores, visual_scores, u), axis=1)
        c2_pred = np.argmax(_c2_loo_scores(text_scores, visual_scores, normalizers, u), axis=1)
        c1_sel = c1_pred[hold_labeled]
        c2_sel = c2_pred[hold_labeled]
        m1 = _metrics(c1_sel, gt_index, _CLASSES)
        m2 = _metrics(c2_sel, gt_index, _CLASSES)
        s1 = float(np.mean([m1["per_class_f1"][c] for c in supported]))
        u1 = m1["per_class_f1"][unsupported]
        s2 = float(np.mean([m2["per_class_f1"][c] for c in supported]))
        u2 = m2["per_class_f1"][unsupported]
        h1 = 2 * s1 * u1 / (s1 + u1) if s1 + u1 else 0.0
        h2 = 2 * s2 * u2 / (s2 + u2) if s2 + u2 else 0.0
        loo_rows.append([unsupported, "C1", f"{s1:.6f}", f"{u1:.6f}", f"{h1:.6f}", f"{m1['macro_f1']:.6f}", f"{m1['accuracy']:.6f}", f"{m1['macro_iou']:.6f}"])
        loo_rows.append([unsupported, "C2", f"{s2:.6f}", f"{u2:.6f}", f"{h2:.6f}", f"{m2['macro_f1']:.6f}", f"{m2['accuracy']:.6f}", f"{m2['macro_iou']:.6f}"])
        summary_rows.append({
            "unsupported": unsupported,
            "C1_supported_F1": s1, "C1_unsupported_F1": u1, "C1_H": h1,
            "C1_all_macro_f1": m1["macro_f1"], "C1_accuracy": m1["accuracy"], "C1_macro_iou": m1["macro_iou"],
            "C2_supported_F1": s2, "C2_unsupported_F1": u2, "C2_H": h2,
            "C2_all_macro_f1": m2["macro_f1"], "C2_accuracy": m2["accuracy"], "C2_macro_iou": m2["macro_iou"],
        })
        print(f"fold={unsupported:11s} C1 S={s1:.4f} U={u1:.4f} H={h1:.4f} | C2 S={s2:.4f} U={u2:.4f} H={h2:.4f}")

    # text-only baseline under identical folds
    text_pred = text_pred_frozen[hold_labeled]
    text_metrics = _metrics(text_pred, gt_index, _CLASSES)
    s_text_list, u_text_list = [], []
    for unsupported in _CLASSES:
        supported = [c for c in _CLASSES if c != unsupported]
        s_text_list.append(float(np.mean([text_metrics["per_class_f1"][c] for c in supported])))
        u_text_list.append(text_metrics["per_class_f1"][unsupported])
    S_text = float(np.mean(s_text_list))
    U_text = float(np.mean(u_text_list))
    H_text = 2 * S_text * U_text / (S_text + U_text) if S_text + U_text else 0.0

    # ---------- Phase E: fully-supported comparison ----------
    visual_pred = visual_pred_frozen[hold_labeled]
    fused_pred = fused_pred_frozen[hold_labeled]
    c2_pred_full = np.argmax(_c2_scores(text_scores, visual_scores, normalizers), axis=1)[hold_labeled]
    full = {
        "text_only": _metrics(text_pred, gt_index, _CLASSES),
        "visual_only": _metrics(visual_pred, gt_index, _CLASSES),
        "C0_fused": _metrics(fused_pred, gt_index, _CLASSES),
        "C1_fused": _metrics(fused_pred, gt_index, _CLASSES),
        "C2": _metrics(c2_pred_full, gt_index, _CLASSES),
    }
    print("\n=== fully-supported ===")
    for name, m in full.items():
        print(f"{name:12s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} macro_iou={m['macro_iou']:.4f}")
    print("C2 normalizers ||0.5*t+0.5*v||:", {c: f"{float(n):.4f}" for c, n in zip(_CLASSES, normalizers)})
    print("text-visual prototype cosine:", {c: f"{float(np.dot(text_protos[i], visual_protos[i])):.4f}" for i, c in enumerate(_CLASSES)})

    # ---------- Phase F: image-cluster bootstrap of original P0 ----------
    rng = np.random.default_rng(args.bootstrap_seed)
    images = sorted(set(image_ids_labeled))
    image_to_positions: dict[str, list[int]] = {}
    for pos, image_id in enumerate(image_ids_labeled):
        image_to_positions.setdefault(image_id, []).append(pos)
    boot_metrics = {"text_only": [], "visual_only": [], "fused_text_visual": [], "delta_f1": [], "delta_iou": [], "delta_acc": []}
    for _ in range(args.bootstrap_repeats):
        chosen_images = rng.choice(images, size=len(images), replace=True)
        positions = np.concatenate([np.asarray(image_to_positions[img], dtype=np.int64) for img in chosen_images])
        gt_b = gt_index[positions]
        mt = _metrics(text_pred[positions], gt_b, _CLASSES)
        mv = _metrics(visual_pred[positions], gt_b, _CLASSES)
        mf = _metrics(fused_pred[positions], gt_b, _CLASSES)
        boot_metrics["text_only"].append(mt["macro_f1"])
        boot_metrics["visual_only"].append(mv["macro_f1"])
        boot_metrics["fused_text_visual"].append(mf["macro_f1"])
        boot_metrics["delta_f1"].append(mf["macro_f1"] - mt["macro_f1"])
        boot_metrics["delta_iou"].append(mf["macro_iou"] - mt["macro_iou"])
        boot_metrics["delta_acc"].append(mf["accuracy"] - mt["accuracy"])

    def ci(values):
        arr = np.asarray(values)
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    print("\n=== image-cluster bootstrap (5000, seed 42) ===")
    fused_full = full["C0_fused"]
    text_full = full["text_only"]
    point_f1 = fused_full["macro_f1"] - text_full["macro_f1"]
    point_iou = fused_full["macro_iou"] - text_full["macro_iou"]
    point_acc = fused_full["accuracy"] - text_full["accuracy"]
    for key, point in (("delta_f1", point_f1), ("delta_iou", point_iou), ("delta_acc", point_acc)):
        mean, lo, hi = ci(boot_metrics[key])
        print(f"{key:10s} point={point:+.4f} boot_mean={mean:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]")

    # ---------- write outputs ----------
    with (run_dir / "calibration_loco.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("fold_unsupported,method,supported_macro_f1,unsupported_f1,harmonic_H,all_macro_f1,accuracy,macro_iou\n")
        for row in loo_rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    with (run_dir / "calibration_loco_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "folds": summary_rows,
            "text_only_baseline": {"S_text": S_text, "U_text": U_text, "H_text": H_text,
                                   "per_fold_S_text": s_text_list, "per_fold_U_text": u_text_list},
        }, handle, indent=2, sort_keys=True)
    with (run_dir / "calibration_fully_supported.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("method,accuracy,macro_f1,macro_iou\n")
        for name, m in full.items():
            handle.write(f"{name},{m['accuracy']:.6f},{m['macro_f1']:.6f},{m['macro_iou']:.6f}\n")
    with (run_dir / "calibration_fully_supported_per_class.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("method,class,f1,iou\n")
        for name, m in full.items():
            for c in _CLASSES:
                handle.write(f"{name},{c},{m['per_class_f1'][c]:.6f},{m['per_class_iou'][c]:.6f}\n")
    with (run_dir / "calibration_bootstrap_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "cluster_unit": "image_id",
            "point_estimates": {"delta_f1": point_f1, "delta_iou": point_iou, "delta_acc": point_acc},
            "bootstrap": {key: {"mean": float(np.mean(v)), "ci95_low": float(np.percentile(v, 2.5)), "ci95_high": float(np.percentile(v, 97.5))} for key, v in boot_metrics.items()},
        }, handle, indent=2, sort_keys=True)
    np.savez_compressed(run_dir / "calibration_bootstrap_repeats.npz",
                        **{key: np.asarray(value, dtype=np.float64) for key, value in boot_metrics.items()})
    print("\nwrote calibration_loco.csv / calibration_loco_summary.json / calibration_fully_supported*.csv / calibration_bootstrap_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
