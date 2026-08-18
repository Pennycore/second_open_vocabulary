"""Runner for the Vaihingen blind confirmation of SCC-v1.

    python scripts/run_vaihingen_blind.py --config <yaml> --phase predict
    python scripts/run_vaihingen_blind.py --config <yaml> --phase evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ov_probe.io import InputValidationError, seed_everything, sha256_file, write_json  # noqa: E402
from ov_probe.loveda_partial_support import (  # noqa: E402
    _c2_normalizers,
    _metrics,
    benchmark_all_subsets,
    guard_predictions,
    scc_scores,
)
from ov_probe.vaihingen_blind import (  # noqa: E402
    CLASSES,
    GT_COLOR_MAP,
    TEST_AREAS,
    TRAIN_AREAS,
    _area_from_id,
    _encode_regions,
    _load_candidates,
    _region_gt,
    _text_prototypes,
)
from ov_probe.openai_clip_visual_anchor import _normalize  # noqa: E402


def _read_image(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise InputValidationError(f"Expected HxWxC image, got {arr.shape} from {path}")
    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        lo = float(np.percentile(rgb, 1))
        hi = float(np.percentile(rgb, 99))
        rgb = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vaihingen blind SCC-v1 confirmation.")
    parser.add_argument("--config", required=True, help="Deployment YAML config.")
    parser.add_argument("--phase", required=True, choices=["predict", "evaluate"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    project_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("experiment", {}).get("overwrite") is not False:
        raise InputValidationError("Config must set experiment.overwrite=false.")
    seed_everything(args.seed)
    protocol = json.loads(Path(cfg["paths"]["protocol_file"]).read_text(encoding="utf-8"))
    protocol["sha256"] = hashlib.sha256(Path(cfg["paths"]["protocol_file"]).read_bytes()).hexdigest()

    run_root = Path(cfg["paths"]["output_root"]).resolve()
    image_dir = Path(cfg["paths"]["image_dir"]).resolve()
    label_dir_value = cfg["paths"].get("label_dir")
    label_dir = Path(label_dir_value).resolve() if label_dir_value else None
    candidates_dir = Path(cfg["paths"]["candidates_dir"]).resolve()
    checkpoint = Path(cfg["paths"]["openai_clip_checkpoint"]).resolve()

    if args.phase == "predict":
        if label_dir is not None:
            raise InputValidationError("Predict phase must not configure a GT label directory.")
        run_root.mkdir(parents=True, exist_ok=False)
        all_candidates = _load_candidates(candidates_dir)
        image_ids = sorted(all_candidates)
        # split by area
        train_ids = [i for i in image_ids if _area_from_id(i) in TRAIN_AREAS]
        test_ids = [i for i in image_ids if _area_from_id(i) in TEST_AREAS]
        if not train_ids or not test_ids:
            raise InputValidationError("Train/test area split is incomplete.")
        # load images
        import torch
        import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(protocol["model"]["architecture"], pretrained=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
        model.eval().to(device)
        text_protos, token_hash = _text_prototypes(protocol, checkpoint, device)

        # encode all regions (train + test); labels only from SAM3 weak ids
        feature_rows: dict[str, np.ndarray] = {}
        records: list[dict[str, Any]] = []
        row_index = 0
        for image_id in sorted(image_ids):
            image = _read_image(image_dir / f"{image_id}_RGB.tif")
            candidates = all_candidates[image_id]
            features = _encode_regions(image, candidates, model, preprocess, device)
            feature_rows[image_id] = features
            for index, candidate in enumerate(candidates):
                records.append({
                    "row_index": row_index,
                    "image_id": image_id,
                    "candidate_index": index,
                    "sam3_source_label": str(candidate["class_name"]),
                    "sam3_score": float(candidate["score"]),
                    "mask_area": int(candidate["mask"].sum()),
                    "area": _area_from_id(image_id),
                    "split": "train" if _area_from_id(image_id) in TRAIN_AREAS else "test",
                })
                row_index += 1
        features_all = np.concatenate([feature_rows[i] for i in sorted(feature_rows)], axis=0).astype(np.float16)
        row_indices = np.asarray([r["row_index"] for r in records], dtype=np.int64)

        # visual prototypes from train-area SAM3 weak labels only
        train_positions = [r["row_index"] for r in records if r["split"] == "train"]
        train_features = features_all[train_positions].astype(np.float32)
        visual_protos = np.empty((len(CLASSES), 512), dtype=np.float32)
        counts: dict[str, int] = {}
        for ci, name in enumerate(CLASSES):
            positions = [r["row_index"] for r in records if r["split"] == "train" and r["sam3_source_label"] == name]
            if not positions:
                raise InputValidationError(f"No train-area SAM3 candidates for class {name}.")
            rows = features_all[positions].astype(np.float32)
            counts[name] = len(positions)
            visual_protos[ci] = _normalize(rows.mean(axis=0, keepdims=True))[0]

        test_rows = [r for r in records if r["split"] == "test"]
        test_positions = np.asarray([r["row_index"] for r in test_rows], dtype=np.int64)
        regions = _normalize(features_all.astype(np.float32))
        text_scores = regions @ _normalize(text_protos).T
        visual_scores = regions @ _normalize(visual_protos).T
        normalizers = _c2_normalizers(text_protos, visual_protos)
        anchored = (0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]

        text_pred = np.argmax(text_scores, axis=1).astype(np.int16)
        visual_pred = np.argmax(visual_scores, axis=1).astype(np.int16)
        c2_pred = np.argmax(anchored, axis=1).astype(np.int16)

        # all 2^5 support subsets (SCC + guard), pre-registered enumeration
        subset_results = {}
        for subset_index in range(1 << len(CLASSES)):
            mask = np.asarray([(subset_index >> i) & 1 for i in range(len(CLASSES))], dtype=bool)
            scc = scc_scores(text_scores, anchored, mask)
            scc_pred = np.argmax(scc, axis=1)
            if not mask.any():
                scc_pred = text_pred
            u_idx = [i for i, flag in enumerate(mask) if not flag]
            guard = guard_predictions(text_pred, anchored, u_idx)
            subset_results[subset_index] = {
                "supported": [name for name, flag in zip(CLASSES, mask) if flag],
                "unsupported": [name for name, flag in zip(CLASSES, mask) if not flag],
                "scc_predictions": scc_pred.astype(np.int16),
                "guard_predictions": guard.astype(np.int16),
            }

        out = run_root / "predictions.npz"
        with out.open("xb") as handle:
            np.savez_compressed(
                handle,
                features=features_all,
                row_indices=row_indices,
                test_positions=test_positions,
                text_scores=text_scores.astype(np.float16),
                visual_scores=visual_scores.astype(np.float16),
                anchored_scores=anchored.astype(np.float16),
                text_prototypes=text_protos.astype(np.float16),
                visual_prototypes=visual_protos.astype(np.float16),
                text_pred=text_pred,
                visual_pred=visual_pred,
                c2_pred=c2_pred,
            )
        # per-subset predictions stored in one compressed archive
        subset_path = run_root / "subset_predictions.npz"
        with subset_path.open("xb") as handle:
            np.savez_compressed(handle, **{
                f"sub{idx}_scc": subset_results[idx]["scc_predictions"]
                for idx in subset_results
            }, **{
                f"sub{idx}_guard": subset_results[idx]["guard_predictions"]
                for idx in subset_results
            })
        with (run_root / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        subset_manifest = {
            str(idx): {
                "supported": subset_results[idx]["supported"],
                "unsupported": subset_results[idx]["unsupported"],
            }
            for idx in subset_results
        }
        with (run_root / "subset_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(subset_manifest, handle, indent=2, sort_keys=True)
        manifest = {
            "format_version": 1,
            "phase": "predict",
            "status": "completed",
            "scientific_evidence": True,
            "blind": protocol["blind"],
            "protocol": {"sha256": protocol["sha256"]},
            "model": protocol["model"],
            "text_token_sha256": token_hash,
            "visual_prototype_counts": counts,
            "train_areas": TRAIN_AREAS,
            "test_areas": TEST_AREAS,
            "records": {"count": len(records), "sha256": sha256_file(run_root / "records.jsonl")},
            "outputs": {
                "predictions": {"path": out.name, "sha256": sha256_file(out)},
                "subset_predictions": {"path": subset_path.name, "sha256": sha256_file(subset_path)},
                "subset_manifest": {"path": "subset_manifest.json", "sha256": sha256_file(run_root / "subset_manifest.json")},
            },
        }
        write_json(run_root / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    # ---------------- evaluate ----------------
    if label_dir is None or not label_dir.is_dir():
        raise InputValidationError("Evaluate phase requires the GT label directory.")
    predict_manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if predict_manifest.get("phase") != "predict" or predict_manifest.get("status") != "completed":
        raise InputValidationError("Predict-phase manifest is not a completed predict run.")
    pred_path = run_root / "predictions.npz"
    subset_path = run_root / "subset_predictions.npz"
    if predict_manifest["outputs"]["predictions"]["sha256"] != sha256_file(pred_path):
        raise InputValidationError("Prediction artifact changed since the predict phase.")
    if predict_manifest["outputs"]["subset_predictions"]["sha256"] != sha256_file(subset_path):
        raise InputValidationError("Subset prediction artifact changed since the predict phase.")
    with np.load(pred_path, allow_pickle=False) as archive:
        text_pred = archive["text_pred"].astype(np.int64)
        visual_pred = archive["visual_pred"].astype(np.int64)
        c2_pred = archive["c2_pred"].astype(np.int64)
        test_positions = archive["test_positions"].astype(np.int64)
        text_scores = archive["text_scores"].astype(np.float32)
        visual_scores = archive["visual_scores"].astype(np.float32)
        anchored = archive["anchored_scores"].astype(np.float32)
        text_protos = archive["text_prototypes"].astype(np.float32)
        visual_protos = archive["visual_prototypes"].astype(np.float32)
    with np.load(subset_path, allow_pickle=False) as archive:
        sub_scc = {int(name[3:].split("_")[0]): archive[name].astype(np.int64)
                   for name in archive.files if name.startswith("sub") and name.endswith("_scc")}
        sub_guard = {int(name[3:].split("_")[0]): archive[name].astype(np.int64)
                     for name in archive.files if name.startswith("sub") and name.endswith("_guard")}
    records = [json.loads(line) for line in (run_root / "records.jsonl").open(encoding="utf-8")]
    candidates = _load_candidates(candidates_dir)

    # derive GT for test regions only
    test_records = [r for r in records if r["split"] == "test"]
    label_cache: dict[str, np.ndarray] = {}
    gt_labels: list[str | None] = []
    for row in test_records:
        image_id = str(row["image_id"])
        if image_id not in label_cache:
            label_cache[image_id] = _read_image(label_dir / f"{image_id}_label.tif")
        candidate = candidates[image_id][int(row["candidate_index"])]
        mask = candidate["mask"]
        x0, y0 = candidate["x0"], candidate["y0"]
        crop = label_cache[image_id][y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        pixels = crop[mask]
        votes = {name: int((pixels == np.asarray(color, dtype=np.uint8)).all(axis=1).sum())
                 for name, color in GT_COLOR_MAP.items()}
        total = sum(votes.values())
        gt_labels.append(max(votes, key=lambda name: votes[name]) if total > 0 else None)

    labeled = np.asarray([gt is not None for gt in gt_labels], dtype=bool)
    gt_index = np.asarray([CLASSES.index(gt) for gt in gt_labels if gt is not None], dtype=np.int64)
    test_pos_labeled = test_positions[labeled]

    def eval_pred(pred_all: np.ndarray) -> dict:
        return _metrics(pred_all[test_pos_labeled], gt_index, CLASSES)

    results = {
        "text_only": eval_pred(text_pred),
        "visual_only": eval_pred(visual_pred),
        "C2": eval_pred(c2_pred),
        "SCC_k5_full": eval_pred(sub_scc[31]),
        "guard_k5_full": eval_pred(sub_guard[31]),
    }
    # partial-support: all subsets k=1..4
    subset_rows = []
    for idx in sorted(sub_scc):
        mask = np.asarray([(idx >> i) & 1 for i in range(len(CLASSES))], dtype=bool)
        k = int(mask.sum())
        if k in (0, 5):
            continue
        supported = [name for name, flag in zip(CLASSES, mask) if flag]
        unsupported = [name for name, flag in zip(CLASSES, mask) if not flag]
        for method, pred_all in (("SCC", sub_scc[idx]), ("guard", sub_guard[idx])):
            m = _metrics(pred_all[test_pos_labeled], gt_index, CLASSES)
            s_f1 = float(np.mean([m["per_class_f1"][c] for c in supported]))
            u_f1 = float(np.mean([m["per_class_f1"][c] for c in unsupported]))
            s_iou = float(np.mean([m["per_class_iou"][c] for c in supported]))
            u_iou = float(np.mean([m["per_class_iou"][c] for c in unsupported]))
            h_f1 = 2 * s_f1 * u_f1 / (s_f1 + u_f1) if s_f1 + u_f1 > 0 else 0.0
            h_iou = 2 * s_iou * u_iou / (s_iou + u_iou) if s_iou + u_iou > 0 else 0.0
            subset_rows.append([idx, k, "|".join(supported), "|".join(unsupported), method,
                                f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}", f"{m['macro_iou']:.6f}",
                                f"{s_f1:.6f}", f"{u_f1:.6f}", f"{h_f1:.6f}",
                                f"{s_iou:.6f}", f"{u_iou:.6f}", f"{h_iou:.6f}"])
    with (run_root / "vaihingen_subset_metrics.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("subset_index,k,supported,unsupported,method,OA,macro_f1,mIoU,S_F1,U_F1,H_F1,S_IoU,U_IoU,H_IoU\n")
        for row in subset_rows:
            handle.write(",".join(str(v) for v in row) + "\n")

    with (run_root / "vaihingen_overall.csv").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("method,OA,macro_f1,mIoU\n")
        for name, m in results.items():
            handle.write(f"{name},{m['accuracy']:.6f},{m['macro_f1']:.6f},{m['macro_iou']:.6f}\n")

    # bootstrap (image/area cluster)
    rng = np.random.default_rng(42)
    test_image_ids = [str(r["image_id"]) for r, keep in zip(test_records, labeled) if keep]
    images = sorted(set(test_image_ids))
    image_positions: dict[str, list[int]] = {}
    for position, image_id in enumerate(test_image_ids):
        image_positions.setdefault(image_id, []).append(position)
    boot = {"delta_f1_scc_vs_text": [], "delta_iou_scc_vs_text": [], "delta_oa_scc_vs_text": [],
            "delta_f1_scc_vs_c2": [], "delta_iou_scc_vs_c2": [], "delta_oa_scc_vs_c2": []}
    text_pred_l = text_pred[test_pos_labeled]
    c2_pred_l = c2_pred[test_pos_labeled]
    scc5_pred_l = sub_scc[31][test_pos_labeled]
    for _ in range(5000):
        chosen = rng.choice(images, size=len(images), replace=True)
        positions = np.concatenate([np.asarray(image_positions[img], dtype=np.int64) for img in chosen])
        gt_b = gt_index[positions]
        mt = _metrics(text_pred_l[positions], gt_b, CLASSES)
        mc = _metrics(c2_pred_l[positions], gt_b, CLASSES)
        ms = _metrics(scc5_pred_l[positions], gt_b, CLASSES)
        boot["delta_f1_scc_vs_text"].append(ms["macro_f1"] - mt["macro_f1"])
        boot["delta_iou_scc_vs_text"].append(ms["macro_iou"] - mt["macro_iou"])
        boot["delta_oa_scc_vs_text"].append(ms["accuracy"] - mt["accuracy"])
        boot["delta_f1_scc_vs_c2"].append(ms["macro_f1"] - mc["macro_f1"])
        boot["delta_iou_scc_vs_c2"].append(ms["macro_iou"] - mc["macro_iou"])
        boot["delta_oa_scc_vs_c2"].append(ms["accuracy"] - mc["accuracy"])
    boot_summary = {}
    for key, values in boot.items():
        arr = np.asarray(values)
        boot_summary[key] = {"mean": float(arr.mean()),
                             "ci95_low": float(np.percentile(arr, 2.5)),
                             "ci95_high": float(np.percentile(arr, 97.5))}
    with (run_root / "vaihingen_bootstrap_summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(boot_summary, handle, indent=2, sort_keys=True)

    summary = {
        "phase": "evaluate", "status": "completed",
        "predict_manifest_sha256": sha256_file(run_root / "manifest.json"),
        "predictions_sha256": predict_manifest["outputs"]["predictions"]["sha256"],
        "test_regions": {"total": len(test_records), "labeled": int(labeled.sum()), "unlabeled": int((~labeled).sum())},
        "overall": {name: {"OA": m["accuracy"], "macro_f1": m["macro_f1"], "mIoU": m["macro_iou"]} for name, m in results.items()},
        "bootstrap": boot_summary,
    }
    write_json(run_root / "evaluate_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nwrote vaihingen_overall.csv / vaihingen_subset_metrics.csv / vaihingen_bootstrap_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
