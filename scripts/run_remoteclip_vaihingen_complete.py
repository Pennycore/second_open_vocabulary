"""Artifact-complete, one-shot RemoteCLIP Vaihingen rerun under the frozen protocol.

This entry point deliberately reuses the existing frozen RemoteCLIP Vaihingen
configuration and scoring primitives.  Its only purpose is to persist the
prediction-time float32 score path, every registered partial semantic map, and
the area-level evaluation assets that the earlier aggregate-only run omitted.
It never invokes SAM3, changes a method formula, or accepts protocol knobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_remoteclip_vaihingen_all import (
    CLASSES,
    TEST_AREAS,
    _aggregate,
    _area_from_id,
    _commit_hash,
    _encode_regions,
    _expected_ids,
    _gt_map,
    _load_candidates,
    _metrics_for_subset,
    _new_run_dir,
    _normalize,
    _preflight,
    _read_rgb,
    _write_csv_exclusive,
    _write_json_exclusive,
    _load_model,
    directory_sha256,
    pixel_confusion_fast,
    text_prototypes,
)

from ov_probe.io import InputValidationError, sha256_file
from ov_probe.pixel_ovss import IGNORE_INDEX, assemble_semantic_map, method_predictions, method_score_matrices


MINIMUM_METHODS = ("text_only", "C2", "CTP")
METRIC_FIELDS = ("OA", "macro_f1", "mIoU", "S_F1", "U_F1", "H_F1", "S_IoU", "U_IoU", "H_IoU")


def _write_npz_exclusive(path: Path, **arrays: np.ndarray) -> None:
    """Persist only explicit float32/label artifacts; never silently downcast."""
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _subset_definitions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset_index in range(1 << len(CLASSES)):
        mask = np.asarray([(subset_index >> index) & 1 for index in range(len(CLASSES))], dtype=bool)
        k = int(mask.sum())
        if k not in (2, 3, 4):
            continue
        rows.append({
            "subset": f"subset_{subset_index}", "subset_index": subset_index, "k": k,
            "mask": mask, "supported": [name for name, value in zip(CLASSES, mask) if value],
            "unsupported": [name for name, value in zip(CLASSES, mask) if not value],
        })
    if len(rows) != 25:
        raise InputValidationError("Frozen k=2/3/4 bitmask registration must contain 25 subsets.")
    return rows


def _coverage_mask(shape: tuple[int, int], regions: list[dict[str, Any]]) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for region in regions:
        mask = np.asarray(region["mask"], dtype=bool)
        x0, y0 = int(region["x0"]), int(region["y0"])
        covered[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] |= mask
    return covered


def _pixel_accounting(label_map: np.ndarray, gt: np.ndarray, covered: np.ndarray) -> dict[str, int]:
    if label_map.shape != gt.shape or label_map.shape != covered.shape:
        raise InputValidationError("Pixel-accounting inputs must share geometry.")
    gt_valid = gt != IGNORE_INDEX
    assigned = label_map != IGNORE_INDEX
    conflict = (~assigned) & covered
    uncovered = (~assigned) & (~covered)
    return {
        "pixels_total": int(label_map.size),
        "gt_valid_pixels": int(gt_valid.sum()),
        "assigned_pixels": int((assigned & gt_valid).sum()),
        "assigned_pixels_total": int(assigned.sum()),
        "valid_evaluated_pixels": int((assigned & gt_valid).sum()),
        "conflict_ignore_pixels": int((conflict & gt_valid).sum()),
        "conflict_ignore_pixels_total": int(conflict.sum()),
        "uncovered_pixels": int((uncovered & gt_valid).sum()),
        "uncovered_pixels_total": int(uncovered.sum()),
    }


def _metric_row(matrix: np.ndarray, subset: dict[str, Any]) -> dict[str, Any]:
    if not subset["unsupported"]:
        # Full support has no unsupported partition.  Preserve the ordinary
        # metrics but explicitly leave U/H fields undefined instead of writing
        # a NaN that could be mistaken for a numeric result.
        result = _aggregate([{"confusion_matrix": matrix.tolist()}])
        row: dict[str, Any] = {"OA": result["OA"], "macro_f1": result["macro_f1"], "mIoU": result["mIoU"]}
        row["S_F1"] = result["macro_f1"]
        row["S_IoU"] = result["mIoU"]
        row["U_F1"] = row["H_F1"] = None
        row["U_IoU"] = row["H_IoU"] = None
    else:
        result = _metrics_for_subset(matrix, subset["supported"], subset["unsupported"])
        row = {field: result[field] for field in METRIC_FIELDS}
    row["valid_pixels"] = result["valid_pixels"]
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        row[f"TP_{name}"] = tp
        row[f"FP_{name}"] = fp
        row[f"FN_{name}"] = fn
        row[f"F1_{name}"] = result["per_class_f1"][name]
        row[f"IoU_{name}"] = result["per_class_iou"][name]
    return row


def _write_confusion(path: Path, matrix: np.ndarray) -> None:
    tp = np.diag(matrix).astype(np.int64)
    fp = (matrix.sum(axis=0) - tp).astype(np.int64)
    fn = (matrix.sum(axis=1) - tp).astype(np.int64)
    _write_npz_exclusive(path, confusion_matrix=matrix.astype(np.int64), tp=tp, fp=fp, fn=fn,
                         class_names=np.asarray(CLASSES, dtype="U32"))


def _bootstrap(
    matrices: dict[int, dict[str, np.ndarray]], subset: dict[str, Any], seed: int = 42, repeats: int = 5000,
) -> dict[str, Any]:
    areas = list(TEST_AREAS)
    if sorted(matrices) != areas:
        raise InputValidationError("Bootstrap clusters differ from the frozen five test areas.")
    metric_names = METRIC_FIELDS

    def delta(drawn: list[int]) -> dict[str, float]:
        ctp = _metric_row(np.sum([matrices[area]["CTP"] for area in drawn], axis=0), subset)
        c2 = _metric_row(np.sum([matrices[area]["C2"] for area in drawn], axis=0), subset)
        return {name: float(ctp[name] - c2[name]) for name in metric_names}

    point = delta(areas)
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(repeats, dtype=np.float64) for name in metric_names}
    for index in range(repeats):
        drawn = [int(value) for value in rng.choice(areas, size=len(areas), replace=True)]
        values = delta(drawn)
        for name, value in values.items():
            samples[name][index] = value
    direction = {str(area): delta([area]) for area in areas}
    return {
        "comparison": "CTP_minus_C2",
        "cluster_unit": "Vaihingen test area",
        "areas": areas,
        "seed": seed,
        "repeats": repeats,
        "point_estimate": point,
        "bootstrap": {
            name: {"mean": float(values.mean()), "ci95_low": float(np.quantile(values, 0.025)),
                   "ci95_high": float(np.quantile(values, 0.975))}
            for name, values in samples.items()
        },
        "direction_by_area": direction,
        "positive_direction_count": {name: int(sum(values[name] > 0 for values in direction.values())) for name in metric_names},
        "note": "Five independent test areas are the clusters; resamples are not a large-sample significance claim.",
    }


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")}


def _report(output: Path, manifest: dict[str, Any], k_summary: list[dict[str, Any]], bootstrap: dict[str, Any]) -> None:
    lines = [
        "# RemoteCLIP Vaihingen artifact-complete partial-support rerun",
        "",
        "This is a single reproducibility rerun under the frozen RemoteCLIP protocol. It reuses the registered split, SAM3 candidate cache, checkpoint, prompts, prototype construction, CTP-v1, FusionCanvas and evaluation definitions. No training, tuning, or method change occurred.",
        "",
        "## Prediction artifact contract",
        "",
        "- Prediction artifacts were sealed in `manifest.json` before any GT label was opened.",
        "- `scores_float32/` stores the final region/class score arrays and predictions as float32/int64; no float16 score cache is used for evaluation reconstruction.",
        "- `semantic_maps/` contains full-support and every registered k=2/3/4 subset × test-area × method map.",
        "- `confusion_matrices/`, `per_area_metrics.csv`, and `pixel_accounting.csv` provide the per-area evidence assets.",
        "",
        "## Partial-support subset-level mean ± population std",
        "",
        "| k | Method | OA | Macro F1 | mIoU | S-IoU | U-IoU | H-IoU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in k_summary:
        lines.append(
            f"| {row['k']} | {row['method']} | {row['OA_mean']:.6f} ± {row['OA_std']:.6f} | "
            f"{row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | {row['mIoU_mean']:.6f} ± {row['mIoU_std']:.6f} | "
            f"{row['S_IoU_mean']:.6f} ± {row['S_IoU_std']:.6f} | {row['U_IoU_mean']:.6f} ± {row['U_IoU_std']:.6f} | {row['H_IoU_mean']:.6f} ± {row['H_IoU_std']:.6f} |")
    lines.extend([
        "",
        "## Area-cluster bootstrap",
        "",
        "Each registered subset has a CTP−C2 bootstrap entry in `bootstrap_summary.json` (seed 42; 5,000 resamples; five test-area clusters). Full-support has no supported/unsupported partition; partial support supplies S/U/H metrics.",
        "",
        f"- Completed run: `{manifest['run_id']}`",
        f"- Checkpoint SHA-256: `{manifest['sources']['checkpoint']['sha256']}`",
        f"- Candidate-cache SHA-256: `{manifest['sources']['candidates']['sha256']}`",
        "",
        "## Claim boundary",
        "",
        "The rerun evaluates one remote-sensing-specific CLIP backbone using the frozen protocol. Together with the original generic-CLIP study, it supports the bounded wording that the observed calibration effect is not specific to the original OpenAI CLIP backbone. It does not establish universal backbone independence.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _run(config: Path, output_root: Path) -> Path:
    preflight = _preflight(config)
    if preflight.status != "ready":
        raise InputValidationError("Preflight blocked: " + " | ".join(preflight.errors))
    cfg = preflight.config
    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    project_root = Path(preflight.sources["project_root"])
    try:
        output_root.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise InputValidationError("Artifact-complete output root must stay inside the second-paper project.") from exc
    run_dir = _new_run_dir(output_root, "run_")
    run_dir.mkdir(parents=True, exist_ok=False)
    log = (run_dir / "run.log").open("x", encoding="utf-8", newline="\n")
    try:
        def note(message: str) -> None:
            log.write(message + "\n")
            log.flush()
            print(message, flush=True)

        score_dir = run_dir / "scores_float32"
        semantic_dir = run_dir / "semantic_maps"
        confusion_dir = run_dir / "confusion_matrices"
        manifests_dir = run_dir / "manifests"
        for directory in (score_dir, semantic_dir, confusion_dir, manifests_dir):
            directory.mkdir(exist_ok=False)
        protocol = json.loads(paths["remoteclip_protocol_file"].read_text(encoding="utf-8"))
        import torch
        model, preprocess, tokenizer, torch = _load_model(paths["remoteclip_checkpoint"], protocol, "cuda")
        text_proto, token_hash = text_prototypes(model, tokenizer, protocol, "cuda", torch)
        if text_proto.shape != (len(CLASSES), cfg["integrity"]["required_feature_dimension"]):
            raise InputValidationError("RemoteCLIP text-prototype feature dimension differs from frozen protocol.")
        train_ids, test_ids, all_ids = _expected_ids(cfg)
        records: list[dict[str, Any]] = []
        all_features: list[np.ndarray] = []
        train_rows: list[dict[str, Any]] = []
        test_by_image: dict[str, tuple[tuple[int, int], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]] = {}
        for image_id in all_ids:
            shape, regions = _load_candidates(paths, image_id)
            image = _read_rgb(paths["image_dir"] / f"{image_id}_RGB.tif")
            if tuple(image.shape[:2]) != shape:
                raise InputValidationError(f"Image/candidate shape mismatch for {image_id}.")
            features = _encode_regions(image, regions, model, preprocess, "cuda", int(cfg["runtime"]["image_batch"]))
            split = "train" if image_id in train_ids else "test"
            local_rows: list[dict[str, Any]] = []
            for candidate_index, region in enumerate(regions):
                row = {"row_index": len(records), "image_id": image_id, "candidate_index": candidate_index,
                       "sam3_source_label": region["class_name"], "sam3_score": region["score"],
                       "x0": region["x0"], "y0": region["y0"], "split": split, "area": _area_from_id(image_id)}
                records.append(row); local_rows.append(row)
                if split == "train":
                    train_rows.append(row)
            all_features.append(features)
            if split == "test":
                test_by_image[image_id] = (shape, regions, local_rows, _coverage_mask(shape, regions))
            note(f"encoded {image_id}: {len(regions)} candidates")
        features = _normalize(np.concatenate(all_features, axis=0)).astype(np.float32, copy=False)
        visual = np.empty_like(text_proto, dtype=np.float32)
        prototype_counts: dict[str, int] = {}
        for class_index, class_name in enumerate(CLASSES):
            positions = [row["row_index"] for row in train_rows if row["sam3_source_label"] == class_name]
            if not positions:
                raise InputValidationError(f"No frozen train prototype regions for {class_name}.")
            visual[class_index] = _normalize(features[positions].mean(axis=0, keepdims=True))[0]
            prototype_counts[class_name] = len(positions)
        test_rows = [row for row in records if row["split"] == "test"]
        positions_by_row = {int(row["row_index"]): position for position, row in enumerate(test_rows)}
        test_positions = np.asarray([row["row_index"] for row in test_rows], dtype=np.int64)
        test_features = features[test_positions].astype(np.float32, copy=False)
        text_scores = (test_features @ text_proto.T).astype(np.float32, copy=False)
        visual_scores = (test_features @ visual.T).astype(np.float32, copy=False)
        fused = (0.5 * text_proto + 0.5 * visual).astype(np.float32, copy=False)
        normalizers = np.linalg.norm(fused, axis=1)
        if np.any(normalizers <= 1e-8):
            raise InputValidationError("Frozen C2 prototype has zero norm.")
        anchored = ((0.5 * text_scores + 0.5 * visual_scores) / normalizers[None, :]).astype(np.float32, copy=False)
        text_pred = np.argmax(text_scores, axis=1).astype(np.int64)
        with (run_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in test_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        _write_npz_exclusive(score_dir / "base_region_class_scores_float32.npz", test_features=test_features,
                             text_scores=text_scores, visual_scores=visual_scores, anchored_scores=anchored,
                             text_prototypes=np.asarray(text_proto, dtype=np.float32), visual_prototypes=visual,
                             text_prediction=text_pred)
        subset_defs = [{"subset": "full_support", "subset_index": (1 << len(CLASSES)) - 1, "k": len(CLASSES),
                        "mask": np.ones(len(CLASSES), dtype=bool), "supported": list(CLASSES), "unsupported": []}, *_subset_definitions()]
        prediction_paths: dict[str, dict[str, dict[str, str]]] = {}
        for subset in subset_defs:
            score_set = method_score_matrices(text_scores, visual_scores, anchored, subset["mask"], text_pred)
            prediction_set = method_predictions(score_set, text_pred, subset["mask"])
            for method in MINIMUM_METHODS:
                if score_set[method].dtype != np.float32:
                    score_set[method] = score_set[method].astype(np.float32)
            arrays: dict[str, np.ndarray] = {"support_mask": subset["mask"].astype(np.uint8)}
            for method in MINIMUM_METHODS:
                arrays[f"scores_{method}"] = np.asarray(score_set[method], dtype=np.float32)
                arrays[f"prediction_{method}"] = np.asarray(prediction_set[method], dtype=np.int64)
            _write_npz_exclusive(score_dir / f"{subset['subset']}_region_scores_float32.npz", **arrays)
            prediction_paths[subset["subset"]] = {}
            subset_semantic = semantic_dir / subset["subset"]
            subset_semantic.mkdir(exist_ok=False)
            for image_id, (shape, regions, local_rows, _) in test_by_image.items():
                indices = [positions_by_row[int(row["row_index"])] for row in local_rows]
                prediction_paths[subset["subset"]][image_id] = {}
                for method in MINIMUM_METHODS:
                    prediction = prediction_set[method][indices]
                    final_score = score_set[method][indices, prediction]
                    label_map, _ = assemble_semantic_map(shape, regions, prediction, final_score, CLASSES)
                    path = subset_semantic / f"{method}_{image_id}_semantic.npz"
                    _write_npz_exclusive(path, label_map=label_map.astype(np.uint8, copy=False))
                    prediction_paths[subset["subset"]][image_id][method] = str(path.relative_to(run_dir)).replace("\\", "/")
            note(f"sealed prediction artifacts: {subset['subset']} (k={subset['k']})")
        predict_manifest = {
            "format_version": 1, "run_id": run_dir.name, "status": "predict_completed", "scientific_evidence": False,
            "gt_read": False, "dataset": "ISPRS Vaihingen 2D Semantic Labeling", "test_areas": list(TEST_AREAS),
            "methods": list(MINIMUM_METHODS), "partial_support": {"counts": [2, 3, 4], "policy": "all_deterministic_bitmasks"},
            "code_commit": _commit_hash(), "runner_sha256": sha256_file(Path(__file__)), "config_sha256": preflight.sources["config_sha256"],
            "sources": preflight.sources, "environment": preflight.environment, "text_token_sha256": token_hash,
            "visual_prototype_counts": prototype_counts, "record_count": len(test_rows), "train_record_count": len(train_rows),
            "records_sha256": sha256_file(run_dir / "records.jsonl"), "prediction_paths": prediction_paths,
            "prediction_artifacts": _hash_tree(score_dir) | _hash_tree(semantic_dir),
            "dtype_path": {"source_region_features": "float32", "inference_features": "float32", "saved_region_class_scores": "float32", "FusionCanvas_final_scores": "float32", "saved_semantic_maps": "uint8"},
            "labels": {"read": False},
        }
        _write_json_exclusive(run_dir / "manifest.json", predict_manifest)
        _write_json_exclusive(manifests_dir / "prediction_manifest.json", predict_manifest)
        note("prediction phase sealed; opening GT solely for frozen evaluation")
        label_hash, label_count = directory_sha256(paths["label_dir"], "*_label.tif")
        gt_by_image = {image_id: _gt_map(paths["label_dir"] / f"{image_id}_label.tif") for image_id in test_ids}
        per_area_rows: list[dict[str, Any]] = []
        pixel_rows: list[dict[str, Any]] = []
        subset_aggregate: dict[str, dict[str, np.ndarray]] = {}
        bootstrap_payload: dict[str, Any] = {"format_version": 1, "seed": 42, "repeats": 5000, "cluster_unit": "test area", "comparisons": {}}
        for subset in subset_defs:
            subset_name = subset["subset"]
            subset_aggregate[subset_name] = {method: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64) for method in MINIMUM_METHODS}
            area_matrices: dict[int, dict[str, np.ndarray]] = {area: {} for area in TEST_AREAS}
            for image_id, (shape, _, _, covered) in test_by_image.items():
                area = _area_from_id(image_id)
                gt = gt_by_image[image_id]
                if gt.shape != shape:
                    raise InputValidationError(f"GT/prediction geometry differs for {image_id}.")
                for method in MINIMUM_METHODS:
                    semantic_path = run_dir / prediction_paths[subset_name][image_id][method]
                    with np.load(semantic_path, allow_pickle=False) as archive:
                        label_map = archive["label_map"]
                    result = pixel_confusion_fast(label_map, gt, CLASSES)
                    matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
                    area_matrices[area][method] = matrix
                    subset_aggregate[subset_name][method] += matrix
                    metrics = _metric_row(matrix, subset)
                    accounting = _pixel_accounting(label_map, gt, covered)
                    base = {"subset": subset_name, "subset_index": subset["subset_index"], "k": subset["k"], "supported": "|".join(subset["supported"]),
                            "unsupported": "|".join(subset["unsupported"]), "area": area, "image_id": image_id, "method": method}
                    per_area_rows.append(base | metrics)
                    pixel_rows.append(base | accounting)
                    confusion_subdir = confusion_dir / subset_name
                    confusion_subdir.mkdir(exist_ok=True)
                    _write_confusion(confusion_subdir / f"{method}_{image_id}_confusion.npz", matrix)
            if subset_name != "full_support":
                bootstrap_payload["comparisons"][subset_name] = _bootstrap(area_matrices, subset)
            note(f"evaluated {subset_name} (k={subset['k']})")
        per_subset_rows: list[dict[str, Any]] = []
        for subset in subset_defs:
            for method in MINIMUM_METHODS:
                per_subset_rows.append({"subset": subset["subset"], "subset_index": subset["subset_index"], "k": subset["k"],
                                        "supported": "|".join(subset["supported"]), "unsupported": "|".join(subset["unsupported"]), "method": method,
                                        **_metric_row(subset_aggregate[subset["subset"]][method], subset)})
        k_summary: list[dict[str, Any]] = []
        for k in (2, 3, 4):
            for method in MINIMUM_METHODS:
                rows = [row for row in per_subset_rows if row["k"] == k and row["method"] == method]
                summary: dict[str, Any] = {"k": k, "method": method, "subset_count": len(rows), "aggregation": "metric_per_subset_then_population_mean_std"}
                for metric in METRIC_FIELDS:
                    values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                    summary[f"{metric}_mean"] = float(values.mean()); summary[f"{metric}_std"] = float(values.std(ddof=0))
                k_summary.append(summary)
        per_area_fields = list(per_area_rows[0])
        _write_csv_exclusive(run_dir / "per_area_metrics.csv", per_area_fields, per_area_rows)
        _write_csv_exclusive(run_dir / "per_subset_metrics.csv", list(per_subset_rows[0]), per_subset_rows)
        _write_csv_exclusive(run_dir / "partial_summary_by_k.csv", list(k_summary[0]), k_summary)
        _write_csv_exclusive(run_dir / "pixel_accounting.csv", list(pixel_rows[0]), pixel_rows)
        _write_json_exclusive(run_dir / "bootstrap_summary.json", bootstrap_payload)
        metrics_payload = {"full_support": {row["method"]: row for row in per_subset_rows if row["subset"] == "full_support"},
                           "partial_support": {row["subset"]: {method: next(item for item in per_subset_rows if item["subset"] == row["subset"] and item["method"] == method) for method in MINIMUM_METHODS}
                                               for row in per_subset_rows if row["subset"] != "full_support"},
                           "summary_by_k": k_summary}
        _write_json_exclusive(run_dir / "metrics.json", metrics_payload)
        final = dict(predict_manifest)
        final.update({"status": "completed", "scientific_evidence": True, "gt_read": True,
                      "labels": {"read": True, "sha256": label_hash, "count": label_count}})
        _report(run_dir / "report.md", final, k_summary, bootstrap_payload)
        final["final_artifacts"] = _hash_tree(run_dir)
        _write_json_exclusive(run_dir / "hashes.json", {
            "format_version": 1,
            "prediction_artifacts": predict_manifest["prediction_artifacts"],
            "evaluation_artifacts": final["final_artifacts"],
            "note": "The manifest and this hashes file are excluded from the self-referential evaluation hash map.",
        })
        (run_dir / "manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        _write_json_exclusive(manifests_dir / "final_manifest.json", final)
        note("completed")
        return run_dir
    except Exception:
        failure = {"status": "failed", "scientific_evidence": False, "error": traceback.format_exc()}
        if not (run_dir / "failure.json").exists():
            _write_json_exclusive(run_dir / "failure.json", failure)
        raise
    finally:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot artifact-complete frozen RemoteCLIP Vaihingen rerun.")
    parser.add_argument("--config", type=Path, default=Path("configs/remoteclip_vaihingen_all_v0.yaml"))
    parser.add_argument("--output-root", type=Path, required=True, help="Project-relative fresh root for this new run.")
    parser.add_argument("--preflight", action="store_true", help="Validate frozen inputs only; do not create output.")
    args = parser.parse_args()
    try:
        cfg, root = None, None
        preflight = _preflight(args.config)
        if preflight.status != "ready":
            print(json.dumps({"status": "blocked", "reasons": preflight.errors, "outputs_created": False}, sort_keys=True))
            return 3
        target = args.output_root if args.output_root.is_absolute() else Path(preflight.sources["project_root"]) / args.output_root
        if args.preflight:
            print(json.dumps({"status": "ready", "outputs_created": False, "output_root": str(target)}, sort_keys=True))
            return 0
        output = _run(args.config, target)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "run_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
